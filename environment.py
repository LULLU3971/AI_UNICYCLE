# unicycle_env.py
# ------------------------------------------------------------
# Pygame unicycle game + a training-optimized Gymnasium Env.
# Key improvements vs. the original:
#   * Clear, normalized observation space (sin/cos angle for periodicity)
#   * Minimal, unambiguous discrete action set (no unused jump)
#   * Robust reward shaping (progress, balance, smoothness, time/efficiency)
#   * Proper terminal/”win” condition with big terminal rewards
#   * Deterministic seeding (wobble torque reproducibility for RL)
#   * rgb_array rendering support for video logging
# ------------------------------------------------------------

import math
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


# === Constants (mostly unchanged) ============================================
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 150, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
SKY_BLUE = (135, 206, 235)

GROUND_Y = SCREEN_HEIGHT - 100
# Physics tuned for stability + RL friendliness
GRAVITY = 0.0020
PLAYER_TORQUE = 0.003
WOBBLE_STRENGTH = 0.0010
FRICTION = 0.995
SPEED_ACCEL = 0.1
SPEED_FRICTION = 0.98
JUMP_STRENGTH = 12
JUMP_GRAVITY = 0.4
ANGLE_SPEED_EFFECT = 0.03

FPS = 60


# === Unicycle model ==========================================================
class Unicycle:
    """
    Simple inverted pendulum on a wheel with forward/back speed.
    State (continuous):
        x, y               : wheel center (y ~= GROUND_Y except transient)
        angle              : frame angle (0 = upright, + right, - left)
        angular_velocity
        speed              : horizontal speed (ẋ)
        y_velocity
    Rendering:
        Wheel + frame + seat rectangle (seat helps detect 'falling').
    """
    def __init__(self, x: float, y: float):
        self.wheel_radius = 40
        self.frame_length = 120
        self.seat_width = 40
        self.seat_height = 15
        self.reset(x, y)

    def reset(self, x: float, y: float):
        self.x = x
        self.y = y
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.speed = 0.0
        self.y_velocity = 0.0
        self.is_on_ground = True

    def update(self, action: int, rng_uniform):
        """
        Action mapping (Discrete(5)):
            0: No-op
            1: Move Forward  (accelerate +)
            2: Move Backward (accelerate -)
            3: Balance Left  (apply -torque)
            4: Balance Right (apply +torque)

        NOTE: We accept rng_uniform (callable) so the Env can seed randomness
        deterministically. This avoids non-reproducible wobble during training.
        """
        # --- 1) Balance / rotational dynamics ---
        gravity_torque = GRAVITY * math.sin(self.angle)
        player_torque = 0.0
        if action == 3:  # balance left
            player_torque = -PLAYER_TORQUE
        elif action == 4:  # balance right
            player_torque = PLAYER_TORQUE

        # Small random wobble to make policy robust
        wobble_torque = rng_uniform(-WOBBLE_STRENGTH, WOBBLE_STRENGTH)
        angular_accel = gravity_torque + player_torque + wobble_torque

        self.angular_velocity += angular_accel
        self.angular_velocity *= FRICTION
        self.angle += self.angular_velocity

        # --- 2) Horizontal dynamics ---
        # Angle couples into speed so leaning influences roll
        self.speed += ANGLE_SPEED_EFFECT * math.sin(self.angle)

        if action == 1:  # forward
            self.speed += SPEED_ACCEL
        elif action == 2:  # backward
            self.speed -= SPEED_ACCEL

        self.speed *= SPEED_FRICTION
        self.x += self.speed

        # --- 3) Vertical (mostly grounded; jump disabled by design) ---
        # We keep the simple vertical integrator so seat height changes when
        # the frame rotates, but the wheel remains in contact with ground.
        self.y_velocity += JUMP_GRAVITY
        self.y += self.y_velocity
        if self.y >= GROUND_Y:
            self.y = GROUND_Y
            self.y_velocity = 0.0
            self.is_on_ground = True

    # --- Geometry helpers for collision/fall detection & rendering -----------
    def get_seat_rect(self) -> pygame.Rect:
        seat_center_x = self.x + self.frame_length * math.sin(self.angle)
        seat_center_y = (self.y - self.wheel_radius) - self.frame_length * math.cos(self.angle)
        seat_rect = pygame.Rect(0, 0, self.seat_width, self.seat_height)
        seat_rect.center = (seat_center_x, seat_center_y)
        return seat_rect

    def get_wheel_rect(self) -> pygame.Rect:
        wheel_rect = pygame.Rect(0, 0, self.wheel_radius * 2, self.wheel_radius * 2)
        wheel_rect.center = (self.x, self.y - self.wheel_radius)
        return wheel_rect

    def draw(self, screen: pygame.Surface, center_x: float):
        wheel_pos_y = self.y - self.wheel_radius
        pygame.draw.circle(screen, BLACK, (int(center_x), int(wheel_pos_y)),
                           self.wheel_radius, 5)

        frame_end_x = center_x + self.frame_length * math.sin(self.angle)
        frame_end_y = wheel_pos_y - self.frame_length * math.cos(self.angle)
        pygame.draw.line(screen, BLACK, (center_x, wheel_pos_y),
                         (frame_end_x, frame_end_y), 8)

        seat_rect = self.get_seat_rect()
        # translate to camera space
        seat_rect.centerx -= (self.x - center_x)
        pygame.draw.rect(screen, RED, seat_rect)


# === Game container (no obstacles; 'loss' when seat hits ground) =============
class UnicycleGame:
    """
    Manages Pygame surfaces & the unicycle. Designed to work in two modes:
      - render_mode='human': visible display window
      - render_mode='rgb_array': offscreen Surface for video logging

    We always create a Surface (even in rgb_array) to allow frame capture.
    """
    def __init__(self, render_mode: str = "human"):
        pygame.init()
        self.render_mode = render_mode

        # Fonts are only needed in human mode; keep rgb_array lean
        self.font = None
        self.small_font = None

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Unicycle RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 50)
            self.small_font = pygame.font.Font(None, 36)
        else:
            # Offscreen surface for rgb_array
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()  # still used to throttle FPS

        self.unicycle = Unicycle(0.0, GROUND_Y)
        self.score = 0
        self.game_over = False
        self.reset()

    def reset(self):
        self.game_over = False
        self.score = 0
        self.unicycle.reset(0.0, GROUND_Y)

    def step(self, action: int, rng_uniform) -> bool:
        """
        Returns:
          game_over (bool): True if seat hits ground; else False
        """
        if self.game_over:
            return True

        self.unicycle.update(action, rng_uniform)
        self.score = int(self.unicycle.x)

        # 'Fall' condition: seat touches (or is under) ground line
        seat_rect = self.unicycle.get_seat_rect()
        if seat_rect.bottom >= GROUND_Y:
            self.game_over = True

        return self.game_over

    def render(self) -> Optional[np.ndarray]:
        """
        Draw a frame. If human, blit to display and return None.
        If rgb_array, return HxWx3 uint8 array.
        """
        self.screen.fill(SKY_BLUE)

        # Horizontal camera tracks unicycle; we draw the bike centered
        camera_offset_x = self.unicycle.x - SCREEN_WIDTH / 2

        # Ground & distance markers
        pygame.draw.rect(self.screen, GRAY,
                         (0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y))
        marker_interval = 150
        start_marker = int(camera_offset_x / marker_interval)
        end_marker = int((camera_offset_x + SCREEN_WIDTH) / marker_interval) + 1
        for i in range(start_marker, end_marker):
            mx = i * marker_interval - camera_offset_x
            pygame.draw.line(self.screen, DARK_GRAY, (mx, GROUND_Y),
                             (mx, SCREEN_HEIGHT), 2)

        # Draw unicycle centered horizontally
        self.unicycle.draw(self.screen, SCREEN_WIDTH / 2)

        # HUD (only in human mode to reduce rgb_array noise)
        if self.render_mode == "human" and self.font is not None:
            score_text = self.font.render(f"Score: {self.score}", True, BLACK)
            self.screen.blit(score_text, (10, 10))

            if self.game_over:
                over_text = self.font.render("GAME OVER", True, RED)
                over_rect = over_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
                self.screen.blit(over_text, over_rect)

            pygame.display.flip()
            self.clock.tick(FPS)
            return None
        else:
            # Return frame as rgb array (HxWx3)
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))  # Pygame is (W,H,3)
            self.clock.tick(FPS)
            return frame

    def close(self):
        pygame.quit()


# === Gymnasium Environment ===================================================
class UnicycleEnv(gym.Env):
    """
    Observation Design (normalized to [-1, 1]):
      We include everything needed to be Markov & control the system while
      avoiding angle wrap-around by using sin/cos for periodicity.

      obs = [
        sin(angle),                          # [-1, 1]
        cos(angle),                          # [-1, 1]
        angular_velocity / w_scale,          # roughly in [-1, 1]
        speed / v_scale,                     # roughly in [-1, 1]
        seat_clearance / clearance_scale,    # normalized 'height' margin
      ]

      - seat_clearance := (GROUND_Y - seat_bottom_y). Positive when safe.
        This teaches the agent to keep the seat away from ground (don’t fall).
      - Scales chosen from quick empirical ranges to clip rarely:
          w_scale = 0.5  (rad/step)  -> |ω| > 0.5 is “fast” rotation
          v_scale = 10.0 (px/step)   -> |v| > 10 is “very fast” roll
          clearance_scale = 150.0 (px)
        All values are clipped to [-1, 1] to stabilize learning.

    Action Space (Discrete(5)):
      0: No-op
      1: Forward (+accel)
      2: Backward (-accel)
      3: Balance Left  (-torque)
      4: Balance Right (+torque)

      Rationale: each discrete action maps 1:1 to a control primitive; this
      avoids ambiguous multi-button combos and makes exploration simpler.

    Reward (shaping + precise terminals):
      + Progress toward target distance:
          r_progress = kx * Δx       (kx=0.2)  -> encourages forward motion
      + Upright posture shaping:
          r_upright = - kθ * θ^2     (kθ=2.0)  -> penalizes tilt quadratically
      + Smoothness:
          r_smooth = - kω * ω^2      (kω=0.1)  -> discourages spinning
      + Time/efficiency penalty:
          r_time = - 0.001           -> reach target quickly; no idling
      + Control cost:
          r_ctrl = - 0.002 if action != 0 else 0

      Terminal rewards:
        * Win if x >= TARGET_X:  +100.0 and terminated=True
        * Lose if seat hits ground: -100.0 and terminated=True

      Truncation:
        * Max steps per episode (e.g., 5000) to cap horizons.

    Seeding:
      We seed Python's `random` so wobble torque is reproducible for RL runs.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        target_x: float = 3000.0,
        max_episode_steps: int = 5000,
        w_scale: float = 0.5,
        v_scale: float = 10.0,
        clearance_scale: float = 150.0,
        progress_coef: float = 0.2,
        tilt_coef: float = 2.0,
        spin_coef: float = 0.1,
        time_penalty: float = 0.001,
        control_penalty: float = 0.002,
    ):
        super().__init__()
        self.render_mode = render_mode or "rgb_array"
        self.game = UnicycleGame(render_mode=self.render_mode)

        # --- Spaces ----------------------------------------------------------
        # Observation vector length = 5 (see docstring)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        # Discrete actions as described above
        self.action_space = spaces.Discrete(5)

        # --- Reward/eps params ----------------------------------------------
        self.target_x = float(target_x)
        self.max_episode_steps = int(max_episode_steps)
        self.w_scale = float(w_scale)
        self.v_scale = float(v_scale)
        self.clearance_scale = float(clearance_scale)
        self.progress_coef = float(progress_coef)
        self.tilt_coef = float(tilt_coef)
        self.spin_coef = float(spin_coef)
        self.time_penalty = float(time_penalty)
        self.control_penalty = float(control_penalty)

        # Internal bookkeeping
        self._rng_seed = None
        self._step_count = 0
        self._last_x = 0.0

        # We expose a deterministic uniform sampler for the game's wobble
        # so we can control randomness from env.seed(...)
        self._rng = random.Random()

    # ---------- Helpers ------------------------------------------------------
    def _rng_uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def _seat_clearance(self) -> float:
        # Positive when the seat is above ground; <=0 means a fall
        seat_bottom = self.game.unicycle.get_seat_rect().bottom
        return float(GROUND_Y - seat_bottom)

    def _get_obs(self) -> np.ndarray:
        u = self.game.unicycle
        # sin/cos avoid angle wrap-around discontinuity at ±π
        s = math.sin(u.angle)
        c = math.cos(u.angle)

        # Normalize and clip to [-1, 1] with chosen scales (see class doc)
        ang_vel = np.clip(u.angular_velocity / self.w_scale, -1.0, 1.0)
        speed = np.clip(u.speed / self.v_scale, -1.0, 1.0)
        clearance = np.clip(self._seat_clearance() / self.clearance_scale, -1.0, 1.0)

        obs = np.array([s, c, ang_vel, speed, clearance], dtype=np.float32)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        u = self.game.unicycle
        return {
            "x": float(u.x),
            "speed": float(u.speed),
            "angle": float(u.angle),
            "angular_velocity": float(u.angular_velocity),
            "seat_clearance": float(self._seat_clearance()),
            "score": int(self.game.score),
            "step_count": int(self._step_count),
            "target_x": float(self.target_x),
        }

    # ---------- Gym API ------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # Seed Python RNG used in wobble torque (determinism)
            self._rng.seed(seed)
            self._rng_seed = int(seed)

        self.game.reset()
        self._step_count = 0
        self._last_x = self.game.unicycle.x

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action."

        self._step_count += 1
        prev_x = self.game.unicycle.x
        terminated = self.game.step(action, self._rng_uniform)

        # --- Reward shaping --------------------------------------------------
        u = self.game.unicycle
        dx = u.x - prev_x

        # Progress (scaled): encourages forward motion specifically
        r_progress = self.progress_coef * dx

        # Upright posture: quadratic penalty in angle
        r_upright = - self.tilt_coef * (u.angle ** 2)

        # Smoothness: penalize spinning quickly
        r_smooth = - self.spin_coef * (u.angular_velocity ** 2)

        # Time/efficiency: small negative each step to finish quickly
        r_time = - self.time_penalty

        # Action/energy penalty: discourage twitchy control
        r_ctrl = - self.control_penalty if action != 0 else 0.0

        reward = r_progress + r_upright + r_smooth + r_time + r_ctrl

        # --- Terminal conditions --------------------------------------------
        win = (u.x >= self.target_x)
        if win:
            terminated = True
            reward += 100.0  # big success bonus

        # Game-over (fall) comes from the game logic
        if terminated and not win:
            reward -= 100.0  # harsh failure

        # Truncation by step cap
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        frame = self.game.render()
        if self.render_mode == "rgb_array":
            return frame  # HxWx3 array
        # human mode returns None (drawn to window)

    def close(self):
        self.game.close()


# === Manual & smoke test =====================================================
def _manual_play():
    """
    Simple interactive mode:
      W: forward, S: backward, A: balance left, D: balance right
      R: restart after fall, ESC/close window: quit
    """
    game = UnicycleGame(render_mode="human")
    running = True
    while running:
        action = 0  # default no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = 1
        elif keys[pygame.K_s]:
            action = 2

        if keys[pygame.K_a]:
            action = 3
        elif keys[pygame.K_d]:
            action = 4

        if game.game_over and keys[pygame.K_r]:
            game.reset()

        if not game.game_over:
            game.step(action, random.uniform)

        game.render()

    game.close()


def _random_rollout(render_mode="rgb_array", episodes: int = 1, seed: int = 0):
    env = UnicycleEnv(render_mode=render_mode)
    total = 0.0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        trunc = False
        ep_ret = 0.0
        while not (done or trunc):
            a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            ep_ret += r
            if render_mode == "human":
                env.render()
        total += ep_ret
        print(f"[Random] ep={ep+1} steps={info['step_count']} x={info['x']:.1f} "
              f"score={info['score']} return={ep_ret:.2f}")
    env.close()


if __name__ == "__main__":
    # Toggle one of the following quick tests.
    # 1) Manual play (visible window):
    # _manual_play()

    # 2) Random policy smoke test (rgb_array by default; no window):
    _random_rollout(render_mode="rgb_array", episodes=1, seed=42)
