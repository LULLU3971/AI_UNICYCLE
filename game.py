import pygame
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- 상수 정의 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 150, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
SKY_BLUE = (135, 206, 235)

# 물리 및 게임 상수
GROUND_Y = SCREEN_HEIGHT - 100
GRAVITY = 0.0020
PLAYER_TORQUE = 0.003
WOBBLE_STRENGTH = 0.0010
FRICTION = 0.995
SPEED_ACCEL = 0.1
SPEED_FRICTION = 0.98
JUMP_STRENGTH = 12
JUMP_GRAVITY = 0.4
ANGLE_SPEED_EFFECT = 0.03

OBSTACLE_SPAWN_INTERVAL_MIN = 400
OBSTACLE_SPAWN_INTERVAL_MAX = 750
MAX_OBSTACLES_IN_VIEW = 2 # 관측할 장애물 수

# --- 장애물 클래스 ---
# class Obstacle:
#     def __init__(self, x):
#         self.x = x
#         self.type = random.choice(['top', 'bottom'])
#         self.width = random.randint(60, 120)
        
#         if self.type == 'bottom':
#             self.height = random.randint(50, 150)
#             self.rect = pygame.Rect(self.x, GROUND_Y - self.height, self.width, self.height)
#         else: # top
#             self.height = random.randint(250, 400)
#             y_pos = random.randint(100, 200)
#             self.rect = pygame.Rect(self.x, y_pos, self.width, self.height)

#     def draw(self, screen, camera_offset_x):
#         draw_rect = self.rect.copy()
#         draw_rect.x -= camera_offset_x
#         pygame.draw.rect(screen, GREEN, draw_rect)

# --- 외발자전거 클래스 ---
class Unicycle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.angular_velocity = 0
        self.speed = 0
        self.y_velocity = 0
        self.is_on_ground = True
        
        self.wheel_radius = 40
        self.frame_length = 120
        self.seat_width = 40
        self.seat_height = 15

    def reset(self, x, y):
        self.__init__(x, y)

    def update(self, action: int): # 키보드 입력 대신 action을 받음
        # 1. 좌우 균형
        gravity_torque = GRAVITY * math.sin(self.angle)
        player_torque = 0
        if action == 3: player_torque = -PLAYER_TORQUE # Balance Left
        if action == 4: player_torque = PLAYER_TORQUE # Balance Right
        wobble_torque = random.uniform(-WOBBLE_STRENGTH, WOBBLE_STRENGTH)
        angular_acceleration = gravity_torque + player_torque + wobble_torque
        self.angular_velocity += angular_acceleration
        self.angular_velocity *= FRICTION
        self.angle += self.angular_velocity

        # 2. 전진/후진 속도
        angle_acceleration = ANGLE_SPEED_EFFECT * math.sin(self.angle)
        self.speed += angle_acceleration

        if action == 1: self.speed += SPEED_ACCEL # Forward
        if action == 2: self.speed -= SPEED_ACCEL # Backward
        self.speed *= SPEED_FRICTION
        self.x += self.speed

        # 3. 점프
        # if action == 5 and self.is_on_ground: # Jump
        #     self.y_velocity = -JUMP_STRENGTH
        #     self.is_on_ground = False
        
        self.y_velocity += JUMP_GRAVITY
        self.y += self.y_velocity

        if self.y >= GROUND_Y:
            self.y = GROUND_Y
            self.y_velocity = 0
            self.is_on_ground = True

    def get_seat_rect(self):
        seat_center_x = self.x + self.frame_length * math.sin(self.angle)
        seat_center_y = (self.y - self.wheel_radius) - self.frame_length * math.cos(self.angle)
        seat_rect = pygame.Rect(0, 0, self.seat_width, self.seat_height)
        seat_rect.center = (seat_center_x, seat_center_y)
        return seat_rect

    def get_wheel_rect(self):
        wheel_rect = pygame.Rect(0, 0, self.wheel_radius * 2, self.wheel_radius * 2)
        wheel_rect.center = (self.x, self.y - self.wheel_radius)
        return wheel_rect

    def draw(self, screen, center_x):
        wheel_pos_y = self.y - self.wheel_radius
        pygame.draw.circle(screen, BLACK, (int(center_x), int(wheel_pos_y)), self.wheel_radius, 5)
        
        frame_end_x = center_x + self.frame_length * math.sin(self.angle)
        frame_end_y = wheel_pos_y - self.frame_length * math.cos(self.angle)
        pygame.draw.line(screen, BLACK, (center_x, wheel_pos_y), (frame_end_x, frame_end_y), 8)
        
        seat_rect = self.get_seat_rect()
        seat_rect.centerx -= (self.x - center_x)
        pygame.draw.rect(screen, RED, seat_rect)

# --- 게임 로직을 관리하는 클래스 ---
class UnicycleGame:
    def __init__(self, render_mode='human'):
        pygame.init()
        self.render_mode = render_mode
        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Unicycle RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 50)
            self.small_font = pygame.font.Font(None, 36)
        
        self.unicycle = Unicycle(0, GROUND_Y)
        # self.obstacles = []
        self.score = 0
        self.game_over = False
        # self.next_obstacle_spawn_x = SCREEN_WIDTH
        self.reset()

    def reset(self):
        self.game_over = False
        self.score = 0
        self.unicycle.reset(0, GROUND_Y)
        # self.obstacles = []
        # self.next_obstacle_spawn_x = SCREEN_WIDTH
        # # 초기 장애물 생성
        # for i in range(MAX_OBSTACLES_IN_VIEW):
        #      self.obstacles.append(Obstacle(self.unicycle.x + SCREEN_WIDTH/2 + i * 500))
        #      self.next_obstacle_spawn_x = self.obstacles[-1].x + random.randint(OBSTACLE_SPAWN_INTERVAL_MIN, OBSTACLE_SPAWN_INTERVAL_MAX)


    def step(self, action):
        if self.game_over:
            return True

        self.unicycle.update(action)
        self.score = int(self.unicycle.x)

        # 충돌 처리
        # wheel_rect = self.unicycle.get_wheel_rect()
        # for obs in self.obstacles:
        #     if wheel_rect.colliderect(obs.rect):
        #         is_landing = obs.type == 'bottom' and self.unicycle.y_velocity >= 0 and abs(wheel_rect.bottom - obs.rect.top) < 15
        #         if is_landing:
        #             self.unicycle.y = obs.rect.top
        #             self.unicycle.y_velocity = 0
        #             self.unicycle.is_on_ground = True
        #         else:
        #             self.game_over = True; break
        
        if self.game_over: return True

        seat_rect = self.unicycle.get_seat_rect()
        # for obs in self.obstacles:
        #     if seat_rect.colliderect(obs.rect):
        #         self.game_over = True; break
        if seat_rect.bottom >= GROUND_Y:
            self.game_over = True

        # 장애물 관리
        # if self.unicycle.x > self.next_obstacle_spawn_x:
        #     self.obstacles.append(Obstacle(self.unicycle.x + SCREEN_WIDTH))
        #     self.next_obstacle_spawn_x += random.randint(OBSTACLE_SPAWN_INTERVAL_MIN, OBSTACLE_SPAWN_INTERVAL_MAX)
        
        # # 화면 밖 장애물 제거
        # self.obstacles = [obs for obs in self.obstacles if obs.rect.right > self.unicycle.x - SCREEN_WIDTH]

        return self.game_over

    def render(self):
        if self.render_mode != 'human':
            return

        self.screen.fill(SKY_BLUE)
        camera_offset_x = self.unicycle.x - SCREEN_WIDTH / 2
        
        # 바닥 및 마커
        pygame.draw.rect(self.screen, GRAY, (0, GROUND_Y, SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_Y))
        marker_interval = 150
        start_marker = int(camera_offset_x / marker_interval); end_marker = int((camera_offset_x + SCREEN_WIDTH) / marker_interval) + 1
        for i in range(start_marker, end_marker):
            marker_x = i * marker_interval - camera_offset_x
            pygame.draw.line(self.screen, DARK_GRAY, (marker_x, GROUND_Y), (marker_x, SCREEN_HEIGHT), 2)

        # for obs in self.obstacles: obs.draw(self.screen, camera_offset_x)
        self.unicycle.draw(self.screen, SCREEN_WIDTH / 2)

        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            over_text = self.font.render("GAME OVER", True, RED)
            over_rect = over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(over_text, over_rect)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

# --- 강화학습 환경 클래스 ---
class UnicycleEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.game = UnicycleGame(render_mode=render_mode)
        self.render_mode = render_mode

        # 행동 공간: 0:Nop, 1:Fwd, 2:Bwd, 3:Balance L, 4:Balance R, 5:Jump
        self.action_space = spaces.Discrete(6)

        # 관측 공간 (13개 변수, 정규화된 값)
        # unicycle: angle, angular_vel, speed, y_vel, y_pos
        # 2 * obstacles: rel_x, rel_y, width, height
        obs_size = 5 #+ MAX_OBSTACLES_IN_VIEW * 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        
        self.last_x = 0

    def _get_obs(self):
        u = self.game.unicycle
        
        # 1. 외발자전거 상태
        obs = [
            u.angle / (math.pi),
            u.angular_velocity / 2.0,
            u.speed / 20.0,
            u.y_velocity / 20.0,
            (u.y - GROUND_Y) / (SCREEN_HEIGHT/2)
        ]

        # 2. 장애물 상태
        # upcoming_obstacles = sorted([obs for obs in self.game.obstacles if obs.rect.right > u.x], key=lambda o: o.x)
        
        # for i in range(MAX_OBSTACLES_IN_VIEW):
        #     if i < len(upcoming_obstacles):
        #         o = upcoming_obstacles[i]
        #         obs.extend([
        #             (o.x - u.x) / SCREEN_WIDTH,
        #             (o.rect.top - GROUND_Y) / SCREEN_HEIGHT,
        #             o.width / SCREEN_WIDTH,
        #             o.height / SCREEN_HEIGHT
        #         ])
        #     else: # 장애물이 부족하면 0으로 채움
        #         obs.extend([0, 0, 0, 0])
        
        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    def _get_info(self):
        return {"score": self.game.score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.last_x = self.game.unicycle.x
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.last_x = self.game.unicycle.x
        
        terminated = self.game.step(action)
        
        # 보상 계산
        reward = 0
        # 1. 전진/후진 보상
        reward += (self.game.unicycle.x - self.last_x)
        # 2. 생존 보상
        reward += 0.1
        # 3. 게임오버 페널티
        if terminated:
            reward -= 100.0

        observation = self._get_obs()
        info = self._get_info()
        
        # 에피소드 조기 종료 (Truncated) 조건 (예: 너무 멀리 갔을 때)
        truncated = False
        if self.game.score > 10000:
            truncated = True

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            self.game.render()
        elif self.render_mode == 'rgb_array':
            # (구현 필요) Pygame surface를 numpy array로 변환
            pass

    def close(self):
        self.game.close()

# --- 메인 실행 함수 ---
def main():
    print("실행 모드를 선택하세요:")
    print("1: 직접 플레이하기 (Manual Play)")
    print("2: 강화학습 환경 테스트 (RL Env Test with Random Actions)")
    choice = '1' #input("선택 (1 또는 2): ")

    if choice == '1':
        # 기존의 수동 플레이 방식 (키보드 입력을 action으로 변환)
        game = UnicycleGame(render_mode='human')
        running = True
        while running:
            action = 0 # 기본은 Nop
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]: action = 1
            elif keys[pygame.K_s]: action = 2
            
            if keys[pygame.K_a]: action = 3
            elif keys[pygame.K_d]: action = 4

            # if keys[pygame.K_SPACE]: action = 5

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if game.game_over and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    game.reset()

            if not game.game_over:
                game.step(action)
            
            game.render()
        game.close()

    elif choice == '2':
        # Gymnasium 환경 테스트
        from gymnasium.utils.env_checker import check_env
        
        # 1. 환경 생성 및 체크
        env = UnicycleEnv(render_mode='human')
        # check_env(env) # 호환성 체크
        print("강화학습 환경이 생성되었습니다. 랜덤 행동으로 3 에피소드를 실행합니다.")

        # 2. 랜덤 행동 실행
        for episode in range(3):
            obs, info = env.reset()
            terminated, truncated = False, False
            total_reward = 0
            step_count = 0
            while not (terminated or truncated):
                action = env.action_space.sample() # 랜덤 행동 선택
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                env.render() # 화면에 렌더링
            print(f"에피소드 {episode+1}: 총 스텝 = {step_count}, 최종 점수 = {info['score']}, 총 보상 = {total_reward:.2f}")
        env.close()
    else:
        print("잘못된 선택입니다.")

if __name__ == '__main__':
    main()