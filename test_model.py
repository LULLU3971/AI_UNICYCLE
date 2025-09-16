# test_model.py
import gymnasium as gym
from stable_baselines3 import PPO
from environment import UnicycleEnv  # custom env 불러오기

def main():
    # 1. 환경 생성 (렌더링을 보려면 render_mode='human')
    env = UnicycleEnv(render_mode="human")

    # 2. 모델 불러오기
    model = PPO.load("final_model.zip", env=env)

    # 3. 평가 루프: 10 에피소드 실행
    num_episodes = 10
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        terminated, truncated = False, False
        total_reward = 0

        while not (terminated or truncated):
            # 정책에 따라 행동 선택
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # 보상 누적
            total_reward += reward

            # 환경 렌더링 (게임 창 띄우기)
            env.render()

        print(f"Episode {episode} | Total Reward: {total_reward:.2f}")

    # 4. 환경 종료
    env.close()

if __name__ == "__main__":
    main()
