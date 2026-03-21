import numpy as np

from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.agents.heuristic import HeuristicAgent

def evaluate_heuristic(episodes: int = 20):
    env = TetrisEnv()
    agent = HeuristicAgent()

    rewards = []
    lines = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        total_lines = 0

        while not done:
            action = agent.select_action(env)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_lines += info.get("lines_cleared", 0)

        rewards.append(total_reward)
        lines.append(total_lines)

        print(f"Episode {episode + 1}, reward={total_reward:.2f}, lines={total_lines}")

    print("\n--- Heuristic Results ---")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average lines: {np.mean(lines):.2f}")

if __name__ == "__main__":
    evaluate_heuristic()
