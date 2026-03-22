import numpy as np

from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.agents.heuristic import HeuristicAgent

def evaluate_heuristic(episodes: int = 20, max_steps_per_episode: int | None = 1000):
    env = TetrisEnv()
    agent = HeuristicAgent()

    rewards = []
    lines = []

    for episode in range(episodes):
        obs, _ = env.reset()

        total_reward = 0.0
        total_lines = 0
        step_count = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            if max_steps_per_episode is not None and step_count >= max_steps_per_episode:
                truncated = True
                break

            action = agent.select_action(env)

            obs, reward, terminated, truncated_env, info = env.step(action)
            truncated = truncated or truncated_env

            total_reward += reward
            total_lines += info.get("lines_cleared", 0)
            step_count += 1

        rewards.append(total_reward)
        lines.append(total_lines)

        print(
            f"Episode {episode + 1}: "
            f"reward={total_reward:.2f}, "
            f"lines={total_lines},"
            f"steps={step_count}, "
            f"terminated={terminated}, "
            f"truncated={truncated}"
        )

    print("\n--- Heuristic Results ---")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average lines: {np.mean(lines):.2f}")

    return rewards, lines

if __name__ == "__main__":
    evaluate_heuristic(
        episodes=20,
        max_steps_per_episode=2000,
    )
