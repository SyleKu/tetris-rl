from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO

from tetris_rl.env.tetris_env import TetrisEnv

def load_model(algorithm: str, model_path: str):
    algorithm = algorithm.lower()

    if algorithm == "dqn":
        return DQN.load(model_path)
    elif algorithm == "ppo":
        return PPO.load(model_path)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def evaluate(algorithm: str, model_path: str, episodes: int = 20, max_steps_per_episode: int | None = 1000):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = TetrisEnv()
    model = load_model(algorithm, model_path)

    rewards = []
    lines = []

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        total_lines = 0
        invalid_actions = 0
        step_count = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            if max_steps_per_episode is not None and step_count >= max_steps_per_episode:
                truncated = True
                break

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, terminated, truncated_env, info = env.step(action)
            truncated = truncated or truncated_env

            invalid_actions += int(info.get('invalid_action', False))
            total_reward += reward
            total_lines += info.get('lines_cleared', 0)
            step_count += 1

        rewards.append(total_reward)
        lines.append(total_lines)

        valid_actions = step_count - invalid_actions
        valid_ratio = valid_actions / step_count if step_count > 0 else 0.0

        print(
            f"Episode {episode + 1}: "
            f"reward={total_reward:.2f}, "
            f"lines={total_lines},"
            f"steps={step_count}, "
            f"valid={valid_actions}, "
            f"invalid={invalid_actions}, "
            f"valid_ratio={valid_ratio:.4f}, "
            f"terminated={terminated}, "
            f"truncated={truncated}"
        )

    print("\n--- Evaluation Results ---")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average lines: {np.mean(lines):.2f}")

    return rewards, lines

if __name__ == "__main__":
    evaluate(
        algorithm="dqn",
        model_path="./results/checkpoints/dqn_tetris.zip",
        episodes=20,
        max_steps_per_episode=2000
    )