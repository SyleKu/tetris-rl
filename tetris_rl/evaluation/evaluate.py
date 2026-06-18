from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN

from tetris_rl.env.tetris_env import TetrisEnv


def load_model(algorithm: str, model_path: str):
    algorithm = algorithm.lower()

    if algorithm == "dqn":
        return DQN.load(model_path)
    elif algorithm == "ppo":
        # PPO checkpoints are trained with MaskablePPO (see train_ppo.py)
        return MaskablePPO.load(model_path)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _predict(model, obs, env):
    """Predict deterministically, supplying the action mask for maskable models."""
    if isinstance(model, MaskablePPO):
        action, _ = model.predict(
            obs, deterministic=True, action_masks=env.action_masks()
        )
    else:
        action, _ = model.predict(obs, deterministic=True)
    return int(action)


def evaluate(
    algorithm: str,
    model_path: str,
    episodes: int = 20,
    max_steps_per_episode: int | None = 2000,
    seed: int = 0,
):
    """Evaluate a checkpoint.

    Each episode ``i`` is reset with ``seed + i`` so the piece sequence is fixed
    and the reported metrics are reproducible across runs.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = TetrisEnv()
    model = load_model(algorithm, model_path)

    rewards = []
    lines = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        total_lines = 0
        step_count = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            if max_steps_per_episode is not None and step_count >= max_steps_per_episode:
                truncated = True
                break

            action = _predict(model, obs, env)

            obs, reward, terminated, truncated_env, info = env.step(action)
            truncated = truncated or truncated_env

            total_reward += reward
            total_lines += info.get('lines_cleared', 0)
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

    avg_reward = float(np.mean(rewards))
    avg_lines = float(np.mean(lines))

    print("\n--- Evaluation Results ---")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average lines: {avg_lines:.2f}")

    return {
        "avg_reward": avg_reward,
        "avg_lines": avg_lines,
        "rewards": rewards,
        "lines": lines,
    }


if __name__ == "__main__":
    from tetris_rl.config import DQN_EXPF, PPO_EXPF

    evaluate(
        algorithm="dqn",
        model_path=f"{DQN_EXPF.checkpoint_path(seed=0)}.zip",
        episodes=20,
        max_steps_per_episode=2000,
        seed=0,
    )
    evaluate(
        algorithm="ppo",
        model_path=f"{PPO_EXPF.checkpoint_path(seed=0)}.zip",
        episodes=20,
        max_steps_per_episode=2000,
        seed=0,
    )
