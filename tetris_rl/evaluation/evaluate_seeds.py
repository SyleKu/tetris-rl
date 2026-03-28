import numpy as np

from tetris_rl.evaluation.evaluate import evaluate



def evaluate_seeds(algorithm: str, model_path: list[str], episodes: int = 20, max_steps_per_episode: int | None = 2000):
    all_avg_rewards = []
    all_avg_lines = []

    for model_path in model_path:
        print(f"\n===Evaluating: {model_path} ===")
        result = evaluate(
            algorithm=algorithm,
            model_path=model_path,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
        )
        all_avg_rewards.append(result["avg_reward"])
        all_avg_lines.append(result["avg_lines"])

    mean_reward = float(np.mean(all_avg_rewards))
    std_reward = float(np.std(all_avg_rewards))

    mean_lines = int(np.mean(all_avg_lines))
    std_lines = int(np.std(all_avg_lines))

    print("\n--- Aggregate Results Across Seeds ---")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Average reward: {mean_reward:.2f}  ± {std_reward:.2f}")
    print(f"Average lines: {mean_lines:.2f}  ± {std_lines:.2f}")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_lines": mean_lines,
        "std_lines": std_lines,
        "per_seed_rewards": all_avg_rewards,
        "per_seed_lines": all_avg_lines,
    }



if __name__ == "__main__":
    model_paths = [
        "./results/checkpoints/dqn_expD_200000_seed0.zip",
        "./results/checkpoints/dqn_expD_200000_seed1.zip",
        "./results/checkpoints/dqn_expD_200000_seed2.zip",
        "./results/checkpoints/dqn_expD_300000_seed0.zip",
        "./results/checkpoints/dqn_expD_300000_seed1.zip",
        "./results/checkpoints/dqn_expD_300000_seed2.zip",
    ]

    evaluate_seeds(
        algorithm="dqn",
        model_path=model_paths,
        episodes=20,
        max_steps_per_episode=2000,
    )

    model_paths = [
        "./results/checkpoints/ppo_expD_200000_seed0.zip",
        "./results/checkpoints/ppo_expD_200000_seed1.zip",
        "./results/checkpoints/ppo_expD_200000_seed2.zip",
        "./results/checkpoints/ppo_expD_300000_seed0.zip",
        "./results/checkpoints/ppo_expD_300000_seed1.zip",
        "./results/checkpoints/ppo_expD_300000_seed2.zip",
    ]

    evaluate_seeds(
        algorithm="ppo",
        model_path=model_paths,
        episodes=20,
        max_steps_per_episode=2000,
    )
