import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np

from tetris_rl.config import ObservationConfig
from tetris_rl.evaluation.evaluate import evaluate


class _Tee:
    """Write to several streams at once (e.g. terminal + file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(output_path: str):
    """Duplicate everything printed to stdout into ``output_path`` as well."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with open(output_path, "w") as file:
        sys.stdout = _Tee(original_stdout, file)
        try:
            yield
        finally:
            sys.stdout = original_stdout

def evaluate_seeds(algorithm: str, model_paths: list[str], episodes: int = 20, max_steps_per_episode: int | None = 2000, seed: int = 0, observation_config: ObservationConfig | None = None):
    per_model_rows: list[dict] = []
    all_avg_rewards: list[float] = []
    all_avg_lines: list[float] = []

    for model_path in model_paths:
        print(f"\n===Evaluating: {model_path} ===")

        result = evaluate(
            algorithm=algorithm,
            model_path=model_path,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            seed=seed,
            observation_config=observation_config,
        )

        avg_reward = float(result["avg_reward"])
        avg_lines = float(result["avg_lines"])

        all_avg_rewards.append(avg_reward)
        all_avg_lines.append(avg_lines)

        per_model_rows.append({
            "model": _basename_without_extension(model_path),
            "avg_reward": f"{avg_reward:.2f}",
            "avg_lines": f"{avg_lines:.2f}",
        })

    mean_reward = float(np.mean(all_avg_rewards))
    std_reward = float(np.std(all_avg_rewards))
    mean_lines = float(np.mean(all_avg_lines))
    std_lines = float(np.std(all_avg_lines))

    _print_results_table(
        per_model_rows,
        title=f"{algorithm.upper()} Seed Results",
    )

    summary_rows = [
        {
            "model": "Mean ± Std",
            "avg_reward": f"{mean_reward:.2f} ± {std_reward:.2f}",
            "avg_lines": f"{mean_lines:.2f} ± {std_lines:.2f}",
        }
    ]

    _print_results_table(
        summary_rows,
        title=f"{algorithm.upper()} Aggregate Results Across Seeds",
    )

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_lines": mean_lines,
        "std_lines": std_lines,
        "per_seed_rewards": all_avg_rewards,
        "per_seed_lines": all_avg_lines,
    }

def find_model_paths(prefix: str, checkpoint_dir: str = "./results/checkpoints") -> list[str]:
    base = Path(checkpoint_dir)
    return sorted(str(p) for p in base.glob(f"{prefix}_seed*.zip"))

def collect_model_paths(prefixes: list[str], checkpoint_dir: str = "./results/checkpoints") -> list[str]:
    model_paths: list[str] = []

    for prefix in prefixes:
        model_paths.extend(find_model_paths(prefix, checkpoint_dir))

    return model_paths

def _basename_without_extension(path: str) -> str:
    return Path(path).stem

def _print_results_table(rows: list[dict], title: str) -> None:
    if not rows:
        print(f"\n--- {title} ---")
        print("No rows to display.")
        return

    columns = ["model", "avg_reward", "avg_lines"]
    headers = {
        "model": "Model",
        "avg_reward": "Average Reward",
        "avg_lines": "Average Lines",
    }

    widths = {}
    for col in columns:
        max_content = max(len(str(row[col])) for row in rows)
        widths[col] = max(len(headers[col]), max_content)

    def fmt_row(values: dict[str, str]) -> str:
        return (
            f"{str(values['model']).ljust(widths['model'])} | "
            f"{str(values['avg_reward']).rjust(widths['avg_reward'])} | "
            f"{str(values['avg_lines']).rjust(widths['avg_lines'])}"
        )

    print(f"\n--- {title} ---")
    print(
        f"{headers['model'].ljust(widths['model'])} | "
        f"{headers['avg_reward'].rjust(widths['avg_reward'])} | "
        f"{headers['avg_lines'].rjust(widths['avg_lines'])}"
    )
    print(
        f"{'-' * widths['model']}-+-"
        f"{'-' * widths['avg_reward']}-+-"
        f"{'-' * widths['avg_lines']}"
    )

    for row in rows:
        print(fmt_row(row))

if __name__ == "__main__":
    from tetris_rl.config import DQN_EXPH, PPO_EXPH

    # Configs to evaluate. Observation comes from each config so the env matches
    # what the checkpoint was trained with. Swap to DQN_EXPF/G / PPO_EXPF/G to
    # evaluate earlier experiments instead.
    configs = (DQN_EXPH, PPO_EXPH)
    output_path = "results/eval_results.txt"

    with tee_stdout(output_path):
        for cfg in configs:
            model_paths = find_model_paths(cfg.checkpoint_prefix())
            if not model_paths:
                print(f"\nNo checkpoints found for {cfg.checkpoint_prefix()}_seed*.zip — skipping.")
                continue

            evaluate_seeds(
                algorithm=cfg.algo,
                model_paths=model_paths,
                episodes=20,
                max_steps_per_episode=2000,
                seed=0,
                observation_config=cfg.observation,
            )

    print(f"\nSaved evaluation results to: {output_path}")
