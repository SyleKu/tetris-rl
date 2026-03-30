from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN, PPO

from tetris_rl.env.tetris_env import TetrisEnv

CELL_SIZE = 28
GRID_LINE_COLOR = (60, 60, 60)
EMPTY_CELL_COLOR = (245, 245, 245)
FILLED_CELL_COLOR = (70, 120, 220)
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (20, 20, 20)
PANEL_BG_COLOR = (235, 235, 235)

PANEL_HEIGHT = 80

def load_model(algorithm: str, model_path: str):
    algorithm = algorithm.lower()

    if algorithm == "dqn":
        return DQN.load(model_path)
    elif algorithm == "ppo":
        return PPO.load(model_path)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _get_font():
    """
    Try to load a nicer truetype font.
    Fall back to default if unavailable.
    :return:
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()

def render_board_with_overlay(
        grid: np.ndarray,
        step_count: int,
        total_reward: float,
        total_lines: int,
        cell_size: int = CELL_SIZE,
) -> np.ndarray:
    """
    Convert the Tetris board grid into an RGB image array with a top information panel.
    """
    height, width = grid.shape
    board_width_px = width * cell_size
    board_height_px = height * cell_size

    img_width = board_width_px
    img_height = PANEL_HEIGHT + board_height_px

    image = Image.new("RGB", (img_width, img_height), BG_COLOR)
    draw = ImageDraw.Draw(image)
    font = _get_font()

    # Draw info panel background
    draw.rectangle(
        [0, 0, img_width, PANEL_HEIGHT],
        fill=PANEL_BG_COLOR,
    )

    # Overlay text
    draw.text((10, 10), f"Step: {step_count}", fill=TEXT_COLOR, font=font)
    draw.text((10, 35), f"Reward: {total_reward:.2f}", fill=TEXT_COLOR, font=font)
    draw.text((180, 35), f"Lines: {total_lines}", fill=TEXT_COLOR, font=font)

    # Draw board
    board_y_offset = PANEL_HEIGHT

    for r in range(height):
        for c in range(width):
            x0 = c * cell_size
            y0 = board_y_offset + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            color = FILLED_CELL_COLOR if grid[r, c] == 1 else EMPTY_CELL_COLOR
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=GRID_LINE_COLOR)

    return np.array(image)

def generate_gif(
        algorithm: str,
        model_path: str,
        output_path: str,
        max_steps: int = 200,
        fps: int = 3,
        seed: int | None = None,
):

    env = TetrisEnv()
    model = load_model(algorithm, model_path)

    obs, _ = env.reset(seed=seed)

    frames = []
    step_count = 0
    total_reward = 0.0
    total_lines = 0
    terminated = False
    truncated = False

    # Initial frame
    frames.append(
        render_board_with_overlay(
            grid=env.board.grid,
            step_count=step_count,
            total_reward=total_reward,
            total_lines=total_lines,
        )
    )

    while not (terminated or truncated) and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        total_reward += reward
        total_lines += info.get("lines_cleared", 0)

        frames.append(
            render_board_with_overlay(
                grid=env.board.grid,
                step_count=step_count,
                total_reward=total_reward,
                total_lines=total_lines,
            )
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration= 1 / fps)

    print(f"Saved GIF to: {output_path}")
    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total lines: {total_lines}")

    env.close()

if __name__ == "__main__":
    generate_gif(
        algorithm="dqn",
        model_path="./results/checkpoints/dqn_expD_300000_seed2.zip",
        output_path="./results/gifs/dqn_expD_300000_seed2.gif",
        max_steps=200,
        fps=3,
        seed=0,
    )
