# Tetris-RL

Reinforcement Learning agent for Tetris using a custom Gymnasium environment.

## Project Goals

- Build a custom Tetris environment compatible with Gymnasium
- Implement a heuristic baseline for comparison
- Train reinforcement learning agents on the environment
- Evaluate the effects of reward shaping and state representation
- Document results in a reproducible and research-oriented way

## Research Questions

- Can a reinforcement learning agent learn effective Tetris strategies?
- Does a feature-based state representation work better than a raw board representation?
- How does reward shaping affect training stability?
- How do DQN and PPO compare on this task?

## Planned Methods

- Custom Tetris environment
- Feature-based state representation
- Heuristic baseline
- DQN
- PPO

## Repository Structure

```text
tetris-rl/
├── configs/
├── docs/
├── tetris_rl/
│   ├── env/
│   ├── agents/
│   ├── training/
│   ├── evaluation
│   └── utils/
├── tests/
├── scripts/
└── results/
```

---

## Current Status

- [x] Define tetromino pieces
- [x] Implement board representation
- [X] Feature extraction (height, holes, bumpiness)
- [X] Implement Gymnasium environment
- [X] Add heuristic baseline
- [X] Train DQN agent
- [X] Train PPO agent
- [X] Evaluate performance
- [] Add training plots

---

# Installation

```bash
# Clone repository
git clone https://github.com/SyleKu/tetris-rl.git
cd tetris-rl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt 
```
