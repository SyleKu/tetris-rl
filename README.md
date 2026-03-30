# Tetris-RL

Reinforcement Learning agent for Tetris using a custom Gymnasium environment.

## Showcase

The GIF below shows a rollout of a trained reinforcement learning agent in the final environment

- Environment: full tetromino set
- Observation: flattened board + one-hot tetromino
- Algorithm: DQN / PPO
- Training budget: 1'000'000 timesteps

### DQN

![Tetris RL Showcase](results/gifs/dqn_expD_1000000_seeds0.gif)

### PPO

![Tetris RL Showcase](results/gifs/ppo_expD_1000000_seeds0.gif)


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
├── results/
│   └── gifs/
├── tests/
└── tetris_rl/
    ├── agents/
    ├── env/
    ├── evaluation/
    ├── models/
    ├── training/
    └── visualization/
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

---

## Result Summary

The final experiments were conducted using **Experiment D**, which combines:

- Flattened board representation
- One-hot encoded current tetromino
- Shaped reward function (lines cleared, holes, height, bumpiness)

Each configuration was trained for **300'000 timesteps** and evaluated across **3 random seeds**.

--- 

### DQN Performance

| Metric            | Value              |
|-------------------|--------------------|
| Average Reward    | -3.35 ± 2.31       |
| Average Lines     | 0.07 ± 0.05        |

**Observations:**
- Occasional lines clears (rare but present)
- High variance across seeds
- Generally unstable learning behavior

---

### Performance

| Metric            | Value              |
|-------------------|--------------------|
| Average Reward    | 2.16 ± 11.76       |
| Average Lines     | 0.17 ± 0.24        |

**Observations:**
- Best-performing algorithm overall
- Able to clear multiple lines (up to 2 in a single episode)
- High variance: one strong seed, two weak ones
- Indicates partial learning of useful strategies

---

### Key Takeaways

- Reward shaping significantly improved learning compared to earlier experiments
- PPO outperformed DQN ub terns of:
  - Stability (in best case)
  - Ability to clear lines
- However, bot agents remain far below heuristic baseline performance

---

### Limitations

- Sparse rewards still dominate learning dynamics
- Agents struggle with long-term planning
- High sensitivity to random seed
- No consistent multi-line clearing strategy learned

--- 

### Conclusion

While the agents demonstrate **early signs of learning**, they do not yet achieve robust Tetris gameplay. 

This project highlights:
- The importance of reward design in reinforcement learning
- The difficulty of applying standard RL methods to combinatorial environments like Tetris
- The gap between learned policies and heuristic approaches

---

## Future Work

This project demonstrates initial learning behavior in Tetris using standard reinforcement learning algorithms. However, significant improvements are possible and planned. 

### Short-Term Improvements

- [X] Train longer runs (≥ 1M timesteps) to stabilize learning
- [ ] Tune reward shaping coefficients (especially holes and height penalties)
- [ ] Improve exploration strategies (e.g., epsilon schedules for DQN)
- [ ] Add evaluation over larger episode batches for more reliable metrics

---

### Architecture Improvements

- [ ] Revisit CNN-based approaches with better feature design
  - Current CNN experiments underperformed due to poor reward signal alignment
  - Future work: combine CNN with structured features (hybrid model) 

- [ ] Try alternative policies:
  - Dueling DQN
  - Double DQN
  - PPO with larger networks

---

### Environment Enhancements

- [ ] Extend observation space with engineered features:
  - columns heights
  - number of holes
  - bumpiness
- [ ] Add "next piece preview" to observation (closer to real Tetris)
- [ ] Normalize or scale features for better learning stability

---

### Advanced RL Approaches

- [ ] Implement **action masking** to prevent invalid moves
- [ ] Try **Curriculum Learning** (start with smaller board sizes)
- [ ] Investigate **imitation learning** from heuristic policies
- [ ] Explore **Monte Carlo Tree Search (MCTS)** or planning-based approaches

---

### Evaluation & Visualization

- [ ] Generate more gameplay visualizations (GIFs / videos)
- [ ] Track additional metrics:
  - survival time
  - max height
  - holes over time
- [ ] Compare against heuristic baseline quantitatively

---

### Long-Term Goal

The ultimate goal is to train an agent that:

- reliably clears lines 
- survives for extended periods
- develops structured placement strategies

This likely requires a combination of:
- better reward design
- richer state representations
- and more advanced RL or planning methods

---

### Personal Development Focus

This project will continue to serve as a playground to:

- deepen understanding of reinforcement learning dynamics
- experiment with model architectures and reward engineering
- bridge the gap between theoretical RL and practical performance

Future iterations will focus on **incremental improvements**, guided by experimental results rather than assumptions.
