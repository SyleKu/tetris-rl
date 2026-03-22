**Date:** 2026-03-22

**Experiment ID:** RL-Phase-1

Experiment Log — RL Phase (Action Space Redesign)
---

### Setup
- Environment: custom ```TetrisEnv```
- Pieces: ```I```, ```O```, ```T```
- Observation: board features + piece encoding
- Action space: valid placement enumeration (no invalid actions)
- Reward:
  - ```+5.0 * lines```
  - penalties for:
    - aggregate height
    - holes
    - bumpiness

---

## Baseline - Heuristic Agent

### Configuration

- episodes: 20
- max_steps_per_episode: 2000

### Results

- Average lines: ~797
- Average reward: ~2000+

### Interpretation

- Environment is solvable and stable
- Reward function is reasonable for good policies
- Action space redesign works correctly

---

## DQN Results

### DQN - 10k timesteps

- Avg reward: ~-26
- Avg lines: 0
- Steps: ~8-16
- Episodes terminate quickly

### DQN - 50k timesteps

- Avg reward: ~-48
- Avg lines: 0
- Steps: ~11-22
- Slightly longer survival, but worse reward

### DQN - 100k timesteps

- Avg reward: ~-56
- Avg lines: 0
- Steps: ~14-23
- No meaningful improvement

### Interpretation

- DQN learns to survive slightly longer
- Does not learn line clearing
- Increase training time alone does not solve the problem

---

## PPO Results

### PPO - 10k timesteps

- Avg reward: ~-38
- Avg lines: 0
- Steps: ~10-14

### PPO - 50k timesteps

- Avg reward: ~-19
- Avg lines: 0
- Steps: ~6-11

### Interpretation

- PPO performs slightly better than DQN
- Reward improves, but:
  - still no lines cleared
  - episodes remain short
- PPO also fails to discover meaningful strategy

---

## Key Observations

### 1. Action space redesign successful
   - No invalid actions
   - Stable environment behavior
### 2. Heuristic strongly outperforms RL
  - ~800 lines vs 0 lines
  - Indicates problem is learnable
### 3. RL agents fail to reach sparse rewards
  - No lines clearing observed
  - Rewards dominated by penalties
### 4. Longer training does not help (yet)
  - DQN stagnates or degrades
  - PPO improves reward slightly but no behavior

---

## Hypothesis

The current reward signal is too sparse and indirect:
- Positive reward (```lines```) is rare and hard to discover
- Negative shaping dominates early learning
- Agents learn to avoid bad states, but not to achieve good states

---

## Next Step

**Reward shaping refinement**

Planned experiment:

- Increase weight of line clearing
- Reduce penalty strength
- Potentially introduce:
  - intermediate rewards (e.g. board improvement)
  - or delta-based rewards

---

## Summary

- Environment and action space are now correct
- Heuristic demonstrates high achievable performance
- RL agents currently fail due to rewards signal limitations
- Next focus: **improving learning signal via reward shaping**
