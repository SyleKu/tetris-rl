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


---

## Reward-Shaping Experiment A

### Motivation
Previous RL uns with the redesigned action space showed that both DQN and PPO
could select valid actions, but neither agent learned to clear lines.
The reward signal appeared to be dominated by penalties, while positive rewards
for line clearing were too sparse.

### Change

Adjusted reward shaping to make successful play more attractive:

- increased line clearing reward from `5.0` to `10.0`
- reduced penalties:
  - aggregate height: `0.02 -> 0.01`
  - holes: `0.02 -> 0.01`
  - bumpiness: `0.02 -> 0.01`
- added a small positive reward (`+0.1`) for a valid move

### Hypothesis
A denser and less punitive reward signal may help PPO/DQN discover line-clearing
behavior more easily.

### Planned evaluation
- train PPO for 50k timesteps
- evaluate with 20 episodes
- compare:
  - average reward
  - average lines
  - average episodes length



## Results — Rewards-Shaping Experiment A

### Summary

Reward shaping was adjusted to:
- increase line clearing reward
- reduce penalties for height, holes, and bumpiness
- add a small positive reward for valid moves

---

### DQN Results

| Timesteps | Avg Reward | Avg Lines | Steps  |
|-----------|------------|-----------|--------|
| 10k       | -14.99     | 0.00      | ~8–16  |
| 50k       | -21.10     | 0.00      | ~7–21  |
| 100k      | -32.51     | 0.05      | ~13–29 |

**Observations:**
- Improved reward compared to baseline
- Slight increase in episode length
- First occurrence of line clearing (rare)
- No stable strategy learned

---

### PPO Results

| Timesteps | Avg Reward | Avg Lines | Steps |
|-----------|------------|-----------|-------|
| 10k       | -18.03     | 0.00      | ~8–18 |
| 50k       | -11.05     | 0.00      | ~6–14 |
| 100k      | -11.09     | 0.00      | ~6–12 |

**Observations:**
- Significant improvement in reward values
- More stable behavior
- No line clearing observed
- Converges to local optimum (low-penalty board stated)

---

### Interpretation

- Reward shaping improved learning signal
- Agents learned to avoid bad board states
- However, line clearing remains too rare to be discovered reliably

---

### Key Insight

Agents optimize shaping rewards instead of the actual objective:
- PPO converges to safe but unproductive strategies
- DQN occasionally discovers line clearing via exploration

---

### Conclusion

Experiment A improves stability and reward but does not enable
learning of the core task (line clearing).

---

### Next steps

Introduce delta-based reward (Experiment B)
to provide feedback on board improvements after each move


---

## Reward-Shaping Experiment B (Delta reward)

### Motivation
Experiment A increased the reward for line clearing and reduced penalties,
but line clearing may still remain too sparse for early learning.

### Change
Instead of relying only on absolute board penalties, reward shaping now includes
feature delta between the board state before and after the move:

- reward improvements in:
  - aggregate height
  - holes
  - bumpiness
- keep strong positive reward for line clearing
- keep a small positive reward for valid moves

### Hypothesis
A delta-based reward may provide a denser learning signal and help PPO/DQN detect
whether a move improved or worsened the board, even before line clearing occurs.

### Planned evaluation

- train PPO for 50k timesteps
- evaluate with 20 episodes
- compare against Experiment A

---

## Results — Rewards-Shaping Experiment B (Delta Reward)

### Summary

Experiment B introduced delta-based reward shaping by comparing board features
before and after each move.

Reward components:
- strong reward for line clearing
- small reward for valid moves
- dense feedback based on changes in:
  - aggregate height
  - holes
  - bumpiness

### DQN Results

| Timesteps |  Avg Reward |  Avg Lines |  Steps |
|-----------|------------:|-----------:|-------:|
| 10k       |      -14.51 |       0.00 | ~12–24 |
| 50k       |      -18.36 |       0.00 | ~15–26 |
| 100k      |      -19.39 |       0.05 | ~14–26 |

**Observations:**

- DQB survives somewhat longer than in earlier experiments
- Very rare line clearing appears at 100k
- No stable improvement with more training

### PPO Results

| Timesteps |  Avg Reward |  Avg Lines |  Steps |
|-----------|------------:|-----------:|-------:|
| 10k       |      -13.72 |       0.00 |  ~8–19 |
| 50k       |      -10.45 |       0.15 |  ~9–23 |
| 100k      |      -10.03 |       0.00 |  ~7–15 |


**Observations:**

- PPO shows the best result at 50k with occasional line clearing
- However, this behavior is not stable
- At 100k, line clearing disappears again

### Interpolation

Experiment B provides denser learning feedback than Experiment A and lead to
slightly more promising behavior.

However:
- line clearing remains extremely rare
- neither DQN nor PPO learns a robust policy
- improvements are inconsistent and unstable

### Conclusion

Experiment B shows small progress over Experiment A, but it still does not solve
the core problem.

The agents may learn weak local improvements in board structure, but they do not
reliably discover or optimize the true objective: clearing lines.

### Next steps

Planned next step: **Experiment C**
- keep delta-based reward
- make line clearing much more dominant
- reduce risk of agents optimizing only proxy signals

---

## Rewards-Shaping Experiment C

### Motivation

Experiment B introduces delta-based reward shaping and produces denser feedback,
but agents still failed to learn reliable line clearing.
This suggests that the proxy signals (height, holes, bumpiness) remained too influential
compared to the actual task objective.

### Change
Experiment C keeps delta-based reward shaping, but makes line clearing much more dominant:

- increase line clearing reward from `10.0` to `50.0`
- reduce delta-based shaping weights:
  - `delta_height`: `0.05 -> 0.02`
  - `delta_holes`: `0.20 -> 0.10`
  - `delta_bumpiness`: `0.05 -> 0.02`
- keep small positive reward for valid moves (`+0.1`)

### Hypothesis
If line clearing becomes the dominant reward component while delta rewards remain as auxiliary guidance,
the agent may start discovering actual scoring behavior instead of only optimizing proxy board features.

### Planned evaluation
- train PPO for 50k timesteps
- train DQN for 50k timesteps
- compare:
  - average reward
  - average lines
  - average episode length

## Results — Rewards-Shaping Experiment C

### Summary

Experiment C kept delta-based reward shaping but made line clearing much more dominant.

Changes:
- increased line clearing reward strongly
- reduced auxiliary delta-based shaping weights
- kept a small reward for valid moves

### DQN Results

| Timesteps |  Avg Reward |  Avg Lines |  Steps |
|-----------|------------:|-----------:|-------:|
| 10k       |       -8.44 |       0.00 |  ~7–22 |
| 50k       |       -6.18 |       0.05 | ~17–26 |
| 100k      |      -10.29 |       0.00 | ~10–23 |

**Observations:**
- Slightly better rewards than previous experiments
- One rare line-clearing event at 50k
- No stable improvement with more training

### PPO Results

| Timesteps |  Avg Reward |  Avg Lines |  Steps |
|-----------|------------:|-----------:|-------:|
| 10k       |       -6.72 |       0.00 |  ~8–16 |
| 50k       |       -6.94 |       0.00 |  ~8–15 |
| 100k      |        0.79 |       0.15 |  ~8–19 |

**Observations:**
- PPO shows the strongest result so far
- Occasional line clearing appears at 100k
- However, line clearing remains rare and unstable

### Interpretation

Experiment C show slight progress compared to Experiments A and B,
especially for PPO. However, the improvement is till too weak to conclude
that the agents learned a robust line-clearing strategy.

The mian limitation is likely no longer the reward function alone.
Instead, the current observation space may be too compressed to represent the
board state well enough for RL.

### Conclusion

Reward shaping alone is not sufficient.

The next step should focus on improving state representation rather than
continuing to tune the reward function.

### Next step

Experiment D: richer observation space

Planned change:
- replace compact board statistics with a richer board representation
- e.g. flattened grid or per-column heights
- keep Experiment C reward shaping unchanged.

---

## Experiment D - Richer Observation Space

### Motivation
Experiments A-C suggest that reward shaping alone is not sufficient to produce
robust line-clearing behavior.

A likely bottleneck is the current observation space:
the board is represented only by aggregated statistics, which may be too compressed
to distinguish strategically different board states.

### Change
Replace compact statistics with a richer state representation:

- use `grid.flatten()` to expose the full board layout
- keep `piece_one_hot` to encode the current tetromino
- keep reward function from Experiment C unchanged

### Hypothesis
A richer observation space should give the agent enough spatial information
to learn placement strategies that support actual line clearing.

### Planned evaluation
- train PPO for 10k, 50k, and 100k timesteps
- evaluate average reward, average lines, and average episode length
- compare against Experiment C

## Results - Experiment D (Flattened Grid Observation)

### Summary

Experiment D replaced the compact board statistics with a richer state representation:
- full board encoded as `grid.flatten()`
- current piece encoded via `piece_one_hot`
- reward shaping from Experiment C kept unchanged

### Heuristic Baseline

- Average reward: ~40059
- Average lines: ~797
- Episodes consistently reach truncation limit (2000 steps)

This confirms that the environment remains fully solvable under the richer observation design.

### DQN Results

| Timesteps |  Avg Reward |  Avg Lines |  Steps |
|-----------|------------:|-----------:|-------:|
| 10k       |       -6.04 |       0.05 | ~14–32 |
| 50k       |       -6.91 |       0.00 |  ~7–16 |
| 100k      |       33.02 |       0.80 | ~15–34 |


**Observations:**
- DQN shows the strongest improvement so far
- At 100k, multiple episodes clear 1-4 lines
- This is the first experiment with repeated line-clearing behavior
- Performance is still far below heuristic baseline, but no longer near-zero

### PPO Results

| Timesteps |  Avg Reward |  Avg Lines |  Steps |
|-----------|------------:|-----------:|-------:|
| 10k       |       -7.26 |       0.00 |  ~7–16 |
| 50k       |       -7.37 |       0.00 |  ~5–16 |
| 100k      |       -7.54 |       0.00 | ~12–20 |


**Observations:**
- PPO does not improve under Experiment D
- No line clearing observed
- Reward remains almost constant across training durations

### Interpretation

Experiment D suggests that observation quality was major bottleneck.
Replacing compressed board features with the full board layout significantly improved DQN performance.

However:
- improvement is currently limited to DQN
- PPO remains stuck in a weak local strategy
- learned behavior is still unstable and far from robust play

### Conclusion

Experiment D is the first setup that yields clear non-trivial progress.
The richer observation space appears much more important than additional reward tuning.

### Next step

- continue with DQN under Experiment D
- test longer runs (200k / 300k)
- keep PPO as secondary comparison baseline
