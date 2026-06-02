# Minesweeper Reinforcement Learning

A Double DQN agent that learns to play Minesweeper from raw board observations,
using a fully-convolutional network and invalid-action masking.

## Why this design

Minesweeper is a partially-observable grid game where the action space equals the
number of cells. A few choices make it learnable:

- **Fully-convolutional Q-network.** The board is processed by 3x3 convolutions and
  a final 1x1 conv produces one Q-value per cell. This makes the policy translation-
  equivariant: the same local mine-counting logic applies anywhere on the board.
- **Invalid-action masking.** Already-revealed cells are never selected (set to -inf
  before argmax), both when acting and when bootstrapping the TD target.
- **Safe first click.** The first revealed cell and its neighbors are guaranteed mine-
  free, matching standard Minesweeper and avoiding unwinnable openings.
- **Reward shaping.** +1 win, -1 mine, small positive reward per newly revealed safe
  cell, small penalty for wasted moves. This densifies an otherwise sparse signal.
- **Double DQN + soft target updates** for stable value estimation.

## Layout

```
env/minesweeper_env.py   Gym-style environment (reset/step/render, flood-fill reveal)
agent/dqn_agent.py       CNN Q-network, replay buffer, Double DQN agent
training/train.py        Training loop with linear epsilon decay
training/evaluate.py     Greedy evaluation + win-rate measurement
```

## Usage

```bash
pip install -r requirements.txt

# Train (start small; difficulty scales steeply with board size / mine density)
python training/train.py --rows 6 --cols 6 --mines 6 --episodes 20000 --save model.pt

# Evaluate greedily
python training/evaluate.py --model model.pt --rows 6 --cols 6 --mines 6 --games 1000
```

## Observation encoding

Each cell is `-1` if unrevealed, or `0..8` (adjacent mine count) if revealed. The
network normalizes by 8 before the convolutions.

## Tuning notes

- Win rate on a 6x6 / 6-mine board reaches a useful level within ~20-50k episodes on CPU.
- For bigger boards, increase `episodes`, `buffer_size`, and slow the epsilon decay.
- Mine *density* matters more than absolute size; ~15% mines is a reasonable target.
