"""Training loop for the Minesweeper DQN agent."""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dqn_agent import DQNAgent
from env.minesweeper_env import MinesweeperEnv


def linear_epsilon(step, start=1.0, end=0.05, decay_steps=50_000):
    frac = min(step / decay_steps, 1.0)
    return start + frac * (end - start)


def train(rows=6, cols=6, n_mines=6, episodes=20_000, log_every=500,
          save_path="minesweeper_dqn.pt", seed=0):
    env = MinesweeperEnv(rows, cols, n_mines, seed=seed)
    agent = DQNAgent(rows, cols)
    print(f"Device: {agent.device}")

    step = 0
    win_hist = deque(maxlen=log_every)
    reward_hist = deque(maxlen=log_every)

    for ep in range(1, episodes + 1):
        state = env.reset()
        mask = env.valid_actions_mask()
        ep_reward = 0.0

        while True:
            eps = linear_epsilon(step)
            action = agent.act(state, mask, eps)
            next_state, reward, done, info = env.step(action)
            next_mask = env.valid_actions_mask()
            agent.buffer.push(state.copy(), action, reward, next_state.copy(),
                              float(done), next_mask.copy())
            agent.learn()

            state, mask = next_state, next_mask
            ep_reward += reward
            step += 1
            if done:
                win_hist.append(1 if info.get("win") else 0)
                break

        reward_hist.append(ep_reward)

        if ep % log_every == 0:
            wr = 100.0 * np.mean(win_hist) if win_hist else 0.0
            ar = np.mean(reward_hist) if reward_hist else 0.0
            print(f"ep {ep:6d} | eps {eps:.3f} | win% {wr:5.1f} | avg_reward {ar:6.2f}")

    agent.save(save_path)
    print(f"Saved model to {save_path}")
    return agent


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--mines", type=int, default=6)
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--save", type=str, default="minesweeper_dqn.pt")
    args = p.parse_args()
    train(args.rows, args.cols, args.mines, args.episodes, save_path=args.save)
