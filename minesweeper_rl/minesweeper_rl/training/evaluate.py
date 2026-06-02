"""Evaluate a trained agent and optionally print a sample game."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.dqn_agent import DQNAgent
from env.minesweeper_env import MinesweeperEnv


def evaluate(model_path, rows, cols, n_mines, games=1000, render_one=True):
    env = MinesweeperEnv(rows, cols, n_mines)
    agent = DQNAgent(rows, cols)
    agent.load(model_path)

    wins = 0
    for g in range(games):
        state = env.reset()
        mask = env.valid_actions_mask()
        while True:
            action = agent.act(state, mask, epsilon=0.0)
            state, _, done, info = env.step(action)
            mask = env.valid_actions_mask()
            if done:
                wins += int(bool(info.get("win")))
                break
        if render_one and g == 0:
            print("Sample finished board:\n" + env.render() + "\n")

    print(f"Win rate over {games} games: {100.0 * wins / games:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--mines", type=int, default=6)
    p.add_argument("--games", type=int, default=1000)
    args = p.parse_args()
    evaluate(args.model, args.rows, args.cols, args.mines, args.games)
