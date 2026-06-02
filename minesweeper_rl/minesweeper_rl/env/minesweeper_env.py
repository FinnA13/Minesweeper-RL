"""Minesweeper environment with a Gym-style API."""
from __future__ import annotations

import numpy as np

# Cell encoding in the *observation* (what the agent sees):
#   -1            : unrevealed
#    0..8         : revealed, value = adjacent mine count
# Internally we also track mines and revealed masks.

UNREVEALED = -1


class MinesweeperEnv:
    def __init__(self, rows: int = 9, cols: int = 9, n_mines: int = 10, seed: int | None = None):
        self.rows = rows
        self.cols = cols
        self.n_mines = n_mines
        self.n_cells = rows * cols
        self.rng = np.random.default_rng(seed)
        self._neighbors = self._precompute_neighbors()
        self.reset()

    # ---------- setup ----------
    def _precompute_neighbors(self):
        nb = [[] for _ in range(self.n_cells)]
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            nb[idx].append(nr * self.cols + nc)
        return nb

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mines = np.zeros(self.n_cells, dtype=bool)
        self.revealed = np.zeros(self.n_cells, dtype=bool)
        self.counts = np.zeros(self.n_cells, dtype=np.int8)
        self.done = False
        self.first_move = True
        self.won = False
        return self._obs()

    def _place_mines(self, safe_idx: int):
        # Never place a mine on the first clicked cell or its neighbors.
        forbidden = set(self._neighbors[safe_idx]) | {safe_idx}
        candidates = [i for i in range(self.n_cells) if i not in forbidden]
        mine_pos = self.rng.choice(candidates, size=self.n_mines, replace=False)
        self.mines[mine_pos] = True
        for i in range(self.n_cells):
            if not self.mines[i]:
                self.counts[i] = sum(self.mines[j] for j in self._neighbors[i])

    # ---------- observation ----------
    def _obs(self):
        obs = np.full(self.n_cells, UNREVEALED, dtype=np.float32)
        obs[self.revealed] = self.counts[self.revealed]
        return obs.reshape(self.rows, self.cols)

    def valid_actions_mask(self):
        # Only unrevealed cells are valid moves.
        return ~self.revealed

    # ---------- step ----------
    def _flood_reveal(self, idx: int):
        stack = [idx]
        while stack:
            cur = stack.pop()
            if self.revealed[cur]:
                continue
            self.revealed[cur] = True
            if self.counts[cur] == 0 and not self.mines[cur]:
                for nb in self._neighbors[cur]:
                    if not self.revealed[nb]:
                        stack.append(nb)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")

        if self.revealed[action]:
            # Wasted move on an already-revealed cell.
            return self._obs(), -0.3, False, {"invalid": True}

        if self.first_move:
            self._place_mines(action)
            self.first_move = False

        if self.mines[action]:
            self.revealed[action] = True
            self.done = True
            return self._obs(), -1.0, True, {"explosion": True}

        before = self.revealed.sum()
        self._flood_reveal(action)
        newly = self.revealed.sum() - before

        # Win when every non-mine cell is revealed.
        if self.revealed.sum() == self.n_cells - self.n_mines:
            self.done = True
            self.won = True
            return self._obs(), 1.0, True, {"win": True}

        # Small positive shaping per newly revealed safe cell.
        reward = 0.1 + 0.02 * newly
        return self._obs(), reward, False, {"revealed": int(newly)}

    def render(self):
        symbols = []
        for i in range(self.n_cells):
            if not self.revealed[i]:
                symbols.append(".")
            elif self.mines[i]:
                symbols.append("*")
            elif self.counts[i] == 0:
                symbols.append(" ")
            else:
                symbols.append(str(self.counts[i]))
        rows = ["".join(symbols[r * self.cols:(r + 1) * self.cols]) for r in range(self.rows)]
        return "\n".join(rows)
