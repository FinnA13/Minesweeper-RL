"""Double DQN agent with a small CNN and invalid-action masking."""
from __future__ import annotations

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "next_mask"])


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


class MinesweeperNet(nn.Module):
    """Per-cell Q-values. Output shape == board shape (one Q per action)."""

    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Conv2d(64, 1, 1)  # 1x1 conv -> one Q-value per cell

    def forward(self, x):
        # x: (B, 1, R, C). Normalize: unrevealed (-1) stays, counts scaled.
        x = x / 8.0
        x = self.conv(x)
        q = self.head(x)              # (B, 1, R, C)
        return q.view(x.size(0), -1)  # (B, R*C)


class DQNAgent:
    def __init__(self, rows, cols, lr=1e-3, gamma=0.99, buffer_size=50_000,
                 batch_size=256, tau=0.005, device=None):
        self.rows, self.cols = rows, cols
        self.n_actions = rows * cols
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.online = MinesweeperNet(rows, cols).to(self.device)
        self.target = MinesweeperNet(rows, cols).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def _prep(self, state):
        t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return t.view(1, 1, self.rows, self.cols)

    @torch.no_grad()
    def act(self, state, mask, epsilon: float):
        valid = np.flatnonzero(mask.ravel())
        if random.random() < epsilon:
            return int(np.random.choice(valid))
        q = self.online(self._prep(state)).cpu().numpy().ravel()
        q_masked = np.full_like(q, -np.inf)
        q_masked[valid] = q[valid]
        return int(np.argmax(q_masked))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(np.array([b.state for b in batch]), dtype=torch.float32,
                                 device=self.device).view(-1, 1, self.rows, self.cols)
        next_states = torch.as_tensor(np.array([b.next_state for b in batch]), dtype=torch.float32,
                                      device=self.device).view(-1, 1, self.rows, self.cols)
        actions = torch.as_tensor([b.action for b in batch], device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b.done for b in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.array([b.next_mask for b in batch]), dtype=torch.bool,
                                     device=self.device)

        q = self.online(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online(next_states)
            next_q_online[~next_masks] = -1e9
            next_actions = next_q_online.argmax(1, keepdim=True)
            next_q_target = self.target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q_target * (1 - dones)

        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        # Soft target update.
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * op.data)
        return loss.item()

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.online.load_state_dict(sd)
        self.target.load_state_dict(sd)
