"""Tkinter GUI for Minesweeper RL.

Two modes:
  * Human    - you click cells to play.
  * AI Watch - load a trained model and watch the agent play, step-by-step
               or on auto-play.

Board size (rows / cols / mines) is set from inside the window, so the same
window works for any configuration. When watching the AI, the board size must
match the size the model was trained on (the network is sized to the board).

Run from the project root:
    python gui/app.py        (Windows)
    python3 gui/app.py       (macOS / Linux)
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.minesweeper_env import MinesweeperEnv

# PyTorch is only needed for AI mode. Import lazily so human mode works
# even if torch isn't installed.
try:
    from agent.dqn_agent import DQNAgent
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# Classic Minesweeper number colors.
NUM_COLORS = {
    1: "#1976d2", 2: "#388e3c", 3: "#d32f2f", 4: "#7b1fa2",
    5: "#c2185b", 6: "#0097a7", 7: "#424242", 8: "#9e9e9e",
}
HIDDEN_BG = "#c0c7d0"
REVEAL_BG = "#e9edf2"
MINE_BG = "#e57373"
AI_PICK_BG = "#fff59d"


class MinesweeperGUI:
    def __init__(self, root, rows=6, cols=6, n_mines=6):
        self.root = root
        self.rows, self.cols, self.n_mines = rows, cols, n_mines
        self.env = MinesweeperEnv(rows, cols, n_mines)
        self.agent = None
        self.agent_dims = None  # (rows, cols) the loaded model expects
        self.mode = "human"
        self.auto_job = None
        self.last_pick = None

        root.title("Minesweeper RL")
        root.configure(bg="#2b2f3a")

        self._build_controls()
        self._build_size_controls()
        self.board_frame = tk.Frame(root, bg="#2b2f3a")
        self.board_frame.grid(row=2, column=0, padx=12, pady=6)
        self._build_status()
        self._build_board()
        self.new_game()

    # ---------- UI construction ----------
    def _build_controls(self):
        bar = tk.Frame(self.root, bg="#2b2f3a")
        bar.grid(row=0, column=0, padx=12, pady=(12, 4), sticky="ew")

        def mkbtn(text, cmd, state="normal"):
            b = tk.Button(bar, text=text, command=cmd,
                          highlightbackground="#2b2f3a", state=state)
            b.pack(side="left", padx=3)
            return b

        mkbtn("New Game", self.new_game)
        self.mode_btn = mkbtn("Mode: Human", self.toggle_mode)
        self.step_btn = mkbtn("AI Step", self.ai_step, state="disabled")
        self.auto_btn = mkbtn("Auto Play", self.toggle_auto, state="disabled")
        mkbtn("Load Model", self.load_model)

    def _build_size_controls(self):
        bar = tk.Frame(self.root, bg="#2b2f3a")
        bar.grid(row=1, column=0, padx=12, pady=4, sticky="ew")

        def add_spin(label, init, lo, hi):
            tk.Label(bar, text=label, bg="#2b2f3a", fg="#dfe3ea").pack(side="left", padx=(6, 2))
            var = tk.IntVar(value=init)
            sp = tk.Spinbox(bar, from_=lo, to=hi, width=4, textvariable=var,
                            highlightbackground="#2b2f3a")
            sp.pack(side="left")
            return var

        self.rows_var = add_spin("Rows", self.rows, 2, 24)
        self.cols_var = add_spin("Cols", self.cols, 2, 24)
        self.mines_var = add_spin("Mines", self.n_mines, 1, 200)
        tk.Button(bar, text="Apply Size", command=self.apply_size,
                  highlightbackground="#2b2f3a").pack(side="left", padx=8)

    def _build_status(self):
        self.status = tk.Label(self.root, text="", bg="#2b2f3a", fg="#dfe3ea",
                               font=("Helvetica", 12), pady=8)
        self.status.grid(row=3, column=0, sticky="ew", pady=(0, 10))

    def _build_board(self):
        for w in self.board_frame.winfo_children():
            w.destroy()
        self.buttons = []
        for r in range(self.rows):
            row_btns = []
            for c in range(self.cols):
                idx = r * self.cols + c
                # Real Button widgets render and click reliably on macOS.
                b = tk.Button(self.board_frame, text="", width=2, height=1,
                              font=("Helvetica", 15, "bold"),
                              command=lambda i=idx: self.human_click(i))
                # highlightbackground sets the visible color on macOS (bg is ignored there).
                b.configure(bg=HIDDEN_BG, highlightbackground=HIDDEN_BG,
                            activebackground=HIDDEN_BG, disabledforeground="#000")
                b.grid(row=r, column=c, padx=1, pady=1)
                row_btns.append(b)
            self.buttons.append(row_btns)

    # ---------- size handling ----------
    def apply_size(self):
        try:
            rows = int(self.rows_var.get())
            cols = int(self.cols_var.get())
            mines = int(self.mines_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid size", "Rows, cols and mines must be whole numbers.")
            return
        max_mines = rows * cols - 9  # safe first click clears a 3x3 region
        if mines < 1 or mines > max_mines:
            messagebox.showerror(
                "Too many mines",
                f"For a {rows}x{cols} board, mines must be between 1 and {max_mines} "
                "(the first click and its neighbors are always safe).")
            return
        self.stop_auto()
        self.rows, self.cols, self.n_mines = rows, cols, mines
        self.env = MinesweeperEnv(rows, cols, mines)
        self._build_board()
        # A model trained on a different size can no longer be used.
        if self.agent and self.agent_dims != (rows, cols):
            self.agent = None
            self.agent_dims = None
            if self.mode == "ai":
                self.step_btn.config(state="disabled")
                self.auto_btn.config(state="disabled")
            self.set_status(f"Board set to {rows}x{cols}. Reload a model trained on this size.")
        self.new_game()

    # ---------- game flow ----------
    def new_game(self):
        self.stop_auto()
        self.state = self.env.reset()
        self.mask = self.env.valid_actions_mask()
        self.last_pick = None
        if self.mode == "human":
            self.set_status(f"{self.rows}x{self.cols}, {self.n_mines} mines - good luck!")
        else:
            self.set_status("Ready - AI Step or Auto Play." if self.agent
                            else "Load a model trained on this board size.")
        self.redraw()

    def toggle_mode(self):
        if self.mode == "human":
            self.mode = "ai"
            self.mode_btn.config(text="Mode: AI")
            ai_state = "normal" if self.agent else "disabled"
            self.step_btn.config(state=ai_state)
            self.auto_btn.config(state=ai_state)
            if not self.agent:
                self.set_status("Load a trained model to watch the AI.")
        else:
            self.mode = "human"
            self.mode_btn.config(text="Mode: Human")
            self.step_btn.config(state="disabled")
            self.auto_btn.config(state="disabled")
            self.stop_auto()
        self.new_game()

    def human_click(self, idx):
        if self.mode != "human" or self.env.done:
            return
        if self.env.revealed.ravel()[idx]:
            return
        self.apply_action(idx)

    def ai_step(self):
        if not self.agent or self.env.done:
            return
        action = self.agent.act(self.state, self.mask, epsilon=0.0)
        self.last_pick = action
        self.apply_action(action)

    def apply_action(self, idx):
        self.state, reward, done, info = self.env.step(idx)
        self.mask = self.env.valid_actions_mask()
        self.redraw()
        if done:
            self.set_status("Solved! Win." if info.get("win") else "Boom - hit a mine.")
            self.stop_auto()

    # ---------- auto play ----------
    def toggle_auto(self):
        if self.auto_job is not None:
            self.stop_auto()
        else:
            self.auto_btn.config(text="Stop")
            self._auto_tick()

    def _auto_tick(self):
        if self.env.done or not self.agent:
            self.stop_auto()
            return
        self.ai_step()
        if not self.env.done:
            self.auto_job = self.root.after(400, self._auto_tick)
        else:
            self.stop_auto()

    def stop_auto(self):
        if self.auto_job is not None:
            self.root.after_cancel(self.auto_job)
            self.auto_job = None
        self.auto_btn.config(text="Auto Play")

    # ---------- model loading ----------
    def load_model(self):
        if not TORCH_OK:
            messagebox.showerror("PyTorch missing",
                                 "PyTorch isn't installed, so AI mode is unavailable.\n"
                                 "Install it with:  pip install torch")
            return
        path = filedialog.askopenfilename(
            title="Select trained model (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        try:
            agent = DQNAgent(self.rows, self.cols)
            agent.load(path)
            self.agent = agent
            self.agent_dims = (self.rows, self.cols)
            self.set_status(f"Loaded: {os.path.basename(path)} "
                            f"(for {self.rows}x{self.cols} boards)")
            if self.mode == "ai":
                self.step_btn.config(state="normal")
                self.auto_btn.config(state="normal")
        except Exception as e:
            messagebox.showerror(
                "Load failed",
                f"Could not load this model for a {self.rows}x{self.cols} board.\n\n{e}\n\n"
                "Set Rows/Cols to match the size the model was trained on, click "
                "Apply Size, then load again.")

    # ---------- rendering ----------
    def _set_cell(self, b, text, color, fg):
        # Set both bg and highlightbackground so it looks right on every platform.
        b.config(text=text, bg=color, highlightbackground=color,
                 activebackground=color, fg=fg)

    def redraw(self):
        revealed = self.env.revealed.reshape(self.rows, self.cols)
        counts = self.env.counts.reshape(self.rows, self.cols)
        mines = self.env.mines.reshape(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                b = self.buttons[r][c]
                if revealed[r][c]:
                    if mines[r][c]:
                        self._set_cell(b, "*", MINE_BG, "#7f0000")
                    else:
                        n = int(counts[r][c])
                        self._set_cell(b, str(n) if n else "", REVEAL_BG,
                                       NUM_COLORS.get(n, "#000"))
                else:
                    bg = AI_PICK_BG if (idx == self.last_pick and not self.env.done) else HIDDEN_BG
                    self._set_cell(b, "", bg, "#000")
        if self.env.done and not self.env.won:
            for r in range(self.rows):
                for c in range(self.cols):
                    if mines[r][c] and not revealed[r][c]:
                        self._set_cell(self.buttons[r][c], "*", MINE_BG, "#7f0000")

    def set_status(self, text):
        self.status.config(text=text)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--mines", type=int, default=6)
    args = p.parse_args()

    root = tk.Tk()
    MinesweeperGUI(root, args.rows, args.cols, args.mines)
    root.mainloop()


if __name__ == "__main__":
    main()
