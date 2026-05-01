#!/usr/bin/env python3
"""Behavioral Cloning: train initial policy from planner trajectories.

Input: planner_trajectories/trajectories.npz
  - states: (N, 9) = torso + arm_7 + gripper
  - actions: (N, 10) = GR00T format [base(3), ee_delta(6), gripper(1)]

Policy: MLP that predicts action from state
  Input: 9D state
  Output: 10D action

Usage:
  python3 train_bc.py [--epochs 500] [--batch-size 256] [--lr 3e-4]
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


class TrajectoryDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.states = torch.from_numpy(data["states"])    # (N, 9)
        self.actions = torch.from_numpy(data["actions"])   # (N, 10)
        print(f"Loaded {len(self.states)} samples from {npz_path}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """Simple MLP policy for behavioral cloning."""
    def __init__(self, state_dim=12, action_dim=10, hidden=[512, 256, 128]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description="BC training from planner trajectories")
    parser.add_argument("--data", type=str,
                        default="planner_trajectories/trajectories.npz")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="bc_policy.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = TrajectoryDataset(args.data)

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {train_size}, Val: {val_size}")

    # Model
    policy = BCPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Normalize states and actions
    all_states = dataset.states
    all_actions = dataset.actions
    state_mean = all_states.mean(dim=0).to(device)
    state_std = all_states.std(dim=0).to(device)
    state_std = torch.clamp(state_std, min=1e-6)
    action_mean = all_actions.mean(dim=0).to(device)
    action_std = all_actions.std(dim=0).to(device)
    action_std = torch.clamp(action_std, min=1e-6)

    print(f"State mean: {state_mean.cpu().numpy()}")
    print(f"Action std: {action_std.cpu().numpy()}")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Train
        policy.train()
        train_loss = 0.0
        for states, actions in train_loader:
            states = (states.to(device) - state_mean) / state_std
            actions_norm = (actions.to(device) - action_mean) / action_std

            pred = policy(states)
            # Weighted loss: gripper action (dim 9) gets 10x weight for grasp/lift steps
            raw_actions = actions.to(device)
            is_grasp = (raw_actions[:, 9] > 0.5).float()  # gripper=1 steps
            sample_weight = 1.0 + 9.0 * is_grasp  # 1x for approach, 10x for grasp/lift
            per_sample_loss = ((pred - actions_norm) ** 2).mean(dim=-1)
            loss = (per_sample_loss * sample_weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(states)

        train_loss /= train_size

        # Validate
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, actions in val_loader:
                states = (states.to(device) - state_mean) / state_std
                actions_norm = (actions.to(device) - action_mean) / action_std

                pred = policy(states)
                loss = loss_fn(pred, actions_norm)
                val_loss += loss.item() * len(states)

        val_loss /= val_size

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": policy.state_dict(),
                "state_mean": state_mean.cpu(),
                "state_std": state_std.cpu(),
                "action_mean": action_mean.cpu(),
                "action_std": action_std.cpu(),
            }, args.output)

    print(f"\nDone! Best val_loss: {best_val_loss:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
