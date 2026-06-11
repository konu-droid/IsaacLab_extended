"""
Inspect the latest ik_pnp wandb run's reward curves (temporary analysis helper).

Pulls the most recent run tagged ``ik_pnp`` from the konu-droid/LerobotPick
project and prints first/middle/last means of each reward/state metric so
learning trends are visible from the terminal.
"""

import sys
import wandb


def summarize_history(history, keys):
    """
    Print start/middle/end averages for each metric key.

    Args:
        history: List of row dicts from run.scan_history().
        keys:    Metric names to summarize.
    """
    n = len(history)
    if n < 10:
        print(f"only {n} history rows — too short to summarize")
        return
    thirds = [history[: n // 3], history[n // 3 : 2 * n // 3], history[2 * n // 3 :]]
    print(f"{'metric':<28}{'start':>12}{'middle':>12}{'end':>12}")
    for key in keys:
        means = []
        for chunk in thirds:
            vals = [r[key] for r in chunk if r.get(key) is not None]
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        print(f"{key:<28}{means[0]:>12.4f}{means[1]:>12.4f}{means[2]:>12.4f}")


def main():
    api = wandb.Api()
    runs = api.runs("konu-droid/LerobotPick", filters={"tags": "ik_pnp"}, order="-created_at")
    if not runs:
        print("no ik_pnp runs found")
        sys.exit(1)
    run = runs[0]
    print(f"run: {run.name} ({run.id}) state={run.state} created={run.created_at}")

    keys = [
        "reward/total_reward", "reward/reach", "reward/gripper", "reward/lift",
        "reward/place", "reward/success", "reward/release",
        "state/pick_dist", "state/place_dist", "state/picked",
        "state/was_lifted", "state/at_target", "state/lift_height", "state/contact_val",
    ]
    history = list(run.scan_history(keys=keys + ["_step"]))
    print(f"history rows: {len(history)}")
    summarize_history(history, keys)


if __name__ == "__main__":
    main()
