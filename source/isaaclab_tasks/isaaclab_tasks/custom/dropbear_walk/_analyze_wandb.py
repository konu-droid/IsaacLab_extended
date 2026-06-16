# Copyright (c) 2025. Pull the latest dropbear_walk wandb run and summarize learning trends.
#
# Usage: python _analyze_wandb.py [run_name_or_index]
#   no arg     -> most recent run
#   <name>     -> run with that display name
#
# Prints, for each logged metric, the early-vs-late mean and the direction of change, plus a
# verdict on the headline locomotion signals (total reward, vel_to_goal, episode survival).

import sys
import numpy as np
import wandb

ENTITY = "konu-droid"
PROJECT = "dropbear_walk"

# metrics where "up" is the desired learning direction
WANT_UP = {
    "reward/total", "reward/progress", "reward/heading", "reward/gait", "reward/air_time",
    "reward/foot_clearance", "reward/arm_swing", "reward/upright", "reward/head_height",
    "reward/goal_bonus", "reward/alive", "state/vel_to_goal", "state/head_z",
    "state/contact_matches",
}
# metrics where "down" is desired
WANT_DOWN = {"state/goal_dist", "state/terminated_frac"}


def main():
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", order="-created_at"))
    if not runs:
        print("no runs found")
        return

    sel = sys.argv[1] if len(sys.argv) > 1 else None
    run = runs[0]
    if sel is not None:
        match = [r for r in runs if r.name == sel]
        if match:
            run = match[0]

    print(f"run: {run.name}  state={run.state}  created={run.created_at}")
    print(f"url: {run.url}")

    hist = run.history(samples=2000, pandas=True)
    if hist is None or len(hist) == 0:
        print("no logged history yet")
        return
    print(f"logged rows: {len(hist)}")

    cols = [c for c in hist.columns if c.startswith(("reward/", "state/"))]
    n = len(hist)
    k = max(1, n // 5)  # compare first fifth vs last fifth

    print(f"\n{'metric':28s} {'early':>10s} {'late':>10s} {'delta':>10s}  trend")
    rows = []
    for c in sorted(cols):
        series = hist[c].dropna().to_numpy()
        if series.size < 2:
            continue
        early = float(np.mean(series[:k]))
        late = float(np.mean(series[-k:]))
        delta = late - early
        good = ""
        if c in WANT_UP:
            good = "OK" if delta > 0 else "..down"
        elif c in WANT_DOWN:
            good = "OK" if delta < 0 else "..up"
        print(f"{c:28s} {early:10.3f} {late:10.3f} {delta:10.3f}  {good}")
        rows.append((c, early, late, delta))

    print("\n-- headline verdict --")
    d = {c: (e, l, dl) for c, e, l, dl in rows}
    def verdict(metric, want_up=True):
        if metric not in d:
            print(f"  {metric}: (not logged)")
            return
        e, l, dl = d[metric]
        direction = "increasing" if dl > 0 else "decreasing"
        good = (dl > 0) == want_up
        print(f"  {metric}: {e:.3f} -> {l:.3f} ({direction}) {'[good]' if good else '[watch]'}")
    verdict("reward/total", True)
    verdict("state/vel_to_goal", True)
    verdict("state/terminated_frac", False)
    verdict("state/goal_dist", False)
    verdict("reward/upright", True)


if __name__ == "__main__":
    main()
