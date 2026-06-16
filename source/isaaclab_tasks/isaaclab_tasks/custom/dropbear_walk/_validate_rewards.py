# Copyright (c) 2025. Offline reward-landscape sanity check for the Dropbear walk task.
#
# This standalone script extracts ``compute_rewards`` from ``dropbear_walk_env.py`` and
# evaluates it on a set of synthetic scenarios WITHOUT launching Isaac Sim. It verifies the
# sign and monotonicity of the reward landscape (e.g. upright > fallen, toward-goal >
# away-from-goal) before any expensive GPU training run.
#
# Run: python _validate_rewards.py

import os
import re
import ast
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_compute_rewards():
    """Extract and compile the ``compute_rewards`` function in isolation (no isaaclab import).

    Returns:
        The ``compute_rewards`` callable, with the ``@torch.jit.script`` decorator stripped
        so it runs as plain eager torch for easy inspection.
    """
    src = open(os.path.join(HERE, "dropbear_walk_env.py")).read()
    tree = ast.parse(src)
    func_src = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_rewards":
            # drop decorators (torch.jit.script) and re-emit the source segment
            node.decorator_list = []
            func_src = ast.get_source_segment(src, node)
            break
    assert func_src is not None, "compute_rewards not found"
    ns = {"torch": torch}
    exec(func_src, ns)
    return ns["compute_rewards"]


def _scales():
    """Return the reward scales/params parsed from the cfg file (keeps this test in sync)."""
    cfg_src = open(os.path.join(HERE, "dropbear_walk_env_cfg.py")).read()
    vals = {}
    for name, val in re.findall(r"^\s*(rew_scale_\w+|target_\w+|feet_separation_threshold|"
                                r"contact_force_threshold|goal_reached_threshold)\s*=\s*([-\d.e]+)",
                                cfg_src, re.M):
        vals[name] = float(val)
    return vals


def build_inputs(n, **kw):
    """Build a batch of reward inputs for ``n`` envs, with optional per-scenario overrides."""
    z = torch.zeros(n)
    o = torch.ones(n)
    data = dict(
        vel_to_goal=z.clone(),
        heading_dot=o.clone(),
        goal_dist=torch.full((n,), 4.0),
        proj_gravity=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
        head_z=torch.full((n,), 1.6),
        root_lin_vel_b=torch.zeros(n, 3),
        root_ang_vel_b=torch.zeros(n, 3),
        feet_z=torch.tensor([[0.02, 0.10]]).repeat(n, 1),
        feet_pos_xy=torch.tensor([[0.0, 0.1], [0.0, -0.1]]).repeat(n, 1, 1),
        feet_vel_z=torch.zeros(n, 2),
        net_contact_f=torch.tensor([[[0.0, 0.0, 50.0], [0.0, 0.0, 0.0]]]).repeat(n, 1, 1),
        current_air_time=torch.tensor([[0.0, 0.3]]).repeat(n, 1),
        contact_target=torch.tensor([[True, False]]).repeat(n, 1),
        shoulder_ang=torch.tensor([[-0.2, 0.2]]).repeat(n, 1),
        actions=torch.zeros(n, 14),
        prev_actions=torch.zeros(n, 14),
        joint_vel=torch.zeros(n, 14),
        joint_pos=torch.zeros(n, 14),
        applied_torque=torch.zeros(n, 14),
        dof_lower=torch.full((14,), -1.5),
        dof_upper=torch.full((14,), 1.5),
        reset_terminated=torch.zeros(n, dtype=torch.bool),
    )
    data.update(kw)
    return data


def main():
    compute_rewards = _load_compute_rewards()
    s = _scales()

    def run(inp):
        scale_args = [
            s["rew_scale_progress"], s["rew_scale_heading"], s["rew_scale_alive"], s["rew_scale_gait"],
            s["rew_scale_air_time"], s["rew_scale_foot_clearance"], s["rew_scale_arm_swing"],
            s["rew_scale_upright"], s["rew_scale_head_height"], s["rew_scale_goal_bonus"],
            s["rew_scale_lin_vel_z"], s["rew_scale_ang_vel_xy"], s["rew_scale_action_rate"],
            s["rew_scale_joint_vel"], s["rew_scale_joint_torque"], s["rew_scale_feet_near"],
            s["rew_scale_contact_impact"], s["rew_scale_dof_limit"], s["rew_scale_terminated"],
        ]
        param_args = [
            s["target_speed"], s["target_head_height"], s["target_foot_clearance"], s["target_air_time"],
            s["feet_separation_threshold"], s["contact_force_threshold"], s["goal_reached_threshold"],
        ]
        tensor_keys = [
            "vel_to_goal", "heading_dot", "goal_dist", "proj_gravity", "head_z", "root_lin_vel_b",
            "root_ang_vel_b", "feet_z", "feet_pos_xy", "feet_vel_z", "net_contact_f", "current_air_time",
            "contact_target", "shoulder_ang", "actions", "prev_actions", "joint_vel", "joint_pos",
            "applied_torque", "dof_lower", "dof_upper", "reset_terminated",
        ]
        total, log = compute_rewards(*scale_args, *param_args, *[inp[k] for k in tensor_keys])
        return total.mean().item(), log

    scenarios = {
        "A_walk_to_goal_upright": build_inputs(
            1,
            vel_to_goal=torch.full((1,), 0.6),
            root_lin_vel_b=torch.tensor([[0.6, 0.0, 0.0]]),
        ),
        "B_stand_still_upright": build_inputs(1),
        "C_move_away_from_goal": build_inputs(
            1,
            vel_to_goal=torch.full((1,), -0.6),
            heading_dot=torch.full((1,), -1.0),
            root_lin_vel_b=torch.tensor([[-0.6, 0.0, 0.0]]),
        ),
        "D_tilted_torso": build_inputs(
            1,
            proj_gravity=torch.tensor([[0.6, 0.0, -0.4]]),
            head_z=torch.full((1,), 1.3),
        ),
        "E_fallen_terminated": build_inputs(
            1,
            proj_gravity=torch.tensor([[0.9, 0.0, -0.1]]),
            head_z=torch.full((1,), 0.8),
            reset_terminated=torch.ones(1, dtype=torch.bool),
        ),
        "F_wrong_gait_phase": build_inputs(
            1,
            vel_to_goal=torch.full((1,), 0.6),
            contact_target=torch.tensor([[False, True]]),  # opposite of actual contact
        ),
        "G_jerky_bouncy": build_inputs(
            1,
            vel_to_goal=torch.full((1,), 0.6),
            root_lin_vel_b=torch.tensor([[0.6, 0.0, 1.0]]),  # vertical bounce
            actions=torch.ones(1, 14),                       # large action delta
            joint_vel=torch.full((1, 14), 5.0),
        ),
    }

    print(f"{'scenario':28s} {'total':>8s}")
    totals = {}
    comp = {}
    for name, inp in scenarios.items():
        t, log = run(inp)
        totals[name] = t
        comp[name] = log
        print(f"{name:28s} {t:8.3f}")

    print("\n-- component breakdown (key terms) --")
    keys = ["reward/progress", "reward/heading", "reward/gait", "reward/upright",
            "reward/head_height", "reward/arm_swing", "reward/foot_clearance",
            "reward/pen_lin_vel_z", "reward/pen_action_rate", "reward/termination"]
    hdr = "term".ljust(24) + "".join(f"{n.split('_')[0]:>8s}" for n in scenarios)
    print(hdr)
    for k in keys:
        row = k.replace("reward/", "").ljust(24)
        for name in scenarios:
            row += f"{comp[name][k]:8.3f}"
        print(row)

    print("\n-- monotonicity assertions --")
    checks = [
        ("walk_to_goal > stand_still", totals["A_walk_to_goal_upright"] > totals["B_stand_still_upright"]),
        ("stand_still > move_away", totals["B_stand_still_upright"] > totals["C_move_away_from_goal"]),
        ("upright > tilted", totals["B_stand_still_upright"] > totals["D_tilted_torso"]),
        ("tilted > fallen", totals["D_tilted_torso"] > totals["E_fallen_terminated"]),
        ("correct_gait > wrong_gait", totals["A_walk_to_goal_upright"] > totals["F_wrong_gait_phase"]),
        ("smooth > jerky", totals["A_walk_to_goal_upright"] > totals["G_jerky_bouncy"]),
    ]
    ok = True
    for desc, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
        ok = ok and passed
    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
