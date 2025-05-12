import numpy as np


def get_comp_time_and_cost(solver, cost):
    diff_time = np.array(solver.comp_times.diff_times)
    fp_time = np.array(solver.comp_times.fp_times)
    bp_time = np.array(solver.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    comp_time = [0]
    tmp = 0
    for t in total:
        tmp += t
        comp_time.append(tmp)
    original_cost_history = []
    for traj in solver.history.traj:
        cost.calc_stage_cost(traj.every_traj)
        cost.calc_terminal_cost(traj.every_traj)
        original_cost = (
            np.sum(traj.every_traj.stage_costs, axis=1) + traj.every_traj.terminal_costs
        )
        original_cost_history.append(original_cost[traj.traj_idx])
    return comp_time, original_cost_history


def calc_jerk(us: np.ndarray, u_max: np.ndarray, horizon: int, dt: float) -> float:
    dus = (us[:, 1:] - us[:, :-1]) / dt
    jerk = np.sum((dus / u_max) * (dus / u_max), axis=1) / horizon
    jerk_cost = np.mean(jerk)
    return jerk_cost


def print_metric(metric, org, zi, li, ci):
    print("original " + metric + " :", org)
    print(
        "ZI " + metric + " :",
        zi,
        100 * (zi - org) / org,
    )
    print(
        "LI " + metric + " :",
        li,
        100 * (li - org) / org,
    )
    print(
        "CI " + metric + " :",
        ci,
        100 * (ci - org) / org,
    )
