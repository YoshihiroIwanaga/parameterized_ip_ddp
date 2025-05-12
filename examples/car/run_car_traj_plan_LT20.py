# import math

import matplotlib.pyplot as plt
import numpy as np

from parametric_ddp.interpolator.cubic.Constraint import Constraint as CubicConstraint
from parametric_ddp.interpolator.cubic.CubicInterpolationConfigWrapper import (
    CubicInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.cubic.CubicInterpolationWrapper import (
    CubicInterpolationWrapper,
)
from parametric_ddp.interpolator.linear.LinearInterpolationConfigWrapper import (
    LinearInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.linear.LinearInterpolationWrapper import (
    LinearInterpolationWrapper,
)
from parametric_ddp.interpolator.zero_order.ZeroOrderHolderConfigWrapper import (
    ZeroOrderHolderConfigWrapper,
)
from parametric_ddp.interpolator.zero_order.ZeroOrderHolderWrapper import (
    ZeroOrderHolderWrapper,
)
from parametric_ddp.ip_ddp.InteriorPointDdpSolver import InteriorPointDdpSolver
from parametric_ddp.utils.OptimalControlProblem import OptimalControlProblem
from parametric_ddp.utils.TrajInfo import TrajInfo
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.car.CarConfig import CarConfig
from problem_setting.car.CarConstraint import CarConstraint
from problem_setting.car.CarCost import CarCost
from problem_setting.car.CarDynamics import CarDynamics
from problem_setting.car.constraint_points_LT20 import (
    extra_const_eval_time_steps_from_knot_cubic,
    extra_const_eval_time_steps_from_knot_linear,
    extra_const_eval_time_steps_from_knot_zero_order,
    extra_const_indices_cubic,
    extra_const_indices_linear,
    extra_const_indices_zero_order,
    extra_const_num_cubic,
    extra_const_num_linear,
    extra_const_num_zero_order,
    margin_cubic,
    margin_linear,
    margin_zero_order,
)
from problem_setting.car.utils.plot_lib import create_gif, draw_car
from utils.performance_metric_lib import calc_jerk, get_comp_time_and_cost, print_metric
from utils.plot_traj_lib import plot_cost_vs_time, plot_input, plot_state


def plot_xy_path(
    ax,
    xs_opt,
    xs_opt_zero_order,
    xs_opt_linear,
    xs_opt_cubic,
    solver_zero_order,
    solver_linear,
    solver_cubic,
    original_label,
    zi_label,
    li_label,
    ci_label,
    original_color="red",
    zi_color="royalblue",
    li_color="darkgreen",
    ci_color="darkorange",
):
    patches_list = draw_car(
        0,
        0,
        0,
        0,
        ax.transData,
    )
    for patch_ in patches_list:
        ax.add_patch(patch_)

    patches_list = draw_car(
        cfg.x_ref[0],
        cfg.x_ref[1],
        cfg.x_ref[2],
        cfg.x_ref[4],
        ax.transData,
        ls_="dashed",
    )
    for patch_ in patches_list:
        ax.add_patch(patch_)

    ax.plot(xs_opt[0, :], xs_opt[1, :], original_color, label=original_label)
    ax.plot(
        xs_opt_zero_order[0, :],
        xs_opt_zero_order[1, :],
        zi_color,
        label=zi_label,
    )
    ax.plot(
        solver_zero_order.traj.xs[solver_zero_order.traj.traj_idx, 0, :],
        solver_zero_order.traj.xs[solver_zero_order.traj.traj_idx, 1, :],
        "o",
        color=zi_color,
    )
    ax.plot(xs_opt_linear[0, :], xs_opt_linear[1, :], li_color, label=li_label)
    ax.plot(
        solver_linear.traj.xs[solver_linear.traj.traj_idx, 0, :],
        solver_linear.traj.xs[solver_linear.traj.traj_idx, 1, :],
        "o",
        color=li_color,
    )
    ax.plot(xs_opt_cubic[0, :], xs_opt_cubic[1, :], ci_color, label=ci_label)
    ax.plot(
        solver_cubic.traj.xs[solver_cubic.traj.traj_idx, 0, :],
        solver_cubic.traj.xs[solver_cubic.traj.traj_idx, 1, :],
        "o",
        color="darkorange",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend()


state_name = [
    "$x$[m]",
    "$y$[m]",
    "$\phi$[rad]",
    "$v$[m/sec]",
    "$\psi$[rad]",
]

input_name = [
    "$a$[m/sec$^{2}$]",
    "$\omega$[rad/sec]",
]


if __name__ == "__main__":
    # problem setting
    knot_step = 20
    steps_per_knot = 25 * [knot_step]
    # Original
    cfg = CarConfig(horizon=sum(steps_per_knot))
    dynamics = CarDynamics(cfg)
    cost = CarCost(cfg)
    const = CarConstraint(cfg)
    traj_info = TrajInfo(cfg, const.lc, const.lc_terminal, cfg.step_size_num)
    ocp = OptimalControlProblem(dynamics, const, cost, cfg)
    solver = InteriorPointDdpSolver(ocp, traj_info, cfg)
    solver.run()
    cfg.knows_optimal_cost = True
    cfg.optimal_cost = solver.get_optimal_cost()
    traj_info = TrajInfo(cfg, const.lc, const.lc_terminal, cfg.step_size_num)
    solver = InteriorPointDdpSolver(ocp, traj_info, cfg)
    solver.run()
    xs_opt, us_opt = solver.get_optimal_traj()

    # ---- Zero Order ------------------------------------------------
    cfg_zero_order = ZeroOrderHolderConfigWrapper(
        cfg,
        steps_per_knot,
    )
    cost_zero_order = CarCost(cfg_zero_order)
    const_zero_order = CarConstraint(cfg_zero_order)
    const_zero_order.set_margin(margin_zero_order)
    traj_info_zero_order = TrajInfoWrapper(
        cfg_zero_order,
        const_zero_order.lc + extra_const_num_zero_order,
        const_zero_order.lc_terminal,
        const.lc,
        cfg.step_size_num,
    )
    ocp_zero_order = ZeroOrderHolderWrapper(
        dynamics,
        const_zero_order,
        cost_zero_order,
        cfg_zero_order,
        extra_const_indices=extra_const_indices_zero_order,
        extra_const_eval_time_steps_from_knot=extra_const_eval_time_steps_from_knot_zero_order,
    )
    solver_zero_order = InteriorPointDdpSolver(
        ocp_zero_order, traj_info_zero_order, cfg_zero_order
    )
    solver_zero_order.run()
    xs_opt_zero_order, us_opt_zero_order = solver_zero_order.get_optimal_traj()
    cfg_zero_order.knows_optimal_cost = True
    cfg_zero_order.optimal_cost = solver_zero_order.get_optimal_cost()
    traj_info_zero_order = TrajInfoWrapper(
        cfg_zero_order,
        const_zero_order.lc + extra_const_num_zero_order,
        const_zero_order.lc_terminal,
        const.lc,
        cfg.step_size_num,
    )
    solver_zero_order = InteriorPointDdpSolver(
        ocp_zero_order, traj_info_zero_order, cfg_zero_order
    )
    solver_zero_order.run()
    xs_opt_zero_order, us_opt_zero_order = solver_zero_order.get_optimal_traj()

    # ---- Lienar ------------------------------------------------
    cfg_linear = LinearInterpolationConfigWrapper(
        cfg,
        steps_per_knot,
    )
    cost_linear = CarCost(cfg_linear)
    const_linear = CarConstraint(cfg_linear)
    const.set_margin(margin_linear)
    const_linear.set_margin(margin_linear)
    traj_info_linear = TrajInfoWrapper(
        cfg_linear,
        const_linear.lc + extra_const_num_linear,
        const_linear.lc_terminal,
        const.lc,
        cfg.step_size_num,
    )
    ocp_linear = LinearInterpolationWrapper(
        dynamics,
        const_linear,
        const,
        cost_linear,
        cfg_linear,
        extra_const_indices=extra_const_indices_linear,
        extra_const_eval_time_steps_from_knot=extra_const_eval_time_steps_from_knot_linear,
    )
    solver_linear = InteriorPointDdpSolver(ocp_linear, traj_info_linear, cfg_linear)
    solver_linear.run()
    cfg_linear.knows_optimal_cost = True
    cfg_linear.optimal_cost = solver_linear.get_optimal_cost()
    traj_info_linear = TrajInfoWrapper(
        cfg_linear,
        const_linear.lc + extra_const_num_linear,
        const_linear.lc_terminal,
        const.lc,
        cfg.step_size_num,
    )
    solver_linear = InteriorPointDdpSolver(ocp_linear, traj_info_linear, cfg_linear)
    solver_linear.run()
    xs_opt_linear, us_opt_linear = solver_linear.get_optimal_traj()

    # ------Cubic--------------------------------------------------------
    cfg_cubic = CubicInterpolationConfigWrapper(
        cfg,
        steps_per_knot,
    )
    cost_cubic = CarCost(cfg_cubic)
    const_cubic = CubicConstraint(cfg_cubic)
    const.set_margin(margin_cubic)
    const_cubic.set_margin(np.hstack((margin_cubic, np.zeros((1, 4, 1)))))
    traj_info_cubic = TrajInfoWrapper(
        cfg_cubic,
        const_cubic.lc + extra_const_num_cubic,
        const_cubic.lc_terminal,
        const.lc,
        cfg.step_size_num,
    )
    ocp_cubic = CubicInterpolationWrapper(
        dynamics,
        const_cubic,
        const,
        cost_cubic,
        cfg_cubic,
        extra_const_indices=extra_const_indices_cubic,
        extra_const_eval_time_steps_from_knot=extra_const_eval_time_steps_from_knot_cubic,
    )
    solver_cubic = InteriorPointDdpSolver(ocp_cubic, traj_info_cubic, cfg_cubic)
    solver_cubic.run()
    cfg_cubic.knows_optimal_cost = True
    cfg_cubic.optimal_cost = solver_cubic.get_optimal_cost()
    traj_info_cubic = TrajInfoWrapper(
        cfg_cubic,
        const_cubic.lc + extra_const_num_cubic,
        const_cubic.lc_terminal,
        const.lc,
        cfg.step_size_num,
    )
    solver_cubic = InteriorPointDdpSolver(ocp_cubic, traj_info_cubic, cfg_cubic)
    solver_cubic.run()
    xs_opt_cubic, us_opt_cubic = solver_cubic.get_optimal_traj()

    # Print Performance Metric
    diff_time = np.array(solver.comp_times.diff_times)
    fp_time = np.array(solver.comp_times.fp_times)
    bp_time = np.array(solver.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    org_total_iter = solver.total_iter
    org_comp_time = np.sum(total)
    org_diff_time_per_iter = np.mean(diff_time[np.where(fp_time > 1e-5)])
    org_fp_time_per_iter = np.mean(fp_time[np.where(fp_time > 1e-5)])
    org_bp_time_per_iter = np.mean(bp_time[np.where(fp_time > 1e-5)])
    org_cost = solver.get_optimal_cost()
    org_jerk = calc_jerk(us_opt, cfg.u_max, cfg.horizon, cfg.dt)

    diff_time = np.array(solver_zero_order.comp_times.diff_times)
    fp_time = np.array(solver_zero_order.comp_times.fp_times)
    bp_time = np.array(solver_zero_order.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    zi_total_iters = solver_zero_order.total_iter
    zi_comp_time = np.sum(total)
    zi_diff_time_per_iter = np.mean(diff_time[np.where(fp_time > 1e-5)])
    zi_fp_time_per_iter = np.mean(fp_time[np.where(fp_time > 1e-5)])
    zi_bp_time_per_iter = np.mean(bp_time[np.where(fp_time > 1e-5)])
    ocp.calc_cost(solver_zero_order.traj.every_traj)  # type:ignore
    zi_cost = solver_zero_order.traj.every_traj.costs[  # type:ignore
        solver_zero_order.traj.traj_idx
    ]
    zi_jerk = calc_jerk(us_opt_zero_order, cfg.u_max, cfg.horizon, cfg.dt)

    diff_time = np.array(solver_linear.comp_times.diff_times)
    fp_time = np.array(solver_linear.comp_times.fp_times)
    bp_time = np.array(solver_linear.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    li_total_iters = solver_linear.total_iter
    li_comp_time = np.sum(total)
    li_diff_time_per_iter = np.mean(diff_time[np.where(fp_time > 1e-5)])
    li_fp_time_per_iter = np.mean(fp_time[np.where(fp_time > 1e-5)])
    li_bp_time_per_iter = np.mean(bp_time[np.where(fp_time > 1e-5)])
    ocp.calc_cost(solver_linear.traj.every_traj)  # type:ignore
    li_cost = solver_linear.traj.every_traj.costs[  # type:ignore
        solver_linear.traj.traj_idx
    ]
    li_jerk = calc_jerk(us_opt_linear, cfg.u_max, cfg.horizon, cfg.dt)

    diff_time = np.array(solver_cubic.comp_times.diff_times)
    fp_time = np.array(solver_cubic.comp_times.fp_times)
    bp_time = np.array(solver_cubic.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    ci_total_iter = solver_cubic.total_iter
    ci_comp_time = np.sum(total)
    ci_diff_time_per_iter = np.mean(diff_time[np.where(fp_time > 1e-5)])
    ci_fp_time_per_iter = np.mean(fp_time[np.where(fp_time > 1e-5)])
    ci_bp_time_per_iter = np.mean(bp_time[np.where(fp_time > 1e-5)])
    ocp.calc_cost(solver_cubic.traj.every_traj)  # type:ignore
    ci_cost = solver_cubic.traj.every_traj.costs[  # type:ignore
        solver_cubic.traj.traj_idx
    ]
    ci_jerk = calc_jerk(us_opt_cubic, cfg.u_max, cfg.horizon, cfg.dt)

    print_metric("time", org_comp_time, zi_comp_time, li_comp_time, ci_comp_time)
    print_metric(
        "T diff",
        org_diff_time_per_iter,
        zi_diff_time_per_iter,
        li_diff_time_per_iter,
        ci_diff_time_per_iter,
    )
    print_metric(
        "T BP",
        org_bp_time_per_iter,
        zi_bp_time_per_iter,
        li_bp_time_per_iter,
        ci_bp_time_per_iter,
    )
    print_metric(
        "T FP",
        org_fp_time_per_iter,
        zi_fp_time_per_iter,
        li_fp_time_per_iter,
        ci_fp_time_per_iter,
    )
    print_metric("iter", org_total_iter, zi_total_iters, li_total_iters, ci_total_iter)
    print_metric("cost", org_cost, zi_cost, li_cost, ci_cost)
    print_metric("jerk", org_jerk, zi_jerk, li_jerk, ci_jerk)

    # ---Plot Cost-----------------------------------------------------
    original_color = "red"
    zi_color = "royalblue"
    li_color = "darkgreen"
    ci_color = "darkorange"

    if cfg.run_ddp:
        file_name = "./car_DDP_LT20_"
        original_label = "IP-DDP"
        zi_label = "P-IP-DDP(Z-I)"
        li_label = "P-IP-DDP(L-I)"
        ci_label = "P-IP-DDP(C-I)"
    else:
        file_name = "./car_iLQR_LT20_"
        original_label = "IP-iLQR"
        zi_label = "P-IP-iLQR(Z-I)"
        li_label = "P-IP-iLQR(L-I)"
        ci_label = "P-IP-iLQR(C-I)"
    diff_time = np.array(solver.comp_times.diff_times)
    fp_time = np.array(solver.comp_times.fp_times)
    bp_time = np.array(solver.comp_times.bp_times)
    total = diff_time + fp_time + bp_time
    comp_time = [0]
    tmp = 0
    for t in total:
        tmp += t
        comp_time.append(tmp)
    cost_history = []
    for traj in solver.history.traj:
        cost_history.append(traj.costs[traj.traj_idx])

    comp_time_zero_order, original_cost_history_zero_order = get_comp_time_and_cost(
        solver_zero_order, cost
    )
    comp_time_linear, original_cost_history_linear = get_comp_time_and_cost(
        solver_linear, cost
    )
    comp_time_cubic, original_cost_history_cubic = get_comp_time_and_cost(
        solver_cubic, cost
    )
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot()
    plot_cost_vs_time(
        ax,
        comp_time,
        comp_time_zero_order,
        comp_time_linear,
        comp_time_cubic,
        cost_history,
        original_cost_history_zero_order,
        original_cost_history_linear,
        original_cost_history_cubic,
        original_label,
        zi_label,
        li_label,
        ci_label,
    )
    plt.savefig(
        file_name + "cost_vs_time.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.clf()
    plt.close()

    # -Plot state and input -------------------------------
    knot_frames = [0]
    f = 0
    for steps in steps_per_knot:
        f += steps
        knot_frames.append(f)

    fig = plt.figure(figsize=(5, 1.35 * cfg.n))
    plot_state(
        fig,
        cfg,
        xs_opt,
        xs_opt_zero_order,
        xs_opt_linear,
        xs_opt_cubic,
        knot_frames,
        state_name,
        original_label,
        zi_label,
        li_label,
        ci_label,
    )
    plt.savefig(
        file_name + "x_trajs.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(5, 1.35 * cfg.m))
    plot_input(
        fig,
        cfg,
        us_opt,
        us_opt_zero_order,
        us_opt_linear,
        us_opt_cubic,
        knot_frames,
        input_name,
        original_label,
        zi_label,
        li_label,
        ci_label,
    )
    plt.savefig(
        file_name + "u_trajs.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.clf()
    plt.close()

    # ---Plot xy-path-----------------------------------------------------
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    plot_xy_path(
        ax,
        xs_opt,
        xs_opt_zero_order,
        xs_opt_linear,
        xs_opt_cubic,
        solver_zero_order,
        solver_linear,
        solver_cubic,
        original_label,
        zi_label,
        li_label,
        ci_label,
    )
    plt.savefig(
        file_name + "_path.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.clf()
    plt.close()

    # crearte gif
    ts = np.linspace(
        0.0,
        max(
            comp_time[-1],
            comp_time_zero_order[-1],
            comp_time_linear[-1],
            comp_time_cubic[-1],
        ),
        50,
    )
    xs = []
    us = []
    zi_interpolated_xs = []
    zi_interpolated_us = []
    li_interpolated_xs = []
    li_interpolated_us = []
    ci_interpolated_xs = []
    ci_interpolated_us = []
    for t_ in ts:
        index = np.abs(np.array(comp_time) - t_).argmin()
        index_zero_order = np.abs(np.array(comp_time_zero_order) - t_).argmin()
        index_linear = np.abs(np.array(comp_time_linear) - t_).argmin()
        index_cubic = np.abs(np.array(comp_time_cubic) - t_).argmin()
        traj = solver.history.traj[index]
        traj_zero_order = solver_zero_order.history.traj[index_zero_order]
        traj_linear = solver_linear.history.traj[index_linear]
        traj_cubic = solver_cubic.history.traj[index_cubic]
        xs.append(traj.xs[traj.traj_idx, :, :])
        us.append(traj.us[traj.traj_idx, :, :])
        zi_interpolated_xs.append(
            traj_zero_order.every_traj.xs[traj_zero_order.every_traj.traj_idx, :, :]
        )
        zi_interpolated_us.append(
            traj_zero_order.every_traj.us[traj_zero_order.every_traj.traj_idx, :, :]
        )
        li_interpolated_xs.append(
            traj_linear.every_traj.xs[traj_linear.every_traj.traj_idx, :, :]
        )
        li_interpolated_us.append(
            traj_linear.every_traj.us[traj_linear.every_traj.traj_idx, :, :]
        )
        ci_interpolated_xs.append(
            traj_cubic.every_traj.xs[traj_cubic.every_traj.traj_idx, :, :]
        )
        ci_interpolated_us.append(
            traj_cubic.every_traj.us[traj_cubic.every_traj.traj_idx, :, :]
        )

    create_gif(
        file_name + "opt.gif",
        xs,
        us,
        zi_interpolated_xs,
        zi_interpolated_us,
        li_interpolated_xs,
        li_interpolated_us,
        ci_interpolated_xs,
        ci_interpolated_us,
        xs_opt,
        cfg.horizon,
        cfg.x_ref,
        cfg.x_min,
        cfg.x_max,
        cfg.u_min,
        cfg.u_max,
        3.0,
    )
