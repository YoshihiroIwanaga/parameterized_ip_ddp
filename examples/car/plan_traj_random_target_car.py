import math

import matplotlib.pyplot as plt
import numpy as np

from parametric_ddp.interpolator.cubic.Constraint import (
    Constraint as CubicCarConstraint,
)
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

# These can be obtained by running find_car_margin_linear.py#
margin_zero_order = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
).reshape(1, 8, 1)
extra_const_indices_zero_order = [[]]
extra_const_eval_time_steps_from_knot_zero_order = []
extra_const_num_zero_order = 0
for indices in extra_const_indices_zero_order:
    extra_const_num_zero_order += len(indices)

margin_linear = np.array(
    [
        0.06465517,
        0.07479983,
        0.0,
        0.0,
        0.06465517,
        0.07479983,
        0.0,
        0.0,
    ]
).reshape(1, 8, 1)
extra_const_indices_linear = [[0, 4]]
extra_const_eval_time_steps_from_knot_linear = [10]
extra_const_indices_linear = [[]]
extra_const_eval_time_steps_from_knot_linear = []
extra_const_num_linear = 0
for indices in extra_const_indices_linear:
    extra_const_num_linear += len(indices)

margin_cubic = np.array(
    [
        0.09232234,
        0.07013535,
        0.0,
        0.0,
        0.09232234,
        0.07013535,
        0.0,
        0.0,
    ]
).reshape(1, 8, 1)
extra_const_indices_cubic = [[0, 4]]
extra_const_eval_time_steps_from_knot_cubic = [9]
extra_const_indices_cubic = [[]]
extra_const_eval_time_steps_from_knot_cubic = []
extra_const_num_cubic = 0
for indices in extra_const_indices_cubic:
    extra_const_num_cubic += len(indices)
############################################################


class OptimizationLog:
    def __init__(self, N):
        self.comp_times = np.zeros(N)
        self.diff_time_per_iter = np.zeros(N)
        self.bp_time_per_iter = np.zeros(N)
        self.fp_time_per_iter = np.zeros(N)
        self.total_iters = np.zeros(N)
        self.costs = np.zeros(N)
        self.max_const_vaiolation = np.zeros(N)
        self.jerks = np.zeros(N)
        self.success = np.zeros(N)


# def calc_jerk(us):
#     dus = (us[:, 1:] - us[:, :-1]) / dt
#     jerk = np.sum((dus / config.u_max) * (dus / config.u_max), axis=1) / horizon
#     jerk_cost = np.mean(jerk)
#     return jerk_cost


def is_success(xs, x_ref):
    d_error = np.linalg.norm(xs[:2, -1] - x_ref[:2])
    Y_error = np.abs(xs[2, -1] - x_ref[2])
    if d_error < 0.05 and Y_error < math.radians(0.5):
        return 1
    else:
        return 0


if __name__ == "__main__":
    cnt = 0
    success_cnt = 0
    N = 5

    x_refs = []
    y_refs = []
    Y_refs = []
    xs_list = []
    us_list = []
    zi_xs_list = []
    zi_us_list = []
    li_xs_list = []
    li_us_list = []
    ci_xs_list = []
    ci_us_list = []

    knot_step = 20
    steps_per_knot = 25 * [knot_step]

    org_log = OptimizationLog(N)
    zi_log = OptimizationLog(N)
    li_log = OptimizationLog(N)
    ci_log = OptimizationLog(N)
    while True:
        print(success_cnt)
        rng = np.random.default_rng(cnt)
        Y_r = rng.uniform(-math.pi, math.pi)
        x_r = rng.uniform(-10.0, 10.0)
        y_max = min(10.0, 15.0 - x_r)
        y_r = rng.uniform(-y_max, y_max)
        x_ref = np.array([x_r, y_r, Y_r, 0.0, 0.0])

        # Original
        cfg = CarConfig(x_ref=x_ref, horizon=sum(steps_per_knot))
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
        if solver.terminate_condition != "Reach the Given Optimal Cost":
            print("Original Error!!!")
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
        if solver_zero_order.terminate_condition != "Reach the Given Optimal Cost":
            print("Zero Error!!!")
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
        if solver_linear.terminate_condition != "Reach the Given Optimal Cost":
            print("Linear Error!!!")
        xs_opt_linear, us_opt_linear = solver_linear.get_optimal_traj()

        # ------Cubic--------------------------------------------------------
        cfg_cubic = CubicInterpolationConfigWrapper(
            cfg,
            steps_per_knot,
        )
        cost_cubic = CarCost(cfg_cubic)
        const_cubic = CubicCarConstraint(cfg_cubic)
        const.set_margin(margin_cubic)
        const_cubic.set_margin(np.hstack((margin_cubic, np.zeros((1, 4, 1)))))
        # const_cubic.margin
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
        # solver_cubic.prints_info = True
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
        # solver_cubic.prints_info = True
        solver_cubic.run()
        if solver_cubic.terminate_condition != "Reach the Given Optimal Cost":
            print("Cubic Error!!!")
            print(solver_cubic.terminate_condition)
        xs_opt_cubic, us_opt_cubic = solver_cubic.get_optimal_traj()

        cnt += 1
        # print(np.linalg.norm(xs_opt[-1] - xs_opt_linear[-1]))
        # 別の局所解に収束した場合はスキップ
        if (
            (np.linalg.norm(xs_opt[-1] - xs_opt_zero_order[-1]) > 10)
            or (np.linalg.norm(xs_opt[-1] - xs_opt_linear[-1]) > 10)
            or (np.linalg.norm(xs_opt[-1] - xs_opt_cubic[-1]) > 10)
        ):
            print("skip")
            continue
        else:
            x_refs.append(x_r)
            y_refs.append(y_r)
            Y_refs.append(Y_r)
            diff_time = np.array(solver.comp_times.diff_times)
            fp_time = np.array(solver.comp_times.fp_times)
            bp_time = np.array(solver.comp_times.bp_times)
            total = diff_time + fp_time + bp_time
            org_log.total_iters[success_cnt] = solver.total_iter
            org_log.comp_times[success_cnt] = np.sum(total)
            org_log.diff_time_per_iter[success_cnt] = np.mean(
                diff_time[np.where(fp_time > 1e-5)]
            )
            org_log.fp_time_per_iter[success_cnt] = np.mean(
                fp_time[np.where(fp_time > 1e-5)]
            )
            org_log.bp_time_per_iter[success_cnt] = np.mean(
                bp_time[np.where(fp_time > 1e-5)]
            )
            org_log.costs[success_cnt] = solver.get_optimal_cost()

            diff_time = np.array(solver_zero_order.comp_times.diff_times)
            fp_time = np.array(solver_zero_order.comp_times.fp_times)
            bp_time = np.array(solver_zero_order.comp_times.bp_times)
            total = diff_time + fp_time + bp_time
            zi_log.total_iters[success_cnt] = solver_zero_order.total_iter
            zi_log.comp_times[success_cnt] = np.sum(total)
            zi_log.diff_time_per_iter[success_cnt] = np.mean(
                diff_time[np.where(fp_time > 1e-5)]
            )
            zi_log.fp_time_per_iter[success_cnt] = np.mean(
                fp_time[np.where(fp_time > 1e-5)]
            )
            zi_log.bp_time_per_iter[success_cnt] = np.mean(
                bp_time[np.where(fp_time > 1e-5)]
            )
            ocp.calc_cost(solver_zero_order.traj.every_traj)
            zi_log.costs[success_cnt] = solver_zero_order.traj.every_traj.costs[
                solver_zero_order.traj.traj_idx
            ]
            # print("Zero-Order Interpolated iLQR took : ", np.sum(total), "[sec]")

            diff_time = np.array(solver_linear.comp_times.diff_times)
            fp_time = np.array(solver_linear.comp_times.fp_times)
            bp_time = np.array(solver_linear.comp_times.bp_times)
            total = diff_time + fp_time + bp_time
            li_log.total_iters[success_cnt] = solver_linear.total_iter
            li_log.comp_times[success_cnt] = np.sum(total)
            li_log.diff_time_per_iter[success_cnt] = np.mean(
                diff_time[np.where(fp_time > 1e-5)]
            )
            li_log.fp_time_per_iter[success_cnt] = np.mean(
                fp_time[np.where(fp_time > 1e-5)]
            )
            li_log.bp_time_per_iter[success_cnt] = np.mean(
                bp_time[np.where(fp_time > 1e-5)]
            )
            ocp.calc_cost(solver_linear.traj.every_traj)
            li_log.costs[success_cnt] = solver_linear.traj.every_traj.costs[
                solver_linear.traj.traj_idx
            ]
            # print("Linear Interpolated iLQR took : ", np.sum(total), "[sec]")
            # print(fp_time)
            # print(solver_linear.total_iter)
            # print("Linear iLQR total iter", solver_linear.total_iter)

            diff_time = np.array(solver_cubic.comp_times.diff_times)
            fp_time = np.array(solver_cubic.comp_times.fp_times)
            bp_time = np.array(solver_cubic.comp_times.bp_times)
            total = diff_time + fp_time + bp_time
            ci_log.total_iters[success_cnt] = solver_cubic.total_iter
            ci_log.comp_times[success_cnt] = np.sum(total)
            ci_log.diff_time_per_iter[success_cnt] = np.mean(
                diff_time[np.where(fp_time > 1e-5)]
            )
            ci_log.fp_time_per_iter[success_cnt] = np.mean(
                fp_time[np.where(fp_time > 1e-5)]
            )
            ci_log.bp_time_per_iter[success_cnt] = np.mean(
                bp_time[np.where(fp_time > 1e-5)]
            )
            ocp.calc_cost(solver_cubic.traj.every_traj)
            ci_log.costs[success_cnt] = solver_cubic.traj.every_traj.costs[
                solver_cubic.traj.traj_idx
            ]
            # print("Cubic Interpolated iLQR took : ", np.sum(total), "[sec]")
            # print("Linear iLQR total iter", solver_linear.total_iter)

            xs_list.append(xs_opt)
            us_list.append(us_opt)
            zi_xs_list.append(xs_opt_zero_order)
            zi_us_list.append(us_opt_zero_order)
            li_xs_list.append(xs_opt_linear)
            li_us_list.append(us_opt_linear)
            ci_xs_list.append(xs_opt_cubic)
            ci_us_list.append(us_opt_cubic)

            success_cnt += 1

        if success_cnt == N:
            break

    def print_metric(metric, org, zi, li, ci):
        print("original " + metric + " :", np.mean(org))
        print(
            "ZI " + metric + " :",
            np.mean(zi),
            100 * (np.mean(zi) - np.mean(org)) / np.mean(org),
        )
        print(
            "LI " + metric + " :",
            np.mean(li),
            100 * (np.mean(li) - np.mean(org)) / np.mean(org),
        )
        print(
            "CI " + metric + " :",
            np.mean(ci),
            100 * (np.mean(ci) - np.mean(org)) / np.mean(org),
        )

    print_metric("cost", org_log.costs, zi_log.costs, li_log.costs, ci_log.costs)
    print_metric(
        "time",
        org_log.comp_times,
        zi_log.comp_times,
        li_log.comp_times,
        ci_log.comp_times,
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    # patches_list = draw_fork_lift_from_above(
    #     0,
    #     0,
    #     0,
    #     0,
    #     comparator.const.shrink_rate,
    #     comparator.const.zmp_x_max,
    #     ax.transData,
    # )
    for traj_ in xs_list:
        ax.plot(traj_[0, :], traj_[1, :], "red")
    ax.quiver(
        np.array(x_refs),
        np.array(y_refs),
        3.0 * np.cos(np.array(Y_refs)),
        3.0 * np.sin(np.array(Y_refs)),
    )
    # for patch_ in patches_list:
    #     ax.add_patch(patch_)
    ax.set_aspect("equal", adjustable="box")
    ax = fig.add_subplot(2, 2, 2)
    for traj_ in zi_xs_list:
        ax.plot(traj_[0, :], traj_[1, :], "royalblue")
    ax.quiver(
        np.array(x_refs),
        np.array(y_refs),
        3.0 * np.cos(np.array(Y_refs)),
        3.0 * np.sin(np.array(Y_refs)),
    )
    # for patch_ in patches_list:
    #     ax.add_patch(patch_)
    ax.set_aspect("equal", adjustable="box")

    ax = fig.add_subplot(2, 2, 3)
    for traj_ in li_xs_list:
        ax.plot(traj_[0, :], traj_[1, :], "darkorange")
    ax.quiver(
        np.array(x_refs),
        np.array(y_refs),
        3.0 * np.cos(np.array(Y_refs)),
        3.0 * np.sin(np.array(Y_refs)),
    )
    # for patch_ in patches_list:
    #     ax.add_patch(patch_)
    ax.set_aspect("equal", adjustable="box")

    ax = fig.add_subplot(2, 2, 4)
    for traj_ in ci_xs_list:
        ax.plot(traj_[0, :], traj_[1, :], "darkorange")
    ax.quiver(
        np.array(x_refs),
        np.array(y_refs),
        3.0 * np.cos(np.array(Y_refs)),
        3.0 * np.sin(np.array(Y_refs)),
    )
    # for patch_ in patches_list:
    #     ax.add_patch(patch_)
    ax.set_aspect("equal", adjustable="box")

    plt.savefig(
        "./trajs" + ".png",
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.clf()
    plt.close()
    # Linear
