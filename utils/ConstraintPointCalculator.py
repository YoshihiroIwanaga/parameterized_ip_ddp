import typing

import numpy as np
from scipy.optimize import minimize

from parametric_ddp.interpolator.cubic.CubicInterpolationConfigWrapper import (
    CubicInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.linear.LinearInterpolationConfigWrapper import (
    LinearInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.zero_order.ZeroOrderHolderConfigWrapper import (
    ZeroOrderHolderConfigWrapper,
)
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper


class ConstraintPointCalculator:
    def __init__(
        self,
        dynamics,
        const,
        original_const,
        ocp,
        traj_info: TrajInfoWrapper,
        cfg: typing.Union[
            ZeroOrderHolderConfigWrapper,
            LinearInterpolationConfigWrapper,
            CubicInterpolationConfigWrapper,
        ],
    ) -> None:
        self.dynamics = dynamics
        self.const = const
        self.original_const = original_const
        self.ocp = ocp
        self.cfg = cfg
        self.org_n: int = self.cfg.org_cfg.n
        self.traj_info: TrajInfoWrapper = traj_info
        self.tol: float = 0.01
        self.max_const_violation_to_add_eval_point: typing.List[float] = (
            self.original_const.lc * [0.15]
        )
        self.cobyla_options = {"maxiter": 2000}

    def obj_func(self, x, step_from_knot, const_idx) -> float:
        x_at_current_knot = x[: self.cfg.n].copy()
        u_at_current_knot = x[self.cfg.n : self.cfg.n + self.cfg.m].copy()
        u_at_next_knot = x[self.cfg.n + self.cfg.m :].copy()

        self.traj_info.xs[0, :, 0] = x_at_current_knot
        self.traj_info.every_traj.xs[0, :, 0] = x_at_current_knot[: self.org_n]
        self.traj_info.us[0, :, 0] = u_at_current_knot
        self.traj_info.us[0, :, 1] = u_at_next_knot

        self.ocp.transit(self.traj_info, 0)
        self.original_const.calc_ith_const_at_(
            self.traj_info.every_traj,
            const_indices=[const_idx],
            time_steps=[step_from_knot],
        )
        return -self.traj_info.every_traj.cs[0, const_idx, step_from_knot]

    def const_at_knot(self, x, margin) -> np.ndarray:
        x_at_current_knot = x[: self.cfg.n].copy()
        u_at_current_knot = x[self.cfg.n : self.cfg.n + self.cfg.m].copy()
        u_at_next_knot = x[self.cfg.n + self.cfg.m :].copy()

        self.traj_info.xs[0, :, 0] = x_at_current_knot
        self.traj_info.every_traj.xs[0, :, 0] = x_at_current_knot[: self.org_n]
        self.traj_info.us[0, :, 0] = u_at_current_knot
        self.traj_info.us[0, :, 1] = u_at_next_knot
        self.ocp.transit(self.traj_info, 0)
        self.ocp.calc_const(self.traj_info)
        return -np.hstack(
            (self.traj_info.cs[0, :, 0] + margin, self.traj_info.cs[0, :, 1] + margin)
        )

    def const_between_knot(
        self, x, margin_, const_indices, step_from_knots
    ) -> np.ndarray:
        x_at_current_knot = x[: self.cfg.n].copy()
        u_at_current_knot = x[self.cfg.n : self.cfg.n + self.cfg.m].copy()
        u_at_next_knot = x[self.cfg.n + self.cfg.m :].copy()

        self.traj_info.xs[0, :, 0] = x_at_current_knot
        self.traj_info.every_traj.xs[0, :, 0] = x_at_current_knot[: self.org_n]
        self.traj_info.us[0, :, 0] = u_at_current_knot
        self.traj_info.us[0, :, 1] = u_at_next_knot
        self.ocp.transit(self.traj_info, 0)

        unique_step_from_knots = list(set(step_from_knots))
        self.original_const.calc_ith_const_at_(
            self.traj_info.every_traj,
            const_indices=const_indices,
            time_steps=unique_step_from_knots,
        )

        const_between_knot = np.zeros(len(const_indices))
        for i, (const_idx, step_from_knot) in enumerate(
            zip(const_indices, step_from_knots)
        ):
            const_between_knot[i] = -(
                self.traj_info.every_traj.cs[0, const_idx, step_from_knot]
                + margin_[const_idx]
            )

        return const_between_knot

    def _calc_max_const_violation(
        self, step_from_knot, const_idx, constraints
    ) -> float:
        max_violation = -np.inf
        for i in range(10):
            x_ini = np.random.randn(self.cfg.n + 2 * self.cfg.m)
            result = minimize(
                fun=self.obj_func,
                x0=x_ini,
                args=(step_from_knot, const_idx),
                constraints=constraints,
                method="COBYLA",
                options=self.cobyla_options,
            )
            if result.success:
                if -1 * result.fun > max_violation:
                    max_violation = -1 * result.fun
                if const_idx < 16:
                    break
        return max_violation

    def calc(self, knot_step) -> None:
        margin_pre = np.zeros(self.const.lc)
        margin = np.zeros(self.const.lc)
        d_margin = np.zeros(self.const.lc)
        iter_ = 0
        enabled_const_indices = []
        enabled_const_steps = []
        enabled_const_indices_and_step = (
            np.ones((self.const.lc, knot_step), dtype=bool) * False
        )

        while True:
            print("iter : ", iter_)
            constraints = (
                {
                    "type": "ineq",
                    "fun": self.const_at_knot,
                    "args": (margin,),
                },
            )
            added = False

            if len(enabled_const_indices) != 0:
                constraints += (
                    {
                        "type": "ineq",
                        "fun": self.const_between_knot,
                        "args": (
                            margin,
                            enabled_const_indices,
                            enabled_const_steps,
                        ),
                    },
                )

            for const_idx in range(self.original_const.lc):
                d_margins = []
                for step_from_knot in range(1, knot_step):
                    if enabled_const_indices_and_step[const_idx, step_from_knot - 1]:
                        d_margins.append(0.0)
                    else:
                        max_violation = self._calc_max_const_violation(
                            step_from_knot, const_idx, constraints
                        )
                        d_margins.append(max_violation)

                max_val = max(d_margins)
                if (
                    margin_pre[const_idx] + max_val
                    < self.max_const_violation_to_add_eval_point[const_idx]
                ):
                    d_margin[const_idx] = max_val
                    if abs(max_val) < 1e-5:
                        d_margin[const_idx] = 0.0
                else:
                    d_margin[const_idx] = 0
                    enabled_const_indices.append(const_idx)
                    step_from_knot_max_violation = d_margins.index(max_val) + 1
                    enabled_const_steps.append(step_from_knot_max_violation)
                    enabled_const_indices_and_step[
                        const_idx, step_from_knot_max_violation - 1
                    ] = True
                    added = True

            margin = np.clip(margin_pre + d_margin, 0, np.inf)
            if np.max(np.abs(margin - margin_pre)) < self.tol and not added:
                break
            iter_ += 1
            margin_pre = margin.copy()

        print("enabled_const_indices : ", enabled_const_indices)
        print("enabled_const_steps : ", enabled_const_steps)
        print("margin : ", margin[: self.original_const.lc])
