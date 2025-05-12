import typing

import numpy as np

from parametric_ddp.interpolator.zero_order.ZeroOrderHolderConfigWrapper import (
    ZeroOrderHolderConfigWrapper,
)
from parametric_ddp.utils.ConstraintPoint import ConstraintPoint
from parametric_ddp.utils.OptimalControlProblem import OptimalControlProblem
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.abstract.Constraint import Constraint
from problem_setting.abstract.Cost import Cost
from problem_setting.abstract.Dynamics import Dynamics


class ZeroOrderHolderWrapper(OptimalControlProblem):
    def __init__(
        self,
        dynamics: Dynamics,
        const: Constraint,
        cost: Cost,
        cfg: ZeroOrderHolderConfigWrapper,
        extra_const_indices: typing.List[typing.List[int]] = [[]],
        extra_const_eval_time_steps_from_knot: typing.List[int] = [],
    ) -> None:
        self.dynamics: Dynamics = dynamics
        self.const: Constraint = const
        self.cost: Cost = cost
        self.cfg: ZeroOrderHolderConfigWrapper = cfg
        self.n: int = cfg.n
        self.m: int = cfg.m
        self.steps_per_knot: typing.List[int] = (
            cfg.steps_per_knot
        )  # [L_0, L_1,..., L_{Te-1}] ∈ N^{Te}
        self.steps_per_knot_array: np.ndarray = np.array(
            self.steps_per_knot
        )  # [L_0, L_1,..., L_{Te-1}] ∈ N^{Te}
        self.time_step_on_knot: typing.List[int] = (
            cfg.time_step_on_knot
        )  # [k_(0), k_(1),..., k_(Te)] ∈ N^{Te+1}
        self.knot_idxs: typing.List[int] = (
            cfg.knot_idxs
        )  # [0,..,0,1,...,1,...,Te,...,Te] ∈ N^{te}
        self.time_step_from_knots: typing.List[int] = (
            cfg.time_step_from_knots
        )  # [0,1..,L_0-1,0,1...,L_1-1,...,0,1,...,L_{Te}-1] ∈ N^{te}
        self.lc: int = const.lc
        self.lc_terminal: int = const.lc_terminal
        self.lc_original: int = const.lc
        # The constraint points between knots
        # ex. ########################################################################################
        # If extra_const_indices = [[1, 5], [2, 4]], extra_const_eval_time_steps_from_knots = [3, 5]
        # c_1(x[k_(i) + 3], u[k_(i) + 3])<0, c_5(x[k_(i) + 3], u[k_(i) + 3])<0
        # c_2(x[k_(i) + 5], u[k_(i) + 5])<0, c_4(x[k_(i) + 5], u[k_(i) + 5])<0
        # are considered
        ###############################################################################################
        self.const_points: typing.List[ConstraintPoint] = []
        for const_indices, eval_time_step_from_knot in zip(
            extra_const_indices, extra_const_eval_time_steps_from_knot
        ):
            self.const_points.append(
                ConstraintPoint(
                    const_indices, eval_time_step_from_knot, self.time_step_on_knot
                )
            )
            self.lc += len(const_indices)

    def set_constant(
        self,
        traj: TrajInfoWrapper,
    ) -> None:
        self.dynamics.set_constant(traj.every_traj)
        self.const.set_constant(traj)
        self.cost.set_constant(traj)
        self.const.set_constant(traj.every_traj)
        # Since the Original stage cost is approximated by
        # Q_T(X_T, U_T) = L_T*q_k(T)(x[k_(T)], u[k_(T)])  --- (20)
        # the hessian is given by L_T*q_xx
        traj.stage_cost_xxs = self.steps_per_knot_array * traj.stage_cost_xxs
        traj.stage_cost_uus = self.steps_per_knot_array * traj.stage_cost_uus
        traj.stage_cost_xus = self.steps_per_knot_array * traj.stage_cost_xus

    def transit(
        self,
        traj: TrajInfoWrapper,
        k: int,  # knot step
    ) -> None:
        traj.every_traj.us = np.repeat(
            traj.us,
            self.steps_per_knot,
            axis=2,
        )
        for i in range(self.steps_per_knot[k]):
            self.dynamics.transit(traj.every_traj, self.time_step_on_knot[k] + i)
        traj.xs[:, :, k + 1] = traj.every_traj.xs[
            :, :, self.time_step_on_knot[k] + self.steps_per_knot[k]
        ]

    def calc_const(self, traj: TrajInfoWrapper) -> None:
        # eval constraint at knot
        self.const.calc(traj)
        # eval constraint at the constraint point
        #######################################################
        #                 | c(x[k_(T)], u[k_(T)])       |
        # C_T(X_T, U_T) = | c_i(x[k_(T)+j], u[k_(T)+j]) |
        #                 |            :                |
        #######################################################
        idx = self.lc_original
        for cp in self.const_points:
            self.const.calc_ith_const_at_(
                traj.every_traj,
                cp.const_indices,
                cp.time_steps_on_eval_point,
            )
            traj.cs[:, idx : idx + cp.const_num, :] = (
                traj.every_traj.cs[:, cp.const_indices, :]
            )[:, :, cp.time_steps_on_eval_point]
            idx += cp.const_num

    def calc_cost(self, traj: TrajInfoWrapper) -> None:
        # eval cost at knot step
        self.cost.calc_stage_cost(traj)
        self.cost.calc_terminal_cost(traj)
        # The Original stage cost is approximated by
        # Q_T(X_T, U_T) = L_T*q_k(T)(x[k_(T)], u[k_(T)])  --- (20)
        traj.costs = (
            np.sum(self.steps_per_knot_array * traj.stage_costs, axis=1)
            + traj.terminal_costs
        )

    def _reset_diff(self, traj: TrajInfoWrapper, t: int, calc_dynamics_hessian: bool):
        # ∂x[k+1]/∂x[k]
        ax_ax = traj.every_traj.fxs[:, :, t]
        # ∂x[k+1]/∂u[k]
        ax_au = traj.every_traj.fus[:, :, t]
        if calc_dynamics_hessian:
            # ∂2x[k+1]/∂x[k]∂x[k]
            # a2x_axx = traj.every_traj.fxxs[:, :, :, t]
            a2x_axx_reshaped = traj.every_traj.fxxs[:, :, :, t].reshape(self.cfg.n, -1)
            # ∂2x[k+1]/∂x[k]∂u[k]
            # a2x_axu = traj.every_traj.fxus[:, :, :, t]
            a2x_axu_reshaped = traj.every_traj.fxus[:, :, :, t].reshape(self.cfg.n, -1)
            # ∂2x[k+1]/∂u[k]∂u[k]
            # a2x_auu = traj.every_traj.fuus[:, :, :, t]
            a2x_auu_reshaped = traj.every_traj.fuus[:, :, :, t].reshape(self.cfg.n, -1)
        else:
            a2x_axx_reshaped = np.zeros(0)
            a2x_axu_reshaped = np.zeros(0)
            a2x_auu_reshaped = np.zeros(0)
        return ax_ax, ax_au, a2x_axx_reshaped, a2x_axu_reshaped, a2x_auu_reshaped

    def diff(self, traj: TrajInfoWrapper, calc_dynamics_hessian: bool) -> None:
        # calc fx, fu at every time steps
        self.dynamics.calc_jacobian(traj.every_traj)
        if calc_dynamics_hessian:
            self.dynamics.calc_hessian(traj.every_traj)
        # calc cx, cu, qx, qu, qxx, quu at knot steps
        self.const.calc_grad(traj)
        self.cost.calc_grad(traj)
        self.cost.calc_hessian(traj)
        # Since the Original stage cost is approximated by
        # Q_T(X_T, U_T) = L_T*q_k(T)(x[k_(T)], u[k_(T)])  --- (20)
        # the gradient is given by L_T*q_x
        traj.stage_cost_xs = self.steps_per_knot_array * traj.stage_cost_xs
        traj.stage_cost_us = self.steps_per_knot_array * traj.stage_cost_us

        # get gradients of constraint at constraint point
        for cp in self.const_points:
            # calc cx, cu at constrait point
            self.const.diff_at_(traj.every_traj, cp.time_steps_on_eval_point)

        for k in range(self.cfg.horizon):
            idx = self.lc_original
            for i in range(self.cfg.steps_per_knot[k]):
                t = self.time_step_on_knot[k] + i
                if i == 0:
                    (
                        ax_ax,
                        ax_au,
                        a2x_axx_reshaped,
                        a2x_axu_reshaped,
                        a2x_auu_reshaped,
                    ) = self._reset_diff(traj, t, calc_dynamics_hessian)
                else:
                    # calc gradient of constraint at constraint point with respect to knot state and input
                    for cp in self.const_points:
                        if i == cp.eval_time_step_from_knot:
                            # ∂c[k+i]/∂x[k] = ∂c[k+i]/∂x[k+i] * ∂x[k+i]/∂x[k]
                            traj.cxs[
                                idx : idx + cp.const_num,
                                :,
                                k,
                            ] = (
                                traj.every_traj.cxs[cp.const_indices, :, :]
                            )[:, :, t] @ ax_ax
                            # ∂c[k+i]/∂u[k] = ∂c[k+i]/∂x[k+i] * ∂x[k+i]/∂u[k] + ∂c[k+i]/∂u[k+i] * ∂u[k+i]/∂u[k]
                            traj.cus[
                                idx : idx + cp.const_num,
                                :,
                                k,
                            ] = (
                                traj.every_traj.cxs[cp.const_indices, :, :]
                            )[:, :, t] @ ax_au + (
                                traj.every_traj.cus[cp.const_indices, :, :]
                            )[
                                :, :, t
                            ]
                            idx += cp.const_num

                    # ∂x[k+i+1]/∂x[k] = ∂f/∂x[k+i] * ∂x[k+i]/∂x[k]
                    ax_ax_ = traj.every_traj.fxs[:, :, t] @ ax_ax  # type:ignore
                    # ∂x[k+i+1]/∂u[k] = ∂f/∂x[k+i] * ∂x[k+i]/∂u[k] + ∂f/∂u[k+i] * ∂u[k+i]/∂u[k]
                    ax_au_ = (
                        traj.every_traj.fxs[:, :, t] @ ax_au  # type:ignore
                        + traj.every_traj.fus[:, :, t]
                    )
                    if calc_dynamics_hessian:
                        # ∂2x[k+i+1]/∂x[k]∂x[k]
                        a2x_axx_reshaped = self._calc_a2x_axx(
                            a2x_axx_reshaped,
                            ax_ax,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                        )

                        # ∂2x[k+i+1]/∂x[k]∂u[k]
                        a2x_axu_reshaped = self._calc_a2x_axu(
                            a2x_axu_reshaped,
                            ax_ax,
                            ax_au,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                        )

                        # ∂2x[k+i+1]/∂u[k]∂u[k]
                        a2x_auu_reshaped = self._calc_a2x_auu(
                            a2x_auu_reshaped,
                            ax_au,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                            traj.every_traj.fuus[:, :, :, t],
                        )

                    ax_ax = ax_ax_
                    ax_au = ax_au_

            traj.fxs[:, :, k] = ax_ax
            traj.fus[:, :, k] = ax_au
            if calc_dynamics_hessian:
                traj.fxxs[:, :, :, k] = a2x_axx_reshaped.reshape(
                    self.cfg.n, self.cfg.n, self.cfg.n
                )
                traj.fxus[:, :, :, k] = a2x_axu_reshaped.reshape(
                    self.cfg.n, self.cfg.n, self.cfg.m
                )
                traj.fuus[:, :, :, k] = a2x_auu_reshaped.reshape(
                    self.cfg.n, self.cfg.m, self.cfg.m
                )

    def _calc_a2x_axx(
        self,
        a2x_axx_reshaped: np.ndarray,
        ax_ax: np.ndarray,
        fx: np.ndarray,
        fxx: np.ndarray,
    ) -> np.ndarray:
        # ∂^2x[k+i+1]/∂x[k]∂x[k]
        # = ∑ ∂x[k+i]/∂x[k]*∂^2x[k+i+1]/∂x[k+i]∂x[k+i]*∂x[k+i]/∂x[k] + ∑∂x[k+i+1]/∂x[k+i]*∂^2x[k+i]/∂x[k+i-1]∂x[k+i-1]
        # a2x_axx = self._einsum_separate(
        #     ax_ax,
        #     traj.every_traj.fxxs[:, :, :, t],
        #     ax_ax,
        # ) + np.einsum(
        #     "kp,pij->kij", traj.every_traj.fxs[:, :, t], a2x_axx
        # )
        if self.dynamics.has_nonzero_fxx:
            return self._einsum_separate(
                ax_ax,
                fxx,
                ax_ax,
            ).reshape(
                self.cfg.n, -1
            ) + (fx @ a2x_axx_reshaped)
        else:
            return a2x_axx_reshaped

    def _calc_a2x_axu(
        self,
        a2x_axu_reshaped: np.ndarray,
        ax_ax: np.ndarray,
        ax_au: np.ndarray,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
    ) -> np.ndarray:
        # ∂2x[k+i+1]/∂x[k]∂u[k]
        # a2x_axu_ = (
        #     self._einsum_separate(
        #         ax_au,
        #         traj.every_traj.fxxs[:, :, :, t],
        #         ax_ax,
        #     )
        #     # + np.einsum(
        #     #     "kpj,pi->kij", traj.every_traj.fxus[:, :, :, t], ax_ax
        #     # )
        #     + np.einsum(
        #         "pij,kp->kij", a2x_axu, traj.every_traj.fxs[:, :, t]
        #     )
        # )
        if self.dynamics.has_nonzero_fxu:
            return (
                self._einsum_separate(ax_au, fxx, ax_ax).reshape(self.cfg.n, -1)
                + np.einsum("kpj,pi->kij", fxu, ax_ax)
                + fx @ a2x_axu_reshaped
            )

        else:
            return (
                self._einsum_separate(ax_au, fxx, ax_ax).reshape(self.cfg.n, -1)
                + fx @ a2x_axu_reshaped
            )

    def _calc_a2x_auu(
        self,
        a2x_auu_reshaped: np.ndarray,
        ax_au: np.ndarray,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
        fuu: np.ndarray,
    ) -> np.ndarray:
        # ∂2x[k+i+1]/∂u[k]∂u[k]
        # a2x_auu_ = (
        #     self._einsum_separate(
        #         ax_au,
        #         traj.every_traj.fxxs[:, :, :, t],
        #         ax_au,
        #     )
        #     # + np.einsum(
        #     #     "kpj,pi->kij", traj.every_traj.fxus[:, :, :, t], ax_au
        #     # )
        #     # + np.einsum(
        #     #     "kqi,qj->kij", traj.every_traj.fxus[:, :, :, t], ax_au
        #     # )
        #     + np.einsum(
        #         "kp,pij->kij", traj.every_traj.fxs[:, :, t], a2x_auu
        #     )
        #     + traj.every_traj.fuus[:, :, :, t]
        # )
        if self.dynamics.has_nonzero_fxu and self.dynamics.has_nonzero_fuu:
            return (
                self._einsum_separate(
                    ax_au,
                    fxx,
                    ax_au,
                ).reshape(self.cfg.n, -1)
                + np.einsum(
                    "kpj,pi->kij",
                    fxu,
                    ax_au,
                )
                + np.einsum(
                    "kqi,qj->kij",
                    fxu,
                    ax_au,
                )
                + fx @ a2x_auu_reshaped
                + fuu.reshape(self.cfg.n, -1)
            )
        elif self.dynamics.has_nonzero_fxu:
            return (
                self._einsum_separate(
                    ax_au,
                    fxx,
                    ax_au,
                ).reshape(self.cfg.n, -1)
                + np.einsum(
                    "kpj,pi->kij",
                    fxu,
                    ax_au,
                )
                + np.einsum(
                    "kqi,qj->kij",
                    fxu,
                    ax_au,
                )
                + fx @ a2x_auu_reshaped
            )
        elif self.dynamics.has_nonzero_fuu:
            return (
                self._einsum_separate(
                    ax_au,
                    fxx,
                    ax_au,
                ).reshape(self.cfg.n, -1)
                + fx @ a2x_auu_reshaped
                + fuu.reshape(self.cfg.n, -1)
            )
        else:
            return (
                self._einsum_separate(
                    ax_au,
                    fxx,
                    ax_au,
                ).reshape(self.cfg.n, -1)
                + fx @ a2x_auu_reshaped
            )
