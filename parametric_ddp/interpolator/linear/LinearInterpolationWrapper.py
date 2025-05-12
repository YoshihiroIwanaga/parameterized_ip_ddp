import typing

import numpy as np

from parametric_ddp.interpolator.linear.LinearInterpolationConfigWrapper import (
    LinearInterpolationConfigWrapper,
)
from parametric_ddp.utils.ConstraintPoint import ConstraintPoint
from parametric_ddp.utils.OptimalControlProblem import OptimalControlProblem
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.abstract.Constraint import Constraint
from problem_setting.abstract.Cost import Cost
from problem_setting.abstract.Dynamics import Dynamics


class LinearInterpolationWrapper(OptimalControlProblem):
    def __init__(
        self,
        dynamics: Dynamics,
        knot_const: Constraint,
        original_const: Constraint,
        cost: Cost,
        cfg: LinearInterpolationConfigWrapper,
        extra_const_indices: typing.List[typing.List[int]] = [[]],
        extra_const_eval_time_steps_from_knot: typing.List[int] = [],
    ) -> None:
        self.dynamics: Dynamics = dynamics
        self.knot_const: Constraint = knot_const
        self.original_const: Constraint = original_const
        self.cost: Cost = cost
        self.cfg: LinearInterpolationConfigWrapper = cfg
        self.n: int = cfg.n
        self.m: int = cfg.m
        self.n_org: int = self.cfg.org_cfg.n
        self.m_org: int = self.cfg.org_cfg.m
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
        self.lc: int = knot_const.lc
        self.lc_terminal: int = knot_const.lc_terminal
        self.lc_on_knot: int = knot_const.lc
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
        self.knot_const.set_constant(traj)
        self.cost.set_constant(traj)
        # inter-knot dynamics is given by
        # F_T(X_T, U_T) =  | f(f(f...f(x, u))) | } n
        #                  | u + L_T*Δt*ν      | } m
        # therfore,
        # F_X =  | *   * | } n
        #        | 0   I | } m
        for i in range(self.m):
            traj.fxs[self.cfg.org_cfg.n + i, self.cfg.org_cfg.n + i, :] = 1.0
        # F_U =  |      *     | } n
        #        | L_T*Δt*I   | } m
        for i, step_per_knot in enumerate(self.steps_per_knot):
            traj.fus[-self.m :, -self.m :, i] = (
                self.cfg.dt * step_per_knot * np.eye(self.m)
            )
        self.original_const.set_constant(traj.every_traj)
        traj.stage_cost_xxs = self.steps_per_knot_array * traj.stage_cost_xxs
        traj.stage_cost_uus = self.steps_per_knot_array * traj.stage_cost_uus
        traj.stage_cost_xus = self.steps_per_knot_array * traj.stage_cost_xus

    def transit(
        self,
        traj: TrajInfoWrapper,
        k: int,  # knot step
    ) -> None:
        for i in range(self.steps_per_knot[k]):
            traj.every_traj.us[:, :, self.time_step_on_knot[k] + i] = (
                traj.xs[:, self.cfg.org_cfg.n :, k] + i * self.cfg.dt * traj.us[:, :, k]
            )
            self.dynamics.transit(
                traj.every_traj,
                self.time_step_on_knot[k] + i,
            )
        # X[T+1] = x(k(T) + L_T)
        traj.xs[:, : self.cfg.org_cfg.n, k + 1] = traj.every_traj.xs[
            :,
            : self.cfg.org_cfg.n,
            self.time_step_on_knot[k] + self.steps_per_knot[k],
        ]
        # s[T+1] = u[k(T)] + L_T*Δt*ν
        traj.xs[:, self.cfg.org_cfg.n :, k + 1] = (
            traj.xs[:, self.cfg.org_cfg.n :, k]
            + self.steps_per_knot[k] * self.cfg.dt * traj.us[:, :, k]
        )

    def calc_const(self, traj: TrajInfoWrapper) -> None:
        # eval constraint at knot
        self.knot_const.calc(traj)
        # eval constraint at the constraint point
        #######################################################
        #                 | c(x[k_(T)], u[k_(T)])       |
        # C_T(X_T, U_T) = | c_i(x[k_(T)+j], u[k_(T)+j]) |
        #                 |            :                |
        #######################################################
        idx = self.lc_on_knot
        for cp in self.const_points:
            self.original_const.calc_ith_const_at_(
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
        traj.costs = (
            np.sum(self.steps_per_knot_array * traj.stage_costs, axis=1)
            + traj.terminal_costs
        )

    def _reset_diff(self, traj: TrajInfoWrapper, t: int, calc_dynamics_hessian: bool):
        # ∂x[k+1]/∂x[k] ∈ R^{n*n}
        ax_ax = traj.every_traj.fxs[:, :, t]
        # ∂x[k+1]/∂s[k] = ∂x[k+1]/∂u[k] ∂u[k]/∂s[k] ∈ R^{n*m}
        ax_as = traj.every_traj.fus[:, :, t]
        # ∂x[k+1]/∂ν[k] ∈ R^{n*m}
        ax_av = np.zeros(
            (
                self.n_org,
                self.m_org,
            )
        )
        if calc_dynamics_hessian:
            a2x_axx_reshaped = traj.every_traj.fxxs[:, :, :, t].reshape(self.n_org, -1)
            a2x_axs_reshaped = traj.every_traj.fxus[:, :, :, t].reshape(self.n_org, -1)
            a2x_ass_reshaped = traj.every_traj.fuus[:, :, :, t].reshape(self.n_org, -1)
            a2x_avv_reshaped = np.zeros((self.n_org, self.m_org * self.m_org))
            a2x_axv_reshaped = np.zeros((self.n_org, self.n_org * self.m_org))
            a2x_asv_reshaped = np.zeros((self.n_org, self.m_org * self.m_org))
        else:
            a2x_axx_reshaped = np.zeros(0)
            a2x_axs_reshaped = np.zeros(0)
            a2x_ass_reshaped = np.zeros(0)
            a2x_avv_reshaped = np.zeros(0)
            a2x_axv_reshaped = np.zeros(0)
            a2x_asv_reshaped = np.zeros(0)
        return (
            ax_ax,
            ax_as,
            ax_av,
            a2x_axx_reshaped,
            a2x_axs_reshaped,
            a2x_ass_reshaped,
            a2x_avv_reshaped,
            a2x_axv_reshaped,
            a2x_asv_reshaped,
        )

    def diff(self, traj: TrajInfoWrapper, calc_dynamics_hessian: bool) -> None:
        # calc fx, fu at every time steps
        self.dynamics.calc_jacobian(traj.every_traj)
        if calc_dynamics_hessian:
            self.dynamics.calc_hessian(traj.every_traj)
        # calc cx, cu, qx, qu, qxx, quu at knot steps
        self.knot_const.calc_grad(traj)
        self.cost.calc_grad(traj)
        self.cost.calc_hessian(traj)
        traj.stage_cost_xs = self.steps_per_knot_array * traj.stage_cost_xs
        traj.stage_cost_us = self.steps_per_knot_array * traj.stage_cost_us

        # get gradients of constraint at constraint point
        for cp in self.const_points:
            # calc cx, cu at constrait point
            self.original_const.diff_at_(traj.every_traj, cp.time_steps_on_eval_point)

        for k in range(self.cfg.horizon):
            idx = self.lc_on_knot
            for i in range(self.cfg.steps_per_knot[k]):
                t = self.time_step_on_knot[k] + i
                if i == 0:
                    (
                        ax_ax,
                        ax_as,
                        ax_av,
                        a2x_axx_reshaped,
                        a2x_axs_reshaped,
                        a2x_ass_reshaped,
                        a2x_avv_reshaped,
                        a2x_axv_reshaped,
                        a2x_asv_reshaped,
                    ) = self._reset_diff(traj, t, calc_dynamics_hessian)
                else:
                    # calc gradient of constraint at constraint point with respect to knot state and input
                    for cp in self.const_points:
                        if i == cp.eval_time_step_from_knot:
                            # ∂c[k+i]/∂X[k] = |∂c[k+i]/∂x[k+i] * ∂x[k+i]/∂x[k], ∂c[k+i]/∂x[k+i] * ∂x[k+i]/∂s[k] + ∂c[k+i]/∂u[k+i] * ∂u[k+i]/∂s[k]|
                            traj.cxs[
                                idx : idx + cp.const_num,
                                :,
                                k,
                            ] = np.hstack(
                                (
                                    (traj.every_traj.cxs[cp.const_indices, :, :])[
                                        :, :, t
                                    ]
                                    @ ax_ax,
                                    (traj.every_traj.cxs[cp.const_indices, :, :])[
                                        :, :, t
                                    ]
                                    @ ax_as
                                    + (traj.every_traj.cus[cp.const_indices, :, :])[
                                        :, :, t
                                    ],
                                )
                            )
                            # ∂c[k+i]/∂U[k] = |∂c[k+i]/∂x[k+i] * ∂x[k+i]/∂ν[k] + ∂c[k+i]/∂u[k+i] * ∂u[k+i]/∂ν[k]|
                            traj.cus[
                                idx : idx + cp.const_num,
                                :,
                                k,
                            ] = (
                                traj.every_traj.cxs[cp.const_indices, :, :]
                            )[:, :, t] @ ax_av + (
                                traj.every_traj.cus[cp.const_indices, :, :]
                            )[
                                :, :, t
                            ] @ (
                                self.cfg.dt * i * np.eye(self.m)
                            )
                            idx += cp.const_num

                    # ∂x[k+i+1]/∂x[k] = ∂f/∂x[k+i] * ∂x[k+i]/∂x[k]
                    ax_ax_ = traj.every_traj.fxs[:, :, t] @ ax_ax  # type:ignore
                    # ∂x[k+i+1]/∂s[k] = ∂f/∂x[k+i] * ∂x[k+i]/∂s[k] + ∂f/∂u[k+i] * ∂u[k+i]/∂s[k]
                    ax_as_ = (
                        traj.every_traj.fxs[:, :, t] @ ax_as  # type:ignore
                        + traj.every_traj.fus[:, :, t]
                    )
                    # ∂x[k+i+1]/∂ν[k] = ∂f/∂x[k+i] * ∂x[k+i]/∂ν[k] + ∂f/∂u[k+i] * ∂u[k+i]/∂ν[k]
                    # ∂u[k+i]/∂ν[k] = i * Δt * I
                    ax_av_ = traj.every_traj.fxs[
                        :, :, t
                    ] @ ax_av + traj.every_traj.fus[  # type:ignore
                        :, :, t
                    ] @ (
                        self.time_step_from_knots[t] * self.cfg.dt * np.eye(self.m)
                    )
                    if calc_dynamics_hessian:
                        # ∂2x[k+i+1]/∂x[k]∂x[k]
                        a2x_axx_reshaped = self._calc_a2x_axx(
                            a2x_axx_reshaped,
                            ax_ax,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                        )
                        # ∂2x[k+i+1]/∂x[k]∂s[k]
                        a2x_axs_reshaped = self._calc_a2x_axs(
                            a2x_axs_reshaped,
                            ax_ax,
                            ax_as,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                        )
                        # ∂2x[k+i+1]/∂s[k]∂s[k]
                        a2x_ass_reshaped = self._calc_a2x_ass(
                            a2x_ass_reshaped,
                            ax_as,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                            traj.every_traj.fuus[:, :, :, t],
                        )
                        # ∂2x[k+i+1]/∂v[k]∂v[k]
                        a2x_avv_reshaped = self._calc_a2x_avv(
                            a2x_avv_reshaped,
                            ax_av,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fus[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                            traj.every_traj.fuus[:, :, :, t],
                        )

                        a2x_axv_reshaped = self._calc_a2x_axv(
                            a2x_axv_reshaped,
                            ax_ax,
                            ax_av,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                        )
                        a2x_asv_reshaped = self.calc_a2x_asv(
                            a2x_asv_reshaped,
                            ax_av,
                            ax_as,
                            traj.every_traj.fxs[:, :, t],
                            traj.every_traj.fxxs[:, :, :, t],
                            traj.every_traj.fxus[:, :, :, t],
                        )
                    ax_ax = ax_ax_
                    ax_as = ax_as_
                    ax_av = ax_av_

            # F_X =  | ax_ax   ax_as | } n
            #        |    0      I   | } m
            traj.fxs[: self.cfg.org_cfg.n, : self.cfg.org_cfg.n, k] = (
                ax_ax
            )  # type:ignore
            traj.fxs[: self.cfg.org_cfg.n, self.cfg.org_cfg.n :, k] = (
                ax_as
            )  # type:ignore
            # F_U =  |    ax_av   | } n
            #        | L_T*Δt*ν*I | } m
            traj.fus[: self.cfg.org_cfg.n, :, k] = ax_av  # type:ignore
            if calc_dynamics_hessian:
                traj.fxxs[
                    : self.cfg.org_cfg.n, : self.cfg.org_cfg.n, : self.cfg.org_cfg.n, k
                ] = a2x_axx_reshaped.reshape(self.n_org, self.n_org, self.n_org)
                traj.fxxs[
                    : self.cfg.org_cfg.n, self.cfg.org_cfg.n :, self.cfg.org_cfg.n :, k
                ] = a2x_ass_reshaped.reshape(self.n_org, self.m_org, self.m_org)
                traj.fxxs[
                    : self.cfg.org_cfg.n, : self.cfg.org_cfg.n, self.cfg.org_cfg.n :, k
                ] = a2x_axs_reshaped.reshape(self.n_org, self.n_org, self.m_org)
                traj.fxxs[
                    : self.cfg.org_cfg.n, self.cfg.org_cfg.n :, : self.cfg.org_cfg.n, k
                ] = a2x_axs_reshaped.reshape(self.n_org, self.m_org, self.n_org)
                traj.fxus[: self.cfg.org_cfg.n, : self.cfg.org_cfg.n, :, k] = (
                    a2x_axv_reshaped.reshape(self.n_org, self.n_org, self.m_org)
                )
                traj.fxus[: self.cfg.org_cfg.n, self.cfg.org_cfg.n :, :, k] = (
                    a2x_asv_reshaped.reshape(self.n_org, self.m_org, self.m_org)
                )
                traj.fuus[: self.cfg.org_cfg.n, :, :, k] = a2x_avv_reshaped.reshape(
                    self.n_org, self.m_org, self.m_org
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
                self.n_org, -1
            ) + (fx @ a2x_axx_reshaped)
        else:
            return a2x_axx_reshaped

    def _calc_a2x_axs(
        self,
        a2x_axs_reshaped: np.ndarray,
        ax_ax: np.ndarray,
        ax_as: np.ndarray,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
    ) -> np.ndarray:
        if self.dynamics.has_nonzero_fxx and self.dynamics.has_nonzero_fxu:
            return (
                self._einsum_separate(
                    ax_as,
                    fxx,
                    ax_ax,
                ).reshape(self.n_org, -1)
                + np.einsum(
                    "pj,gwp,wi->gij",
                    np.eye(self.m),  # au_as
                    fxu,
                    ax_ax,
                )
                + fx @ a2x_axs_reshaped
            )
        elif self.dynamics.has_nonzero_fxx:
            return (
                self._einsum_separate(
                    ax_as,
                    fxx,
                    ax_ax,
                ).reshape(self.n_org, -1)
                + fx @ a2x_axs_reshaped
            )
        else:
            return (
                np.einsum(
                    "pj,gwp,wi->gij",
                    np.eye(self.m),  # au_as
                    fxu,
                    ax_ax,
                ).reshape(self.n_org, -1)
                + fx @ a2x_axs_reshaped
            )

    def _calc_a2x_ass(
        self,
        a2x_ass_reshaped: np.ndarray,
        ax_as: np.ndarray,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
        fuu: np.ndarray,
    ) -> np.ndarray:
        return (
            self._einsum_separate(
                ax_as,
                fxx,
                ax_as,
            ).reshape(self.n_org, -1)
            # + np.einsum(
            #     "rj,gpr,pi->gij",
            #     np.eye(self.m),
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_as,
            # )  # au_as
            + fx @ a2x_ass_reshaped
            # + np.einsum(
            #     "pi,gqp,qj->gij",
            #     np.eye(self.m),
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_as,
            # )
            # + np.einsum(
            #     "pj,gwp,wi->gij",
            #     np.eye(self.m),
            #     traj.every_traj.fuus[:, :, :, t],
            #     np.eye(self.m),
            # )  # au_as
            # + np.einsum(
            #     "gw,wij->gij",
            #     traj.every_traj.fus[:, :, t],
            #     np.zeros((self.m, self.m, self.m)),
            # )  # a2u_as2
        )

    def _calc_a2x_avv(
        self,
        a2x_avv_reshaped: np.ndarray,
        ax_av: np.ndarray,
        fx: np.ndarray,
        fu: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
        fuu: np.ndarray,
    ):
        return (
            self._einsum_separate(
                ax_av,
                fxx,
                ax_av,
            ).reshape(self.n_org, -1)
            # + np.einsum(
            #     "rj,gpr,pi->gij",
            #     self.time_step_from_knots[t]
            #     * self.cfg.dt
            #     * np.eye(self.m),
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_av,
            # )  # au_av
            + fx @ a2x_avv_reshaped
            # + np.einsum(
            #     "pi,gqp,qj->gij",
            #     self.time_step_from_knots[t]
            #     * self.cfg.dt
            #     * np.eye(self.m),
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_av,
            # )
            # + np.einsum(
            #     "pj,gwp,wi->gij",
            #     self.time_step_from_knots[t]
            #     * self.cfg.dt
            #     * np.eye(self.m),
            #     traj.every_traj.fuus[:, :, :, t],
            #     self.time_step_from_knots[t]
            #     * self.cfg.dt
            #     * np.eye(self.m),
            # )
            # + np.einsum(
            #     "gw,wij->gij",
            #     traj.every_traj.fus[:, :, t],
            #     np.zeros((self.m, self.m, self.m)),
            # )  # a2u_av2
        )

    def _calc_a2x_axv(
        self,
        a2x_axv_reshaped: np.ndarray,
        ax_ax: np.ndarray,
        ax_av: np.ndarray,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
    ):
        return (
            self._einsum_separate(
                ax_av,
                fxx,
                ax_ax,
            ).reshape(self.n_org, -1)
            # + np.einsum(
            #     "pj,gwp,wi->gij",
            #     self.time_step_from_knots[t]
            #     * self.cfg.dt
            #     * np.eye(self.m),
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_ax,
            # )  # au_av
            + fx @ a2x_axv_reshaped
        )

    def calc_a2x_asv(
        self,
        a2x_asv_reshaped: np.ndarray,
        ax_av: np.ndarray,
        ax_as,
        fx: np.ndarray,
        fxx: np.ndarray,
        fxu: np.ndarray,
    ):
        return (
            self._einsum_separate(
                ax_av,
                fxx,
                ax_as,
            ).reshape(self.n_org, -1)
            # + np.einsum(
            #     "rj,gpr,pi->gij",
            #     self.time_step_from_knots[t]
            #     * self.cfg.dt
            #     * np.eye(self.m),
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_as,
            # )
            + fx @ a2x_asv_reshaped
            # + np.einsum(
            #     "pi,gqp,qj->gij",
            #     np.eye(self.m),
            #     traj.every_traj.fxus[:, :, :, t],
            #     ax_av,
            # )
            # + np.einsum(
            #     "pj,gwp,wi->gij",
            #     self.time_step_from_knots[t]
            #     * self.cfg.dt
            #     * np.eye(self.m),
            #     traj.every_traj.fuus[:, :, :, t],
            #     np.eye(self.m),
            # )
            # + np.einsum(
            #     "gw,wij->gij",
            #     traj.every_traj.fus[:, :, t],
            #     np.zeros((self.m, self.m, self.m)),
            # )  # a2u_asau
        )
