import numpy as np

from parametric_ddp.ip_ddp.BackwardPass import BackwardPassResult
from parametric_ddp.utils.OptimalControlProblem import OptimalControlProblem
from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config


class ForwardPass:
    def __init__(self, ocp: OptimalControlProblem, cfg: Config) -> None:
        self.ocp: OptimalControlProblem = ocp
        self.horizon: int = cfg.horizon
        self.n: int = cfg.n
        self.m: int = cfg.m
        self.tol: float = cfg.tol
        self.step_sizes: np.ndarray = (
            2 ** np.linspace(0, -14, cfg.step_size_num)  # type: ignore
        ).reshape((cfg.step_size_num, 1))
        self.free_state_idx: list[int] = cfg.free_state_idx
        self.terminal_const_state_idx: list[int] = cfg.terminal_const_state_idx
        self.failed: bool = False

    def reset_filter(self, traj: TrajInfo, barrier_param: float) -> None:
        log_cost = traj.costs[traj.traj_idx] - barrier_param * np.sum(
            np.log(-traj.cs[traj.traj_idx, :, :].reshape((1, -1)))
        )
        self.filter_ = log_cost
        self.failed = False

    def _update_traj(
        self, traj: TrajInfo, traj_pre: TrajInfo, bp_result: BackwardPassResult
    ) -> None:
        for t in range(self.horizon):
            if t == 0:
                if len(self.free_state_idx) > 0:
                    traj.set_x0(
                        traj_pre.xs[
                            traj_pre.traj_idx : traj_pre.traj_idx + 1,
                            :,
                            t,
                        ]
                        + self.step_sizes * bp_result.kx0
                    )
                traj.duals[:, :, t] = (
                    traj_pre.duals[traj_pre.traj_idx : traj_pre.traj_idx + 1, :, t]
                    + self.step_sizes * bp_result.kdual[:, t]
                )
                traj.us[:, :, t] = (
                    traj_pre.us[traj_pre.traj_idx : traj_pre.traj_idx + 1, :, t]
                    + self.step_sizes * bp_result.ku[:, t]
                )
            else:
                delta_x = (
                    traj.xs[:, :, t : t + 1]
                    - traj_pre.xs[
                        traj_pre.traj_idx : traj_pre.traj_idx + 1, :, t : t + 1
                    ]
                )
                if t == self.horizon - 1 and len(self.terminal_const_state_idx) > 0:
                    traj.dual_terminal[:, :, 0] = (
                        traj_pre.dual_terminal[
                            traj_pre.traj_idx : traj_pre.traj_idx + 1, :, 0
                        ]
                        + self.step_sizes * bp_result.kdual_terminal[:, 0]
                        + (bp_result.Kdual_terminal @ delta_x)[:, :, 0]
                    )
                traj.duals[:, :, t] = (
                    traj_pre.duals[traj_pre.traj_idx : traj_pre.traj_idx + 1, :, t]
                    + self.step_sizes * bp_result.kdual[:, t]
                    + (bp_result.Kdual[:, :, t] @ delta_x)[:, :, 0]
                )
                traj.us[:, :, t] = (
                    traj_pre.us[traj_pre.traj_idx : traj_pre.traj_idx + 1, :, t]
                    + self.step_sizes * bp_result.ku[:, t]
                    + (bp_result.Ku[:, :, t] @ delta_x)[:, :, 0]
                )
            self.ocp.transit(traj, t)
        self.ocp.calc_const(traj)
        self.ocp.calc_cost(traj)

    def _overwrite_cost_of_invalid_traj(
        self, traj: TrajInfo, traj_pre: TrajInfo, barrier_param: float
    ) -> None:
        tau = max((0.99, 1 - barrier_param))
        traj.costs[
            np.nonzero(
                traj.cs
                > (1 - tau)
                * traj_pre.cs[traj_pre.traj_idx : traj_pre.traj_idx + 1, :, :]
            )[0]
        ] = np.inf
        traj.costs[
            np.nonzero(
                traj.duals
                < (1 - tau)
                * traj_pre.duals[traj_pre.traj_idx : traj_pre.traj_idx + 1, :, :]
            )[0]
        ] = np.inf
        if len(self.terminal_const_state_idx) > 0:
            traj.costs[
                np.nonzero(
                    traj.cs_terminal
                    > (1 - tau)
                    * traj_pre.cs_terminal[
                        traj_pre.traj_idx : traj_pre.traj_idx + 1, :, :
                    ]
                )[0]
            ] = np.inf

            traj.costs[
                np.nonzero(
                    traj.dual_terminal
                    < (1 - tau)
                    * traj_pre.dual_terminal[
                        traj_pre.traj_idx : traj_pre.traj_idx + 1, :, :
                    ]
                )[0]
            ] = np.inf

    def _select_best_traj(self, traj: TrajInfo, barrier_param: float) -> None:
        traj.set_traj_idx(np.nanargmin(traj.costs))  # type:ignore
        cost = traj.costs[traj.traj_idx]
        log_cost = cost - barrier_param * np.sum(np.log(-traj.cs[traj.traj_idx, :, :]))
        if log_cost >= self.filter_:
            self.failed = True
        else:
            self.filter_ = log_cost

    def update(
        self,
        traj: TrajInfo,
        traj_pre: TrajInfo,
        bp_result: BackwardPassResult,
        barrier_param: float,
    ) -> None:
        self.failed = False
        self._update_traj(traj, traj_pre, bp_result)
        self._overwrite_cost_of_invalid_traj(traj, traj_pre, barrier_param)
        if np.all(np.isinf(traj.costs)):
            self.failed = True
            return
        self._select_best_traj(traj, barrier_param)
