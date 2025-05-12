import numpy as np

from parametric_ddp.interpolator.cubic.CubicInterpolationConfigWrapper import (
    CubicInterpolationConfigWrapper,
)
from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Constraint import Constraint


class Constraint(Constraint):
    def __init__(self, cfg: CubicInterpolationConfigWrapper) -> None:
        self.n: int = cfg.n
        self.m: int = cfg.m
        self.org_n: int = cfg.org_cfg.n
        self.org_m: int = cfg.org_cfg.m
        self.x_max: np.ndarray = cfg.x_max
        self.x_min: np.ndarray = cfg.x_min
        self.u_max: np.ndarray = cfg.u_max
        self.u_min: np.ndarray = cfg.u_min
        self.org_u_max: np.ndarray = cfg.org_cfg.u_max
        self.org_u_min: np.ndarray = cfg.org_cfg.u_min
        self.idx_x_min = cfg.idx_x_min
        self.idx_x_max = cfg.idx_x_max
        self.idx_ex_x_min = cfg.idx_ex_x_min
        self.idx_ex_x_max = cfg.idx_ex_x_max
        self.idx_u_min = cfg.idx_u_min
        self.idx_u_max = cfg.idx_u_max
        self.org_idx_u_min = cfg.org_cfg.idx_u_min
        self.org_idx_u_max = cfg.org_cfg.idx_u_max
        self.lc: int = (
            len(self.idx_x_min)
            + len(self.idx_x_max)
            + len(self.idx_ex_x_min)
            + len(self.idx_ex_x_max)
            + len(self.idx_u_min)
            + len(self.idx_u_max)
            + len(self.org_idx_u_min)
            + len(self.org_idx_u_max)
        )
        self.lc_terminal = 2 * len(cfg.terminal_const_state_idx)
        self.terminal_const_state_idx = cfg.terminal_const_state_idx
        self.margin: np.ndarray = np.zeros((1, self.lc, 1))
        self.LT_dt: np.ndarray = cfg.dt * np.array(
            cfg.steps_per_knot
        )  # dt*[L_0, L_1,..., L_{Te-1}] ∈ N^{Te}

    def set_margin(self, margin: np.ndarray) -> None:
        self.margin = margin

    def set_constant(
        self,
        traj: TrajInfo,
    ) -> None:
        const_idx = 0
        for state_idx in self.idx_x_max:
            traj.cxs[const_idx, state_idx, :] = 1
            const_idx += 1
        for state_idx in self.idx_x_min:
            traj.cxs[const_idx, state_idx, :] = -1
            const_idx += 1

        for state_idx in self.idx_ex_x_max:
            traj.cxs[const_idx, state_idx, :] = 1
            const_idx += 1
        for state_idx in self.idx_ex_x_min:
            traj.cxs[const_idx, state_idx, :] = -1
            const_idx += 1
        for input_idx in self.idx_u_max:
            traj.cus[const_idx, input_idx, :] = 1
            const_idx += 1
        for input_idx in self.idx_u_min:
            traj.cus[const_idx, input_idx, :] = -1
            const_idx += 1
        for i, input_idx in enumerate(self.org_idx_u_max):
            traj.cxs[const_idx, self.org_n + i, :] = 1
            traj.cxs[const_idx, self.org_n + self.org_m + i, :] = 0.5 * self.LT_dt
            const_idx += 1

        const_idx = 0
        for state_idx in self.terminal_const_state_idx:
            traj.cxs_terminal[const_idx, state_idx, :] = 1
            const_idx += 1
        for state_idx in self.terminal_const_state_idx:
            traj.cxs_terminal[const_idx, state_idx, :] = -1
            const_idx += 1

    def calc(self, traj: TrajInfo) -> None:
        u_post_wo_mu = (
            traj.xs[:, self.org_n : self.org_n + self.org_m, :-1]
            + 0.5
            * self.LT_dt
            * traj.xs[:, self.org_n + self.org_m : self.org_n + 2 * self.org_m, :-1]
        )
        traj.cs[:, : self.lc, :] = (
            np.hstack(
                (
                    traj.xs[:, self.idx_x_max, :-1] - self.x_max[self.idx_x_max, 0:1],
                    -traj.xs[:, self.idx_x_min, :-1] + self.x_min[self.idx_x_min, 0:1],
                    traj.xs[:, self.idx_ex_x_max, :-1]
                    - self.x_max[self.idx_ex_x_max, 0:1],
                    -traj.xs[:, self.idx_ex_x_min, :-1]
                    + self.x_min[self.idx_ex_x_min, 0:1],
                    traj.us[:, self.idx_u_max, :] - self.u_max[self.idx_u_max, 0:1],
                    -traj.us[:, self.idx_u_min, :] + self.u_min[self.idx_u_min, 0:1],
                    u_post_wo_mu - self.org_u_max,
                    -1 * u_post_wo_mu + self.org_u_min,
                )
            )
            + self.margin
        )

        if self.lc_terminal > 0:
            traj.cs_terminal = np.hstack(
                (
                    traj.xs[:, self.terminal_const_state_idx, -1:]  # type:ignore
                    - self.x_max[self.terminal_const_state_idx, 0:1],  # type:ignore
                    -traj.xs[:, self.terminal_const_state_idx, -1:]
                    + self.x_min[self.terminal_const_state_idx, 0:1],
                )
            )
