import typing

import numpy as np

from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config


class Constraint:
    def __init__(self, cfg: Config) -> None:
        self.n: int = cfg.n
        self.m: int = cfg.m
        self.x_max: np.ndarray = cfg.x_max
        self.x_min: np.ndarray = cfg.x_min
        self.u_max: np.ndarray = cfg.u_max
        self.u_min: np.ndarray = cfg.u_min
        self.idx_x_min: list[int] = cfg.idx_x_min
        self.idx_x_max: list[int] = cfg.idx_x_max
        self.idx_u_min: list[int] = cfg.idx_u_min
        self.idx_u_max: list[int] = cfg.idx_u_max
        # constranit number
        self.lc: int = (
            len(self.idx_x_min)
            + len(self.idx_x_max)
            + len(self.idx_u_min)
            + len(self.idx_u_max)
        )
        # constranit number on terminal time step
        self.lc_terminal: int = 2 * len(cfg.terminal_const_state_idx)
        self.terminal_const_state_idx: list[int] = cfg.terminal_const_state_idx
        self.margin: np.ndarray = np.zeros((1, self.lc, 1))

    def set_constant(
        self,
        traj: TrajInfo,
    ) -> None:
        const_idx = 0
        for state_idx in self.idx_x_max:
            traj.cxs[const_idx, state_idx, :] = 1
            const_idx += 1
        for input_idx in self.idx_u_max:
            traj.cus[const_idx, input_idx, :] = 1
            const_idx += 1

        for state_idx in self.idx_x_min:
            traj.cxs[const_idx, state_idx, :] = -1
            const_idx += 1
        for input_idx in self.idx_u_min:
            traj.cus[const_idx, input_idx, :] = -1
            const_idx += 1

        const_idx = 0
        for state_idx in self.terminal_const_state_idx:
            traj.cxs_terminal[const_idx, state_idx, :] = 1
            const_idx += 1
        for state_idx in self.terminal_const_state_idx:
            traj.cxs_terminal[const_idx, state_idx, :] = -1
            const_idx += 1

    def set_margin(self, margin) -> None:
        self.margin = margin

    def calc(
        self,
        traj: TrajInfo,
    ) -> None:
        traj.cs[:, : self.lc, :] = (
            np.hstack(
                (
                    traj.xs[:, self.idx_x_max, :-1] - self.x_max[self.idx_x_max, 0:1],
                    traj.us[:, self.idx_u_max, :] - self.u_max[self.idx_u_max, 0:1],
                    -traj.xs[:, self.idx_x_min, :-1] + self.x_min[self.idx_x_min, 0:1],
                    -traj.us[:, self.idx_u_min, :] + self.u_min[self.idx_u_min, 0:1],
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

    def calc_grad(
        self,
        traj: TrajInfo,
    ) -> None:
        pass

    def calc_ith_const_at_(
        self,
        traj: TrajInfo,
        const_indices: typing.List[int],
        time_steps: typing.List[int],
    ) -> None:
        # evaluate constraints at designated indices at designated time steps
        traj.cs[:, : self.lc, time_steps] = (
            np.hstack(
                (
                    (traj.xs[:, self.idx_x_max, :])[:, :, time_steps]
                    - self.x_max[self.idx_x_max, 0:1],
                    (traj.us[:, self.idx_u_max, :])[:, :, time_steps]
                    - self.u_max[self.idx_u_max, 0:1],
                    -(traj.xs[:, self.idx_x_min, :])[:, :, time_steps]
                    + self.x_min[self.idx_x_min, 0:1],
                    -(traj.us[:, self.idx_u_min, :])[:, :, time_steps]
                    + self.u_min[self.idx_u_min, 0:1],
                )
            )
            + self.margin
        )

    def diff_at_(
        self,
        traj: TrajInfo,
        time_steps: typing.List[int],
    ) -> None:
        # evaluate grad of constraints at designated indices at designated time steps
        pass
