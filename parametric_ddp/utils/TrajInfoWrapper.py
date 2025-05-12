import typing

import numpy as np

from parametric_ddp.interpolator.cubic.CubicInterpolationConfigWrapper import (
    CubicInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.linear.LinearInterpolationConfigWrapper import (
    LinearInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.zero_order.ZeroOrderHolderConfigWrapper import (
    ZeroOrderHolderConfigWrapper,
)
from parametric_ddp.utils.TrajInfo import TrajInfo


class TrajInfoWrapper(TrajInfo):
    def __init__(
        self,
        cfg: typing.Union[
            ZeroOrderHolderConfigWrapper,
            LinearInterpolationConfigWrapper,
            CubicInterpolationConfigWrapper,
        ],
        lc: int,
        lc_terminal: int,
        lc_original: int,
        K: int,  # trajectory num
    ) -> None:
        self.n_org = cfg.org_cfg.n
        self.every_traj = TrajInfo(cfg.org_cfg, lc_original, 0, K)
        self.traj_idx: int = 0
        self.xs: np.ndarray = np.zeros((K, cfg.n, cfg.horizon + 1))
        self.xs[:, :, 0] = cfg.x_ini
        self.us: np.ndarray = np.zeros((K, cfg.m, cfg.horizon))
        self.us[:, :, :] = cfg.u_inis
        self.cs: np.ndarray = np.zeros((K, lc, cfg.horizon))
        self.cs_terminal: np.ndarray = np.zeros((K, lc_terminal, 1))
        self.stage_costs: np.ndarray = np.zeros((K, cfg.horizon))
        self.terminal_costs: np.ndarray = np.zeros((K,))
        self.costs: np.ndarray = np.zeros((K,))
        self.duals: np.ndarray = 0.01 * np.ones((K, lc, cfg.horizon))
        self.dual_terminal = 0.01 * np.ones((K, lc_terminal, 1))
        self.fxs: np.ndarray = np.zeros((cfg.n, cfg.n, cfg.horizon))
        self.fus: np.ndarray = np.zeros((cfg.n, cfg.m, cfg.horizon))
        self.fxxs: np.ndarray = np.zeros((cfg.n, cfg.n, cfg.n, cfg.horizon))
        self.fxus: np.ndarray = np.zeros((cfg.n, cfg.n, cfg.m, cfg.horizon))
        self.fuus: np.ndarray = np.zeros((cfg.n, cfg.m, cfg.m, cfg.horizon))
        self.cxs = np.zeros((lc, cfg.n, cfg.horizon))  # ∈R^{lc,n,te}
        self.cus = np.zeros((lc, cfg.m, cfg.horizon))  # ∈R^{lc,m,te}
        self.cxs_terminal = np.zeros((lc_terminal, cfg.n, cfg.horizon))  # ∈R^{lc,n,te}
        self.cus_terminal = np.zeros((lc_terminal, cfg.m, cfg.horizon))  # ∈R^{lc,m,te}
        self.stage_cost_xs = np.zeros((cfg.n, 1, cfg.horizon))  # ∈R^{n,1,Te}
        self.stage_cost_us = np.zeros((cfg.m, 1, cfg.horizon))  # ∈R^{n,1,Te}
        self.stage_cost_xxs = np.zeros((cfg.n, cfg.n, cfg.horizon))  # ∈R^{n,n,Te}
        self.stage_cost_uus = np.zeros((cfg.m, cfg.m, cfg.horizon))  # ∈R^{m,m,Te}
        self.stage_cost_xus = np.zeros((cfg.n, cfg.m, cfg.horizon))  # ∈R^{n,m,Te}
        self.terminal_cost_x = np.zeros((cfg.n,))  # ∈R^{n,}
        self.terminal_cost_xx = np.zeros((cfg.n, cfg.n))  # ∈R^{n,n}

    def set_traj_idx(self, idx: int) -> None:
        self.traj_idx = idx
        self.every_traj.traj_idx = idx

    def set_x0(self, x0: np.ndarray) -> None:
        self.xs[:, :, 0] = x0
        self.every_traj.xs[:, :, 0] = x0[:, : self.n_org]
