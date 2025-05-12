import numpy as np

from problem_setting.abstract.Config import Config


class ZeroOrderHolderConfigWrapper(Config):
    def __init__(
        self,
        cfg: Config,
        steps_per_knot,
    ) -> None:
        self.org_cfg: Config = cfg
        self.n: int = cfg.n
        self.m: int = cfg.m
        self.dt: float = cfg.dt
        self.horizon: int = len(steps_per_knot)
        self.x_ini: np.ndarray = cfg.x_ini
        self.x_ref: np.ndarray = cfg.x_ref
        self.Q_ini: np.ndarray = cfg.Q
        self.Q: np.ndarray = cfg.Q
        self.R: np.ndarray = cfg.R
        self.P: np.ndarray = np.zeros((self.n, self.m))
        self.Q_terminal: np.ndarray = cfg.Q_terminal
        self.x_max: np.ndarray = cfg.x_max
        self.x_min: np.ndarray = cfg.x_min
        self.u_max: np.ndarray = cfg.u_max
        self.u_min: np.ndarray = cfg.u_min
        self.idx_x_min: list[int] = cfg.idx_x_min
        self.idx_x_max: list[int] = cfg.idx_x_max
        self.idx_u_min: list[int] = cfg.idx_u_min
        self.idx_u_max: list[int] = cfg.idx_u_max

        self.free_state_idx: list[int] = cfg.free_state_idx + []
        self.terminal_const_state_idx: list[int] = []
        self.tol: float = cfg.tol  # * max(steps_per_knot)
        self.max_iter: int = cfg.max_iter
        self.run_ddp: bool = cfg.run_ddp
        self.automatic_initilization_barrier_param_is_enabled: bool = (
            cfg.automatic_initilization_barrier_param_is_enabled
        )
        self.barrier_param_ini: float = cfg.barrier_param_ini
        self.step_size_num: int = cfg.step_size_num
        self.cost_change_ratio_convergence_criteria = (
            cfg.cost_change_ratio_convergence_criteria
        )
        # for parametric DDP
        self.steps_per_knot = steps_per_knot
        self.is_step_unique: bool = all(
            val == self.steps_per_knot[0] for val in self.steps_per_knot
        )
        self.time_step_on_knot = [0]
        input_update_step = 0
        for step in self.steps_per_knot:
            input_update_step += step
            self.time_step_on_knot.append(input_update_step)

        knot_idx = 0
        time_step_from_knot = 0
        self.knot_idxs = []
        self.time_step_from_knots = []
        for i in range(cfg.horizon):
            if i == self.time_step_on_knot[knot_idx]:
                knot_idx += 1
                time_step_from_knot = 0
            self.knot_idxs.append(knot_idx)
            self.time_step_from_knots.append(time_step_from_knot)
            time_step_from_knot += 1

        self.u_inis = 0.0 * cfg.u_inis[:, :, self.time_step_on_knot[:-1]]
        self.normalize_Qu_error = True
        self.Qu_normalization_factor = 1.0 / self.steps_per_knot[0]
        self.knows_optimal_cost = False
        self.optimal_cost = 0.0
