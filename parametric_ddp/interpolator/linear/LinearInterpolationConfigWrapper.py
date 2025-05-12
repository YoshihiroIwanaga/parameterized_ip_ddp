import numpy as np

from problem_setting.abstract.Config import Config


class LinearInterpolationConfigWrapper(Config):
    def __init__(
        self,
        cfg: Config,
        steps_per_knot,
    ) -> None:
        self.org_cfg: Config = cfg
        self.n: int = cfg.n + cfg.m
        self.m: int = cfg.m
        self.dt: float = cfg.dt
        self.horizon = len(steps_per_knot)

        self.x_ini: np.ndarray = np.hstack(
            (cfg.x_ini, ((cfg.u_max + cfg.u_min) / 2).reshape(cfg.m))
        )
        self.x_ref: np.ndarray = np.hstack(
            (cfg.x_ref, ((cfg.u_max + cfg.u_min) / 2).reshape(cfg.m))
        )
        self.Q_ini = np.diag(np.hstack((np.diag(cfg.Q), 0.5 * np.diag(cfg.R))))
        self.Q: np.ndarray = np.diag(np.hstack((np.diag(cfg.Q), np.diag(cfg.R))))
        self.R: np.ndarray = 0.05 * cfg.R
        # self.R: np.ndarray = 0.00001 * cfg.R
        self.Q_terminal: np.ndarray = np.diag(
            np.hstack(
                (np.diag(cfg.Q_terminal), 0.5 * steps_per_knot[0] * np.diag(cfg.R))
            )
        )
        self.x_max = np.vstack((cfg.x_max, cfg.u_max))
        self.x_min = np.vstack((cfg.x_min, cfg.u_min))
        self.u_max = np.inf * np.ones((self.m, 1))
        self.u_min = -np.inf * np.ones((self.m, 1))
        self.idx_x_min = cfg.idx_x_min + list(range(cfg.n, self.n))
        self.idx_x_max = cfg.idx_x_max + list(range(cfg.n, self.n))
        self.idx_u_min = []
        self.idx_u_max = []

        # for DDP
        self.free_state_idx = cfg.free_state_idx + list(range(cfg.n, cfg.n + cfg.m))
        self.terminal_const_state_idx = cfg.terminal_const_state_idx + list(
            range(cfg.n, cfg.n + cfg.m)
        )
        self.tol = cfg.tol  # * max(steps_per_knot)
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

        self.u_inis = np.zeros((self.step_size_num, self.m, self.horizon))
        for i in range(self.m):
            if np.isinf(self.u_max[i]) or np.isinf(self.u_min[i]):
                self.u_inis[:, i : i + 1, :] = (
                    0.4 * np.random.rand(self.step_size_num, 1, self.horizon) - 0.2
                )
            else:
                self.u_inis[:, i : i + 1, :] = 0.5 * (
                    (self.u_max[i] - self.u_min[i])
                    * np.random.rand(self.step_size_num, 1, self.horizon)
                    + self.u_min[i]
                )
        self.u_inis[0, :, :] = np.zeros((self.m, self.horizon))
        self.normalize_Qu_error = True
        self.Qu_normalization_factor = 1.0 / self.steps_per_knot[0]
        self.knows_optimal_cost = False
        self.optimal_cost = 0.0
