import numpy as np

from problem_setting.abstract.Config import Config


class LinearSystemConfig(Config):
    def __init__(
        self,
        x_ini: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0]),
        x_ref: np.ndarray = np.array([-3.5, 0.0, 2.0, 0.0]),
        horizon: int = 500,
        dt: float = 0.005,
    ) -> None:
        # Problem Config
        self.n: int = 4  # state simension
        self.m: int = 2  # control input dimension
        self.x_ini: np.ndarray = x_ini
        self.x_ref: np.ndarray = x_ref
        self.dt: float = dt
        self.horizon: int = horizon
        self.Q: np.ndarray = np.diag(np.array([0.01, 0.01, 0.01, 0.01]))
        self.R: np.ndarray = np.diag(np.array([0.001, 0.001]))
        self.Q_terminal: np.ndarray = np.diag(
            np.array(
                [
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                ]
            )
        )
        self.Q_ini = self.Q
        self.x_max = np.array([[np.inf], [np.inf], [np.inf], [np.inf]])
        self.x_min = np.array([[-np.inf], [-np.inf], [-np.inf], [-np.inf]])
        self.u_max = np.array([[2.0], [2.0]])
        self.u_min = np.array([[-2.0], [-2.0]])
        self.idx_x_min = []
        self.idx_x_max = []
        self.idx_u_min = [0, 1]
        self.idx_u_max = [0, 1]

        self.free_state_idx = []
        self.terminal_const_state_idx = []
        self.tol = 1e-6
        self.max_iter: int = 1000
        self.run_ddp: bool = False
        self.automatic_initilization_barrier_param_is_enabled: bool = True
        self.barrier_param_ini: float = 0.001
        self.step_size_num: int = 21
        self.cost_change_ratio_convergence_criteria = 0.999
        self.u_inis = np.zeros((self.step_size_num, self.m, self.horizon))
        self.knows_optimal_cost = False
        self.optimal_cost = 0.0

        self.normalize_Qu_error: bool = False
