import numpy as np


class Config(object):
    def __init__(self) -> None:
        self.n: int = 0  # state dimension
        self.m: int = 0  # control input dimension
        self.dt: float = 0.01
        self.horizon: int = 100  # prediction horizon
        self.x_ini: np.ndarray = np.zeros((self.n,))  # initial state
        self.x_ref: np.ndarray = np.zeros((self.n,))  # target state
        # weight matrices of cost function
        self.Q_ini: np.ndarray = np.diag(np.zeros((self.n)))
        self.Q: np.ndarray = np.diag(np.zeros((self.n)))
        self.R: np.ndarray = np.diag(np.zeros((self.m)))
        self.Q_terminal: np.ndarray = np.diag(np.zeros((self.n)))
        # upper and lower limits of state and control input
        self.x_max: np.ndarray = np.zeros((self.n,))
        self.x_min: np.ndarray = np.zeros((self.n,))
        self.u_max: np.ndarray = np.zeros((self.m,))
        self.u_min: np.ndarray = np.zeros((self.m,))
        # state indecis which must consider its uppar or lower limit
        self.idx_x_min: list[int] = []
        self.idx_x_max: list[int] = []
        # input indecis which must consider its uppar or lower limit
        self.idx_u_min: list[int] = []
        self.idx_u_max: list[int] = []
        self.tol: float = 1e-7
        self.max_iter: int = 100
        self.min_iter: int = 20
        self.run_ddp: bool = True
        self.automatic_initilization_barrier_param_is_enabled: bool = True
        self.barrier_param_ini: float = 0.001
        self.step_size_num: int = 21
        self.cost_change_ratio_convergence_criteria: float = 0.98
        self.u_ini: np.ndarray = np.zeros((self.m, self.horizon))
        self.u_inis: np.ndarray = np.zeros((self.step_size_num, self.m, self.horizon))
        self.free_state_idx: list[int] = []
        self.terminal_const_state_idx: list[int] = []
        self.normalize_Qu_error: bool = False
        self.Qu_normalization_factor: float = 1.0
        self.knows_optimal_cost: bool = False
        self.optimal_cost: float = 0.0
