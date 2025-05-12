import numpy as np
from parametric_ddp.utils.TrajInfo import TrajInfo
from problem_setting.abstract.Config import Config


class BackwardPassResult:
    def __init__(self, n, m, lc, lc_terminal, horizon) -> None:
        self.ku: np.ndarray = np.zeros((m, horizon))
        self.Ku: np.ndarray = np.zeros((m, n, horizon))
        self.kdual: np.ndarray = np.zeros((lc, horizon))
        self.Kdual: np.ndarray = np.zeros((lc, n, horizon))
        self.kdual_terminal: np.ndarray = np.zeros((lc_terminal, 1))
        self.Kdual_terminal: np.ndarray = np.zeros((lc_terminal, n, 1))
        self.kx0: np.ndarray = np.zeros((n,))
        self.Qu_error: float = 0.0
        self.Qx0_error: float = 0.0
        self.const_error: float = 0.0
        self.barrier_param_error: float = 0.0
        self.optimal_error: float = 0.0


class MatrixSet:
    def __init__(self) -> None:
        self.value_function_x: np.ndarray = np.zeros((0, 0))
        self.value_function_xx: np.ndarray = np.zeros((0, 0))
        self.Qx: np.ndarray = np.zeros((0, 0))
        self.Qu: np.ndarray = np.zeros((0, 0))
        self.Qxx: np.ndarray = np.zeros((0, 0))
        self.Qxu: np.ndarray = np.zeros((0, 0))
        self.Quu: np.ndarray = np.zeros((0, 0))
        self.Qdualx: np.ndarray = np.zeros((0, 0))
        self.Qdualu: np.ndarray = np.zeros((0, 0))
        self.Quu_reg: np.ndarray = np.zeros((0, 0))
        self.dual: np.ndarray = np.zeros((0, 0))
        self.cs: np.ndarray = np.zeros((0, 0))
        self.c_inv: np.ndarray = np.zeros((0, 0))
        self.S: np.ndarray = np.zeros((0, 0))
        self.r: np.ndarray = np.zeros((0, 0))
        self.SC_inv: np.ndarray = np.zeros((0, 0))


class BackwardPass:
    def __init__(self, cfg: Config, lc: int, lc_terminal: int) -> None:
        self.horizon: int = cfg.horizon
        self.n: int = cfg.n  # state diemnsion
        self.m: int = cfg.m  # input dimension
        self.lc: int = lc  # number fo constarints
        self.lc_terminal: int = lc_terminal
        self.free_state_idx = cfg.free_state_idx
        self.reg: float = 0.0
        self.reg_max: float = 24.0
        self.b0: np.ndarray = np.zeros((self.m, 1 + self.n))
        self.failed: bool = False
        self.normalize_Qu_error: bool = cfg.normalize_Qu_error
        if self.normalize_Qu_error:
            self.normalize_factor: float = cfg.Qu_normalization_factor
        else:
            self.normalize_factor: float = 1.0
        self.Qu_error_pre: float = 0.0
        self.const_error_pre: float = 0.0
        self.barrier_param_error_pre: float = 0.0
        self.Qx0_error_pre: float = 0.0

    def reset_reg(self) -> None:
        self.reg = 0.0
        self.failed = False

    def _update_reg(self, forward_pass_failed: bool, step_idx: int) -> None:
        if forward_pass_failed or self.failed:
            self.reg += 4
            self.failed = False
        elif step_idx == 0:
            self.reg -= 1
        elif step_idx <= 4:
            self.reg = self.reg
        else:
            self.reg += 1

        if self.reg < 0:
            self.reg = 0
        elif self.reg > self.reg_max:
            self.reg = self.reg_max

    def _diff_Q(self, traj: TrajInfo, ms: MatrixSet, i: int, use_hessian: bool) -> None:
        if self.lc_terminal > 0 and i == self.horizon - 1:
            ms.Qdualx = np.vstack(
                (
                    traj.cxs[:, :, i],
                    traj.cxs_terminal[:, :, 0] @ traj.fxs[:, :, i],
                )
            )
            ms.Qdualu = np.vstack(
                (
                    traj.cus[:, :, i],
                    traj.cxs_terminal[:, :, 0] @ traj.fus[:, :, i],
                )
            )
        else:
            ms.Qdualx = traj.cxs[:, :, i]
            ms.Qdualu = traj.cus[:, :, i]
        ms.Qx = (
            traj.stage_cost_xs[:, :, i]
            + ms.Qdualx.T @ ms.dual
            + traj.fxs[:, :, i].T @ ms.value_function_x
        )
        ms.Qu = (
            traj.stage_cost_us[:, :, i]
            + ms.Qdualu.T @ ms.dual
            + traj.fus[:, :, i].T @ ms.value_function_x
        )
        quu = traj.stage_cost_uus[:, :, i]
        if use_hessian:
            Vx_reshaped = ms.value_function_x[:, :, np.newaxis]

            ms.Qxx = (
                traj.stage_cost_xxs[:, :, i]
                + traj.fxs[:, :, i].T @ ms.value_function_xx @ traj.fxs[:, :, i]
                + np.sum(Vx_reshaped * traj.fxxs[:, :, :, i], axis=0)
                # + np.einsum(
                #     "g,gij->ij",
                #     ms.value_function_x.reshape(self.n),  # type:ignore
                #     traj.fxxs[:, :, :, i],
                # )
            )
            ms.Qxu = (
                traj.stage_cost_xus[:, :, i]
                + traj.fxs[:, :, i].T @ ms.value_function_xx @ traj.fus[:, :, i]
                + np.sum(Vx_reshaped * traj.fxus[:, :, :, i], axis=0)
                # + np.einsum(
                #     "g,gij->ij",
                #     ms.value_function_x.reshape(self.n),  # type:ignore
                #     traj.fxus[:, :, :, i],
                # )
            )
            ms.Quu = (
                quu
                + traj.fus[:, :, i].T @ ms.value_function_xx @ traj.fus[:, :, i]
                + np.sum(Vx_reshaped * traj.fuus[:, :, :, i], axis=0)
                # + np.einsum(
                #     "g,gij->ij",
                #     ms.value_function_x.reshape(self.n),  # type:ignore
                #     traj.fuus[:, :, :, i],
                # )
            )
        else:
            ms.Qxx = (
                traj.stage_cost_xxs[:, :, i]
                + traj.fxs[:, :, i].T @ ms.value_function_xx @ traj.fxs[:, :, i]
            )
            ms.Qxu = (
                traj.stage_cost_xus[:, :, i]
                + traj.fxs[:, :, i].T @ ms.value_function_xx @ traj.fus[:, :, i]
            )
            ms.Quu = (
                quu + traj.fus[:, :, i].T @ ms.value_function_xx @ traj.fus[:, :, i]
            )

        ms.Quu_reg = ms.Quu + quu * (1.6**self.reg - 1)

    def _prepare_matricies(
        self, ms: MatrixSet, traj: TrajInfo, i: int, barrier_param: float
    ) -> None:
        if self.lc_terminal > 0 and i == self.horizon - 1:
            ms.dual = np.vstack(
                (
                    traj.duals[traj.traj_idx, :, i : i + 1],
                    traj.dual_terminal[traj.traj_idx, :, :],
                )
            )
            lc = self.lc + self.lc_terminal
            ms.cs = np.vstack(
                (
                    traj.cs[traj.traj_idx, :, i : i + 1],
                    traj.cs_terminal[traj.traj_idx, :, 0:1],
                )
            )
        else:
            ms.dual = traj.duals[traj.traj_idx, :, i : i + 1]
            lc = self.lc
            ms.cs = traj.cs[traj.traj_idx, :, i : i + 1]

        ms.S = np.diag(ms.dual.reshape(lc))
        ms.r = ms.S @ ms.cs + barrier_param
        ms.c_inv = 1 / ms.cs
        ms.SC_inv = np.diag((ms.dual * ms.c_inv).reshape(lc))

    def _calc_feedforward_and_feedback(
        self, result: BackwardPassResult, ms: MatrixSet, i: int
    ) -> None:
        if i == 0 and len(self.free_state_idx) > 0:
            Qxfree = ms.Qx[self.free_state_idx, :]  # type:ignore
            Qxfreexfree = ms.Qxx[
                np.ix_(self.free_state_idx, self.free_state_idx)  # type:ignore
            ]
            Qxfreeu = ms.Qxu[self.free_state_idx, :]
            Qsxfree = ms.Qdualx[:, self.free_state_idx]
            A_ = np.vstack(
                (
                    np.hstack((ms.Quu_reg, Qxfreeu.T, ms.Qdualu.T)),
                    np.hstack((Qxfreeu, Qxfreexfree, Qsxfree.T)),
                    np.hstack(
                        (
                            ms.S @ ms.Qdualu,
                            ms.S @ Qsxfree,
                            np.diag(ms.cs.reshape(self.lc)),
                        )
                    ),
                )
            )
            b_ = -np.vstack((ms.Qu, Qxfree, ms.r))
            try:
                x_ = np.linalg.solve(A_, b_)
            except np.linalg.LinAlgError:
                self.failed = True
                return
            result.ku[:, i : i + 1] = x_[: self.m, :]
            result.kx0[self.free_state_idx,] = x_[
                self.m : (self.m + len(self.free_state_idx)),
                0,
            ]
            result.kdual[:, i : i + 1] = x_[(self.m + len(self.free_state_idx)) :, :]
        else:
            try:
                R = np.linalg.cholesky(
                    ms.Quu_reg - ms.Qdualu.T @ (ms.SC_inv) @ (ms.Qdualu)
                )
            except np.linalg.LinAlgError:
                self.failed = True
                return

            self.b0[:, 0:1] = ms.Qu - ms.Qdualu.T @ ((ms.c_inv * ms.r))
            self.b0[:, 1:] = ms.Qxu.T - ms.Qdualu.T @ ms.SC_inv @ ms.Qdualx
            try:
                tmp = np.linalg.solve(
                    R.T,
                    self.b0,
                )
                kK = np.linalg.solve(-R, tmp)
                if np.max(np.abs(kK)) > 1e8:
                    self.failed = True
                    return
            except np.linalg.LinAlgError:
                self.failed = True
                return
            result.ku[:, i : i + 1] = kK[:, 0:1]
            result.Ku[:, :, i] = kK[:, 1:]
            if self.lc_terminal > 0 and i == self.horizon - 1:
                k_ = -ms.c_inv * (ms.r + ms.S @ ms.Qdualu @ result.ku[:, i : i + 1])
                K_ = -ms.SC_inv @ (ms.Qdualx + ms.Qdualu @ result.Ku[:, :, i])
                result.kdual[:, i : i + 1] = k_[: self.lc, :]
                result.Kdual[:, :, i] = K_[: self.lc, :]
                result.kdual_terminal = k_[self.lc :, :]
                result.Kdual_terminal = K_[self.lc :, :]
            else:
                result.kdual[:, i : i + 1] = -ms.c_inv * (
                    ms.r + ms.S @ ms.Qdualu @ result.ku[:, i : i + 1]
                )
                result.Kdual[:, :, i] = -ms.SC_inv @ (
                    ms.Qdualx + ms.Qdualu @ result.Ku[:, :, i]
                )

    def _update_V_diff(self, result: BackwardPassResult, ms: MatrixSet, i: int) -> None:
        ms.Quu = ms.Quu - ms.Qdualu.T @ ms.SC_inv @ ms.Qdualu
        ms.Qxu = ms.Qxu - ms.Qdualx.T @ ms.SC_inv @ ms.Qdualu
        ms.Qxx = ms.Qxx - ms.Qdualx.T @ ms.SC_inv @ ms.Qdualx
        ms.Qu = ms.Qu - ms.Qdualu.T @ (ms.c_inv * ms.r)
        ms.Qx = ms.Qx - ms.Qdualx.T @ (ms.c_inv * ms.r)
        ms.value_function_x = (
            ms.Qx
            + result.Ku[:, :, i].T @ ms.Qu
            + result.Ku[:, :, i].T @ ms.Quu @ result.ku[:, i : i + 1]
            + ms.Qxu @ result.ku[:, i : i + 1]
        )
        ms.value_function_xx = (
            ms.Qxx
            + result.Ku[:, :, i].T @ ms.Qxu.T
            + ms.Qxu @ result.Ku[:, :, i]
            + result.Ku[:, :, i].T @ ms.Quu @ result.Ku[:, :, i]
        )
        ms.value_function_xx = 1 / 2 * (ms.value_function_xx + ms.value_function_xx.T)

    def _update_errors(self, result: BackwardPassResult, ms: MatrixSet, i: int) -> None:
        result.Qu_error = max(
            [result.Qu_error, self.normalize_factor * np.max(np.abs(ms.Qu))]
        )
        # error for complementaly condition
        result.barrier_param_error = max(
            [result.barrier_param_error, np.max(np.abs(ms.r))]
        )
        if i == 0 and len(self.free_state_idx) > 0:
            result.Qx0_error = self.normalize_factor * np.max(
                np.abs(ms.Qx[self.free_state_idx, :])
            )

    def update(
        self,
        traj: TrajInfo,
        forward_pass_failed: bool,
        barrier_param: float,
        use_hessian: bool,
    ) -> BackwardPassResult:
        result = BackwardPassResult(
            self.n, self.m, self.lc, self.lc_terminal, self.horizon
        )
        matrix_set = MatrixSet()
        # ∂V/∂x = ∂q_e/ ∂x
        matrix_set.value_function_x = traj.terminal_cost_x.reshape((self.n, 1))
        # ∂^2V/∂x^2 = ∂^2q_e/ ∂x^2
        matrix_set.value_function_xx = traj.terminal_cost_xx
        self._update_reg(forward_pass_failed, traj.traj_idx)

        for i in reversed(range(0, self.horizon)):
            self._prepare_matricies(matrix_set, traj, i, barrier_param)
            self._diff_Q(traj, matrix_set, i, use_hessian)
            self._calc_feedforward_and_feedback(result, matrix_set, i)
            if self.failed:
                result.Qu_error = self.Qu_error_pre
                result.barrier_param_error = self.barrier_param_error_pre
                result.Qx0_error = self.Qx0_error_pre
                return result
            self._update_errors(result, matrix_set, i)
            if i != 0:
                self._update_V_diff(result, matrix_set, i)

        result.optimal_error = max(
            [
                result.Qu_error,
                result.const_error,
                result.barrier_param_error,
                result.Qx0_error,
            ]
        )
        self.Qu_error_pre = result.Qu_error
        self.const_error_pre = result.const_error
        self.barrier_param_error_pre = result.barrier_param_error
        self.Qx0_error_pre = result.Qx0_error
        return result
