from parametric_ddp.interpolator.linear.LinearInterpolationConfigWrapper import (
    LinearInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.linear.LinearInterpolationWrapper import (
    LinearInterpolationWrapper,
)
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.lin_system.LinearSystemConfig import LinearSystemConfig
from problem_setting.lin_system.LinearSystemConstraint import LinearSystemConstraint
from problem_setting.lin_system.LinearSystemCost import LinearSystemCost
from problem_setting.lin_system.LinearSystemDynamics import LinearSystemDynamics
from utils.ConstraintPointCalculator import ConstraintPointCalculator

if __name__ == "__main__":
    knot_step = 25
    steps_per_knot = 2 * [knot_step]
    cfg = LinearSystemConfig(horizon=sum(steps_per_knot))
    dynamics = LinearSystemDynamics(cfg)
    const = LinearSystemConstraint(cfg)
    cfg_linear = LinearInterpolationConfigWrapper(
        cfg,
        steps_per_knot,
    )
    const_linear = LinearSystemConstraint(cfg_linear)
    cost_linear = LinearSystemCost(cfg_linear)
    traj_info_linear = TrajInfoWrapper(
        cfg_linear,
        const_linear.lc,
        const_linear.lc_terminal,
        const.lc,
        cfg.step_size_num,
    )
    ocp_linear = LinearInterpolationWrapper(
        dynamics, const_linear, const, cost_linear, cfg_linear
    )
    calcluator = ConstraintPointCalculator(
        dynamics, const_linear, const, ocp_linear, traj_info_linear, cfg_linear
    )
    calcluator.calc(knot_step)
