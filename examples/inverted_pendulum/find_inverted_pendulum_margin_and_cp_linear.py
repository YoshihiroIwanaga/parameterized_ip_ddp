from parametric_ddp.interpolator.linear.LinearInterpolationConfigWrapper import (
    LinearInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.linear.LinearInterpolationWrapper import (
    LinearInterpolationWrapper,
)
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.inverted_pendulum.InvertedPendulumConfig import (
    InvertedPendulumConfig,
)
from problem_setting.inverted_pendulum.InvertedPendulumConstraint import (
    InvertedPendulumConstraint,
)
from problem_setting.inverted_pendulum.InvertedPendulumCost import InvertedPendulumCost
from problem_setting.inverted_pendulum.InvertedPendulumDynamics import (
    InvertedPendulumDynamics,
)
from utils.ConstraintPointCalculator import ConstraintPointCalculator

if __name__ == "__main__":
    knot_step = 25
    steps_per_knot = 2 * [knot_step]
    cfg = InvertedPendulumConfig(horizon=sum(steps_per_knot))
    dynamics = InvertedPendulumDynamics(cfg)
    const = InvertedPendulumConstraint(cfg)
    cfg_linear = LinearInterpolationConfigWrapper(
        cfg,
        steps_per_knot,
    )
    const_linear = InvertedPendulumConstraint(cfg_linear)
    cost_linear = InvertedPendulumCost(cfg_linear)
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
