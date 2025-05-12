from parametric_ddp.interpolator.cubic.Constraint import Constraint as CubicConstraint
from parametric_ddp.interpolator.cubic.CubicInterpolationConfigWrapper import (
    CubicInterpolationConfigWrapper,
)
from parametric_ddp.interpolator.cubic.CubicInterpolationWrapper import (
    CubicInterpolationWrapper,
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
    cfg_cubic = CubicInterpolationConfigWrapper(
        cfg,
        steps_per_knot,
    )
    const_cubic = CubicConstraint(cfg_cubic)
    cost_cubic = InvertedPendulumCost(cfg_cubic)
    traj_info_cubic = TrajInfoWrapper(
        cfg_cubic, const_cubic.lc, const_cubic.lc_terminal, const.lc, cfg.step_size_num
    )
    ocp_cubic = CubicInterpolationWrapper(
        dynamics, const_cubic, const, cost_cubic, cfg_cubic
    )
    calcluator = ConstraintPointCalculator(
        dynamics, const_cubic, const, ocp_cubic, traj_info_cubic, cfg_cubic
    )
    calcluator.calc(knot_step)
