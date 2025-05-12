from parametric_ddp.interpolator.zero_order.ZeroOrderHolderConfigWrapper import (
    ZeroOrderHolderConfigWrapper,
)
from parametric_ddp.interpolator.zero_order.ZeroOrderHolderWrapper import (
    ZeroOrderHolderWrapper,
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
    cfg_zero_order = ZeroOrderHolderConfigWrapper(
        cfg,
        steps_per_knot,
    )
    const_zero_order = InvertedPendulumConstraint(cfg_zero_order)
    traj_info_zero_order = TrajInfoWrapper(
        cfg_zero_order, const.lc, const.lc_terminal, const.lc, cfg.step_size_num
    )
    cost_zero_order = InvertedPendulumCost(cfg_zero_order)
    ocp_zero_order = ZeroOrderHolderWrapper(
        dynamics, const, cost_zero_order, cfg_zero_order
    )
    calcluator = ConstraintPointCalculator(
        dynamics,
        const_zero_order,
        const,
        ocp_zero_order,
        traj_info_zero_order,
        cfg_zero_order,
    )
    calcluator.calc(knot_step)
