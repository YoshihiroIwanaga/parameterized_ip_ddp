from parametric_ddp.interpolator.zero_order.ZeroOrderHolderConfigWrapper import (
    ZeroOrderHolderConfigWrapper,
)
from parametric_ddp.interpolator.zero_order.ZeroOrderHolderWrapper import (
    ZeroOrderHolderWrapper,
)
from parametric_ddp.utils.TrajInfoWrapper import TrajInfoWrapper
from problem_setting.car.CarConfig import CarConfig
from problem_setting.car.CarConstraint import CarConstraint
from problem_setting.car.CarCost import CarCost
from problem_setting.car.CarDynamics import CarDynamics
from utils.ConstraintPointCalculator import ConstraintPointCalculator

if __name__ == "__main__":
    knot_step = 20
    steps_per_knot = 2 * [knot_step]
    cfg = CarConfig(horizon=sum(steps_per_knot))
    dynamics = CarDynamics(cfg)
    const = CarConstraint(cfg)
    cfg_zero_order = ZeroOrderHolderConfigWrapper(
        cfg,
        steps_per_knot,
    )
    const_zero_order = CarConstraint(cfg_zero_order)
    traj_info_zero_order = TrajInfoWrapper(
        cfg_zero_order, const.lc, const.lc_terminal, const.lc, cfg.step_size_num
    )
    cost_zero_order = CarCost(cfg_zero_order)
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
