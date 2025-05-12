import typing


class ConstraintPoint:
    # ノット点から何ステップ目でどの制約条件を評価したいか
    def __init__(
        self,
        const_indices: typing.List[int],
        eval_time_step_from_knot: int,
        time_steps_on_knot,
    ) -> None:
        self.const_indices: list[int] = const_indices
        self.const_num: int = len(const_indices)
        self.eval_time_step_from_knot: int = eval_time_step_from_knot
        self.time_steps_on_eval_point: list[int] = []
        for idx, time_step_on_knot in enumerate(time_steps_on_knot[:-1]):
            if (time_step_on_knot + self.eval_time_step_from_knot) < time_steps_on_knot[
                idx + 1
            ]:
                self.time_steps_on_eval_point.append(
                    time_step_on_knot + self.eval_time_step_from_knot
                )
