import numpy as np

margin_zero_order = np.array(
    [
        0.0,
        0.0,
    ]
).reshape(1, 2, 1)
extra_const_indices_zero_order = [[]]
extra_const_eval_time_steps_from_knot_zero_order = []
extra_const_num_zero_order = 0
for indices in extra_const_indices_zero_order:
    extra_const_num_zero_order += len(indices)

margin_linear = np.array(
    [
        0.0,
        0.0,
    ]
).reshape(1, 2, 1)
extra_const_indices_linear = [[]]
extra_const_eval_time_steps_from_knot_linear = []
extra_const_num_linear = 0
for indices in extra_const_indices_linear:
    extra_const_num_linear += len(indices)


margin_cubic = np.array(
    [
        0.0,
        0.0,
    ]
).reshape(1, 2, 1)
extra_const_indices_cubic = [[]]
extra_const_eval_time_steps_from_knot_cubic = []
extra_const_num_cubic = 0
for indices in extra_const_indices_cubic:
    extra_const_num_cubic += len(indices)
