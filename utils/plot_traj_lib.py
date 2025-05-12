import matplotlib.pyplot as plt
import numpy as np


def plot_xs(
    xs_org,
    xs_interpolated,
    x_ref,
    x_min,
    x_max,
    steps_per_knot,
    time_step_on_knot,
    horizon,
    interpolation_color="deepskyblue",
) -> None:
    n = xs_org.shape[0]
    knot_frames = [0]
    f = 0
    for steps in steps_per_knot:
        f += steps
        knot_frames.append(f)

    fig = plt.figure(figsize=(15, 3.5 * n))
    for i in range(n):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.plot(xs_interpolated[i, :], color=interpolation_color)
        ax.plot(
            knot_frames,
            xs_interpolated[i, time_step_on_knot],
            "o",
            color=interpolation_color,
        )
        ax.plot(xs_org[i, :], "red", linestyle="dashed")
        if np.isinf(x_min[i]):
            pass
        else:
            ax.plot((horizon + 1) * [x_min[i]], "black")
        if np.isinf(x_max[i]):
            pass
        else:
            ax.plot((horizon + 1) * [x_max[i]], "black")
        ax.plot((horizon + 1) * [x_ref[i]], "blue")
    plt.show()


def plot_us(
    us_org,
    us_interpolated,
    u_min,
    u_max,
    steps_per_knot,
    time_step_on_knot,
    horizon,
    interpolation_color="deepskyblue",
) -> None:
    m = us_org.shape[0]
    knot_frames = [0]
    f = 0
    for steps in steps_per_knot:
        f += steps
        knot_frames.append(f)
    fig = plt.figure(figsize=(15, 3.5 * m))
    for i in range(m):
        ax = fig.add_subplot(m, 1, i + 1)
        ax.plot(us_interpolated[i, :], color=interpolation_color)
        ax.plot(
            knot_frames[:-1],
            us_interpolated[i, time_step_on_knot[:-1]],
            "o",
            color=interpolation_color,
        )
        ax.plot(us_org[i, :], "red", linestyle="dashed")
        if np.isinf(u_min[i]):
            pass
        else:
            ax.plot((horizon + 1) * [u_min[i]], "black")
        if np.isinf(u_max[i]):
            pass
        else:
            ax.plot((horizon + 1) * [u_max[i]], "black")
    plt.show()


def plot_x_traj(
    ax,
    xs,
    i,
    knot_frames,
    config,
    color=None,
    plot_knot=False,
    label=None,
    display_tick=True,
) -> None:
    if plot_knot:
        ax.plot(knot_frames, xs[i, knot_frames], "o", color=color, markersize=3.5)
    # else:
    if label is None:
        ax.plot(xs[i, :], color=color)
    else:
        ax.plot(xs[i, :], color=color, label=label)

    if np.isinf(config.x_min[i]):
        pass
    else:
        ax.plot((config.horizon + 1) * [config.x_min[i]], "black", ls="dotted")
    if np.isinf(config.x_max[i]):
        pass
    else:
        ax.plot((config.horizon + 1) * [config.x_max[i]], "black", ls="dotted")

    if not display_tick:
        ax.tick_params(labelbottom=False)


def plot_u_traj(
    ax,
    us,
    i,
    knot_frames,
    config,
    color=None,
    plot_knot=False,
    label=None,
    display_tick=True,
) -> None:
    if plot_knot:
        ax.plot(
            knot_frames[:-1], us[i, knot_frames[:-1]], "o", color=color, markersize=3.5
        )

    if label is None:
        ax.plot(us[i, :], color=color)
    else:
        ax.plot(us[i, :], color=color, label=label)

    if np.isinf(config.u_min[i]):
        pass
    else:
        ax.plot((config.horizon + 1) * [config.u_min[i]], "black", ls="dotted")
    if np.isinf(config.u_max[i]):
        pass
    else:
        ax.plot((config.horizon + 1) * [config.u_max[i]], "black", ls="dotted")

    if not display_tick:
        ax.tick_params(labelbottom=False)


def plot_cost_vs_time(
    ax,
    comp_time,
    comp_time_zero_order,
    comp_time_linear,
    comp_time_cubic,
    cost_history,
    original_cost_history_zero_order,
    original_cost_history_linear,
    original_cost_history_cubic,
    original_label,
    zi_label,
    li_label,
    ci_label,
    original_color="red",
    zi_color="royalblue",
    li_color="darkgreen",
    ci_color="darkorange",
):
    axins = ax.inset_axes([0.25, 0.6, 0.35, 0.35])
    ax.plot(
        comp_time,
        cost_history,
        original_color,
        label=original_label,
        lw=1.0,
        marker="o",
        markersize=2.0,
    )
    ax.plot(
        comp_time_zero_order,
        original_cost_history_zero_order,
        zi_color,
        label=zi_label,
        lw=1.0,
        marker="o",
        markersize=2.0,
    )
    ax.plot(
        comp_time_linear,
        original_cost_history_linear,
        li_color,
        label=li_label,
        lw=1.0,
        marker="o",
        markersize=2.0,
    )
    ax.plot(
        comp_time_cubic,
        original_cost_history_cubic,
        ci_color,
        label=ci_label,
        lw=1.0,
        marker="o",
        markersize=2.0,
    )
    axins.plot(
        comp_time,
        cost_history,
        original_color,
        lw=1.0,
        marker="o",
        markersize=2.0,
    )
    axins.plot(
        comp_time_zero_order,
        original_cost_history_zero_order,
        zi_color,
        lw=1.0,
        marker="o",
        markersize=2.0,
    )
    axins.plot(
        comp_time_linear,
        original_cost_history_linear,
        li_color,
        lw=1.0,
        marker="o",
        markersize=2.0,
    )
    axins.plot(
        comp_time_cubic,
        original_cost_history_cubic,
        ci_color,
        lw=1.0,
        marker="o",
        markersize=2.0,
    )
    axins.set_xlim(0, 0.2 * comp_time[-1])
    axins.set_ylim(0.5 * min(cost_history), 1.5 * max(cost_history))
    plt.yscale("log")
    plt.ylabel("Original Cost")
    plt.xlabel("Computational Time [sec]")
    plt.grid()
    plt.legend()
    axins.set_yscale("log")
    ax.indicate_inset_zoom(axins)


def plot_state(
    fig,
    cfg,
    xs_opt,
    xs_opt_zero_order,
    xs_opt_linear,
    xs_opt_cubic,
    knot_frames,
    state_name,
    original_label,
    zi_label,
    li_label,
    ci_label,
    original_color="red",
    zi_color="royalblue",
    li_color="darkgreen",
    ci_color="darkorange",
):
    for i in range(cfg.n):
        ax = fig.add_subplot(cfg.n, 1, i + 1)
        if i == (cfg.n - 1):
            display_tick = True
        else:
            display_tick = False
        ax.plot(
            (cfg.horizon + 1) * [cfg.x_ref[i]],
            color="magenta",
            linestyle="dashed",
        )
        plot_x_traj(
            ax,
            xs_opt_cubic,
            i,
            knot_frames,
            cfg,
            color=ci_color,
            label=ci_label,
            plot_knot=True,
            display_tick=display_tick,
        )
        plot_x_traj(
            ax,
            xs_opt_linear,
            i,
            knot_frames,
            cfg,
            color=li_color,
            label=li_label,
            plot_knot=True,
            display_tick=display_tick,
        )
        plot_x_traj(
            ax,
            xs_opt_zero_order,
            i,
            knot_frames,
            cfg,
            color=zi_color,
            label=zi_label,
            plot_knot=True,
            display_tick=display_tick,
        )
        plot_x_traj(
            ax,
            xs_opt,
            i,
            knot_frames,
            cfg,
            color=original_color,
            label=original_label,
            display_tick=display_tick,
        )

        ax.set_ylabel(state_name[i])
        if i == 0:
            # plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=2)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=2,
                handles=handles[::-1],
                labels=labels[::-1],
            )
    fig.align_labels()


def plot_input(
    fig,
    cfg,
    us_opt,
    us_opt_zero_order,
    us_opt_linear,
    us_opt_cubic,
    knot_frames,
    input_name,
    original_label,
    zi_label,
    li_label,
    ci_label,
    original_color="red",
    zi_color="royalblue",
    li_color="darkgreen",
    ci_color="darkorange",
):
    for i in range(cfg.m):
        ax = fig.add_subplot(cfg.m, 1, i + 1)
        if i == (cfg.m - 1):
            display_tick = True
        else:
            display_tick = False

        plot_u_traj(
            ax,
            us_opt_cubic,
            i,
            knot_frames,
            cfg,
            color=ci_color,
            label=ci_label,
            plot_knot=True,
            display_tick=display_tick,
        )
        plot_u_traj(
            ax,
            us_opt_linear,
            i,
            knot_frames,
            cfg,
            color=li_color,
            label=li_label,
            plot_knot=True,
            display_tick=display_tick,
        )
        plot_u_traj(
            ax,
            us_opt_zero_order,
            i,
            knot_frames,
            cfg,
            color=zi_color,
            label=zi_label,
            plot_knot=True,
            display_tick=display_tick,
        )
        plot_u_traj(
            ax,
            us_opt,
            i,
            knot_frames,
            cfg,
            color=original_color,
            label=original_label,
            display_tick=display_tick,
        )

        ax.set_ylabel(input_name[i])
        if i == 0:
            # plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=2)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=2,
                handles=handles[::-1],
                labels=labels[::-1],
            )

    fig.align_labels()
