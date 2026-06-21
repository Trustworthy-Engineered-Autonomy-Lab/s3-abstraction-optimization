import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import pickle

# Helper to classify grid
def classify_grid(nx, ny, ntheta, theta_index, verified_safe_states, ground_truth_check):
    unsafe_agree = 0
    verified_safe = 1
    unsafe_disagree = 2
    class_grid = np.full((ny, nx), unsafe_agree, dtype=np.int8)
    gt_safe_count = 0
    verified_gt_safe_count = 0

    for i in range(nx):
        for j in range(ny):
            state_id = i * (ny * ntheta) + j * ntheta + theta_index
            gt_safe = ground_truth_check.get(state_id) == "goal"
            if gt_safe:
                gt_safe_count += 1
            if state_id in verified_safe_states:
                class_grid[j, i] = verified_safe
                if gt_safe:
                    verified_gt_safe_count += 1
            elif ground_truth_check.get(state_id) == "goal":
                class_grid[j, i] = unsafe_disagree

    recall = np.nan
    if gt_safe_count > 0:
        recall = verified_gt_safe_count / gt_safe_count

    counts = {
        "verified_safe": int(np.count_nonzero(class_grid == verified_safe)),
        "unsafe_agree": int(np.count_nonzero(class_grid == unsafe_agree)),
        "unsafe_disagree": int(np.count_nonzero(class_grid == unsafe_disagree)),
        "ground_truth_safe": gt_safe_count,
        "verified_ground_truth_safe": verified_gt_safe_count,
        "recall": recall,
    }
    return class_grid, counts

# Load gt regions
def load_ground_truth_regions(cache_path):
    with open(cache_path, "rb") as f:
        return pickle.load(f)

# Visualize a slice(s)
def visualize_slice(
    x_edges,
    y_edges,
    theta_edges,
    theta_index,
    verified_safe_states,
    *,
    ground_truth_check=None,
    title=None,
    xlim=None,
    ylim=None,
    figsize=(7, 7),
    alpha=0.72,
    show_legend=True,
    show_axes_grid=False,
    ax=None,
):
    """Plot verification results on one or more theta layers."""

    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    theta_edges = np.asarray(theta_edges, dtype=float)
    theta_indices = np.atleast_1d(theta_index).astype(int).tolist()
    single_slice = np.isscalar(theta_index)

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    ntheta = len(theta_edges) - 1
    for k in theta_indices:
        if not (0 <= k < ntheta):
            raise ValueError(f"theta_index must be in [0, {ntheta - 1}], got {k}")

    verified_safe_states = set(verified_safe_states)
    ground_truth_check = ground_truth_check or {}

    unsafe_agree = 0
    verified_safe = 1
    unsafe_disagree = 2

    colors = [
        (0.82, 0.82, 0.82, alpha),
        (0.15, 0.62, 0.22, alpha),
        (0.86, 0.12, 0.10, alpha),
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(4) - 0.5, 3)

    if single_slice:
        theta_indices = [theta_indices[0]]

    if ax is not None and len(theta_indices) != 1:
        raise ValueError("Pass ax only when plotting a single theta slice")

    if ax is None and len(theta_indices) == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        axes = [ax]
    elif ax is not None:
        fig = ax.figure
        axes = [ax]
    else:
        nplots = len(theta_indices)
        ncols = int(np.ceil(np.sqrt(nplots)))
        nrows = int(np.ceil(nplots / ncols))
        if figsize == (7, 7):
            figsize = (4.2 * ncols, 4.0 * nrows)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        axes = np.ravel(np.asarray(axs)).tolist()

    counts_by_theta = {}
    for plot_ax, k in zip(axes, theta_indices):
        class_grid, counts = classify_grid(
            nx,
            ny,
            ntheta,
            k,
            verified_safe_states,
            ground_truth_check,
        )
        counts_by_theta[k] = counts

        plot_ax.pcolormesh(
            x_edges,
            y_edges,
            class_grid,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors=(0.0, 0.0, 0.0, 0.32),
            linewidth=0.45,
        )

        plot_ax.set_xlim((float(x_edges[0]), float(x_edges[-1])) if xlim is None else xlim)
        plot_ax.set_ylim((float(y_edges[0]), float(y_edges[-1])) if ylim is None else ylim)
        plot_ax.set_aspect("equal", adjustable="box")
        plot_ax.set_xlabel("x")
        plot_ax.set_ylabel("y")
        plot_ax.grid(show_axes_grid, alpha=0.25)

        theta_lo = theta_edges[k]
        theta_hi = theta_edges[k + 1]
        plot_ax.set_title(f"Theta slice {k}: [{theta_lo:.3f}, {theta_hi:.3f}]")

    for unused_ax in axes[len(theta_indices):]:
        unused_ax.set_visible(False)

    if title is not None:
        if len(theta_indices) == 1:
            axes[0].set_title(title)
        else:
            fig.suptitle(title)

    if show_legend:
        legend_counts = counts_by_theta[theta_indices[0]]
        handles = [
            Patch(facecolor=colors[verified_safe], edgecolor="k", label=f"Verified safe ({legend_counts['verified_safe']})"),
            Patch(facecolor=colors[unsafe_agree], edgecolor="k", label=f"Unsafe, GT agrees ({legend_counts['unsafe_agree']})"),
        ]
        if legend_counts["unsafe_disagree"] > 0:
            handles.append(
                Patch(
                    facecolor=colors[unsafe_disagree],
                    edgecolor="k",
                    label=f"Unsafe, GT disagrees ({legend_counts['unsafe_disagree']})",
                )
            )
        axes[0].legend(handles=handles, loc="upper right", frameon=True)

    if single_slice:
        return fig, axes[0], counts_by_theta[theta_indices[0]]
    return fig, axes[:len(theta_indices)], counts_by_theta


if __name__ == "__main__":
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import pyModelChecking as pmc

    import unicycle_abstraction as ua
    import unicycle_objectives as uo
    import unicycle_optimizers as u_opt
    import verification_tools as vt

    shape = [100, 100, 100]
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])
    theta_indices = [1, shape[2] // 2, shape[2] - 2]

    key = jax.random.PRNGKey(0)
    key, k1, k2, k3 = jax.random.split(key, 4)
    params = jnp.concatenate(
        [
            0.5 * jax.random.normal(k1, (shape[0],)),
            0.5 * jax.random.normal(k2, (shape[1],)),
            0.5 * jax.random.normal(k3, (shape[2],)),
        ]
    )

    print("Training grid...")
    params_opt, _, _ = u_opt.gradient_descent(
        params,
        uo.image_volume_over_parent,
        shape=shape,
        domain_lb=domain_lb,
        domain_ub=domain_ub,
        steps=300,
        lr=1e-4,
        grad_clip=1e3,
        print_every=5,
        record_every=5,
    )
    params_opt.block_until_ready()

    x_edges, y_edges, theta_edges = uo.extract_grid_params(
        params_opt,
        shape,
        domain_lb,
        domain_ub,
    )

    print("Building abstraction...")
    kripke_components = ua.build_abstraction(x_edges, y_edges, theta_edges, verbose=False)
    grid_states = kripke_components["kripke_states"][:-1]
    kripke_structure = pmc.Kripke(
        S=kripke_components["kripke_states"],
        S0=grid_states,
        R=list(kripke_components["kripke_transitions"]),
        L=kripke_components["kripke_labels"],
    )

    print("Running CTL model checking...")
    verified_safe_states = vt.model_check_kripke(
        kripke_structure
    )

    print("Loading cached ground truth for the plotted theta slices...")
    gt_reach_regions = load_ground_truth_regions("unicycle-taylor/unicycle_gt_reach_regions_100.pkl")
    ground_truth_check = vt.check_ground_truth_fast_slice(
        x_edges,
        y_edges,
        theta_edges,
        domain_lb,
        domain_ub,
        gt_reach_regions,
    )

    fig, _, counts = visualize_slice(
        x_edges,
        y_edges,
        theta_edges,
        theta_indices,
        verified_safe_states,
        ground_truth_check=ground_truth_check,
        title="Trained verification slices",
    )
    print(f"Visualization counts: {counts}")
    if ground_truth_check is not None:
        for k, slice_counts in counts.items():
            print(f"Theta slice {k} recall: {slice_counts['recall']:.4f}")
    fig.savefig("visualization_slice.png", dpi=200)
    plt.show()
