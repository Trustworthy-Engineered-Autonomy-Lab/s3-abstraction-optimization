# Libraries
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle as pkl



def dynamics(state):
    A = np.array([[0.8, -0.3],
                   [0.3,  0.8]])
    x_star = np.array([5.0, 5.0])
    state = np.asarray(state)
    return (state - x_star) @ A.T + x_star


def build_abstraction(
        shape,
        domain_lb,
        domain_ub,
        *,
        x_edges = None,
        y_edges = None
        ):
    
    nstates_1, nstates_2 = shape

    if x_edges is None or y_edges is None:
        x_edges = np.linspace(domain_lb[0], domain_ub[0], nstates_1 + 1)
        y_edges = np.linspace(domain_lb[1], domain_ub[1], nstates_2 + 1)
    
    transition_system = {}
    for i in range(nstates_1):
        for j in range(nstates_2):
            xlb, xub, ylb, yub = cell_bounds_from_index(
                x_edges,
                y_edges,
                (i, j)
            )

            verts = np.array([[xlb, ylb],
                              [xlb, yub],
                              [xub, ylb],
                              [xub, yub]])
            
            next_verts = np.array([dynamics(vert) for vert in verts])
            next_aabb_verts = enclosing_aabb_verts(next_verts)
            next_cells = intersecting_cells_from_aabb(x_edges, y_edges, next_aabb_verts)

            transition_system[(i, j)] = next_cells
    
    return transition_system


def cell_bounds_from_index(
        x_edges,
        y_edges,
        cell_index
        ):
    i, j = cell_index
    xlb = x_edges[i]
    xub = x_edges[i + 1]
    ylb = y_edges[j]
    yub = y_edges[j + 1]
    return xlb, xub, ylb, yub


def enclosing_aabb_verts(points):
    points = np.asarray(points)
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    return np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax],
    ])


def random_cell_edges(lower_bound, upper_bound, num_cells, rng=None):
    rng = np.random.default_rng(rng)
    interior_edges = np.sort(rng.uniform(lower_bound, upper_bound, size=num_cells - 1))
    return np.concatenate(([lower_bound], interior_edges, [upper_bound]))


def shortest_distance_from_state_to_cell(
        x_edges,
        y_edges,
        state,
        cell_index
        ):
    xlb, xub, ylb, yub = cell_bounds_from_index(x_edges, y_edges, cell_index)
    x, y = np.asarray(state)

    dx = max(xlb - x, 0.0, x - xub)
    dy = max(ylb - y, 0.0, y - yub)
    return np.hypot(dx, dy)


def sample_states_within_delta_of_cell(
        x_edges,
        y_edges,
        cell_index,
        delta,
        num_samples,
        rng=None
        ):
    xlb, xub, ylb, yub = cell_bounds_from_index(x_edges, y_edges, cell_index)
    width = xub - xlb
    height = yub - ylb
    rng = np.random.default_rng(rng)

    if delta < 0:
        raise ValueError("delta must be nonnegative")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    if num_samples == 0:
        return np.empty((0, 2))

    region_areas = np.array([
        width * height,
        delta * height,
        delta * height,
        width * delta,
        width * delta,
        0.25 * np.pi * delta ** 2,
        0.25 * np.pi * delta ** 2,
        0.25 * np.pi * delta ** 2,
        0.25 * np.pi * delta ** 2,
    ])
    region_counts = rng.multinomial(num_samples, region_areas / np.sum(region_areas))
    samples = []

    if region_counts[0] > 0:
        xs = rng.uniform(xlb, xub, size=region_counts[0])
        ys = rng.uniform(ylb, yub, size=region_counts[0])
        samples.append(np.column_stack((xs, ys)))

    if region_counts[1] > 0:
        xs = rng.uniform(xlb - delta, xlb, size=region_counts[1])
        ys = rng.uniform(ylb, yub, size=region_counts[1])
        samples.append(np.column_stack((xs, ys)))

    if region_counts[2] > 0:
        xs = rng.uniform(xub, xub + delta, size=region_counts[2])
        ys = rng.uniform(ylb, yub, size=region_counts[2])
        samples.append(np.column_stack((xs, ys)))

    if region_counts[3] > 0:
        xs = rng.uniform(xlb, xub, size=region_counts[3])
        ys = rng.uniform(yub, yub + delta, size=region_counts[3])
        samples.append(np.column_stack((xs, ys)))

    if region_counts[4] > 0:
        xs = rng.uniform(xlb, xub, size=region_counts[4])
        ys = rng.uniform(ylb - delta, ylb, size=region_counts[4])
        samples.append(np.column_stack((xs, ys)))

    corner_specs = [
        (xlb, ylb, np.pi, 1.5 * np.pi),
        (xub, ylb, 1.5 * np.pi, 2.0 * np.pi),
        (xub, yub, 0.0, 0.5 * np.pi),
        (xlb, yub, 0.5 * np.pi, np.pi),
    ]
    for count, (cx, cy, theta_min, theta_max) in zip(region_counts[5:], corner_specs):
        if count == 0:
            continue
        radii = delta * np.sqrt(rng.uniform(size=count))
        angles = rng.uniform(theta_min, theta_max, size=count)
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)
        samples.append(np.column_stack((xs, ys)))

    all_samples = np.vstack(samples)
    rng.shuffle(all_samples)
    return np.array(all_samples)



def intersecting_cells_from_aabb(
        x_edges,
        y_edges,
        aabb_verts
        ):
    aabb_verts = np.asarray(aabb_verts)

    xmin = np.min(aabb_verts[:, 0])
    xmax = np.max(aabb_verts[:, 0])
    ymin = np.min(aabb_verts[:, 1])
    ymax = np.max(aabb_verts[:, 1])

    i_start = np.searchsorted(x_edges, xmin, side="left") - 1
    i_end = np.searchsorted(x_edges, xmax, side="right") - 1
    j_start = np.searchsorted(y_edges, ymin, side="left") - 1
    j_end = np.searchsorted(y_edges, ymax, side="right") - 1

    i_start = max(i_start, 0)
    j_start = max(j_start, 0)
    i_end = min(i_end, len(x_edges) - 2)
    j_end = min(j_end, len(y_edges) - 2)

    if i_start > i_end or j_start > j_end:
        return []

    return [(i, j)
            for i in range(i_start, i_end + 1)
            for j in range(j_start, j_end + 1)]


def max_sampled_distance_to_successors(
        x_edges,
        y_edges,
        cell,
        transition_system,
        delta=0.0,
        num_samples=1000,
        rng=None
        ):
    successor_cells = transition_system[cell]
    state_samples = sample_states_within_delta_of_cell(
        x_edges,
        y_edges,
        cell,
        delta,
        num_samples,
        rng=rng
    )
    next_state_samples = np.array([dynamics(state) for state in state_samples])

    return max((shortest_distance_from_state_to_cell(
                    x_edges,
                    y_edges,
                    state,
                    successor
                )
                for successor in successor_cells
                for state in next_state_samples),
               default=np.nan)


def compute_min_delta(
        shape,
        domain_lb,
        domain_ub,
        x_edges,
        y_edges,
        delta_iterations=500,
        num_samples=500
        ):
    nstates_1, nstates_2 = shape
    transition_system = build_abstraction(
        shape,
        domain_lb,
        domain_ub,
        x_edges=x_edges,
        y_edges=y_edges
    )

    min_delta = 0.0
    for i in range(nstates_1):
        for j in range(nstates_2):
            cell = (i, j)
            delta = 0.0
            for _ in range(delta_iterations):
                delta_sat = max_sampled_distance_to_successors(
                    x_edges,
                    y_edges,
                    cell,
                    transition_system,
                    delta,
                    num_samples=num_samples
                )
                if np.abs(delta - delta_sat) <= 1e-1:
                    delta = max(delta, delta_sat)
                    break
                delta = max(delta, delta_sat)

            if delta > min_delta:
                min_delta = delta

    return min_delta



if __name__ == "__main__":

    # Previous single-run experiment kept for reference.
    # domain_lb = [0, 0]
    # domain_ub = [10, 10]
    # shape = [50, 50]
    #
    # nstates_1, nstates_2 = shape
    # rng = np.random.default_rng()
    # x_edges = random_cell_edges(domain_lb[0], domain_ub[0], nstates_1, rng=rng)
    # y_edges = random_cell_edges(domain_lb[1], domain_ub[1], nstates_2, rng=rng)
    #
    # min_delta = compute_min_delta(
    #     shape,
    #     domain_lb,
    #     domain_ub,
    #     x_edges,
    #     y_edges,
    #     delta_iterations=500,
    #     num_samples=500
    # )
    # print(f"Smallest possible satisficing delta = {min_delta:.2f}")

    domain_lb = [0, 0]
    domain_ub = [10, 10]
    shape = [50, 50]
    num_trials = 10

    nstates_1, nstates_2 = shape
    trial_ids = np.arange(1, num_trials + 1)
    min_deltas = []
    rng = np.random.default_rng()

    for trial in trial_ids:
        x_edges = random_cell_edges(domain_lb[0], domain_ub[0], nstates_1, rng=rng)
        y_edges = random_cell_edges(domain_lb[1], domain_ub[1], nstates_2, rng=rng)

        min_delta = compute_min_delta(
            shape,
            domain_lb,
            domain_ub,
            x_edges,
            y_edges,
            delta_iterations=1000,
            num_samples=500
        )
        min_deltas.append(min_delta)
        print(f"trial={trial}, shape={shape}, min_delta={min_delta:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(trial_ids, min_deltas, marker="o")
    plt.xlabel("Radnom trial #")
    plt.ylabel("$\hat{d}^{\leftarrow}(\hat{s}, s)$")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




    # domain_lb = [0, 0]
    # domain_ub = [10, 10]
    # shapes = [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60], [70, 70], [80, 80], [90, 90], [100, 100]]
    # calc_runtime = []
    # min_deltas = []
    # for shape in shapes:
    #     nstates_1, nstates_2 = shape
    #     rng = np.random.default_rng()
    #     x_edges = random_cell_edges(domain_lb[0], domain_ub[0], nstates_1, rng=rng)
    #     y_edges = random_cell_edges(domain_lb[1], domain_ub[1], nstates_2, rng=rng)
    #     transition_system = build_abstraction(shape, domain_lb, domain_ub, x_edges=x_edges, y_edges=y_edges)

    #     algorithm_start = time.process_time()
    #     min_delta = 0.0
    #     count = 0
    #     for i in range(nstates_1):
    #         for j in range(nstates_2):
    #             cell = (i, j)
    #             delta = 0.0
    #             for _ in range(500):
    #                 delta_sat = max_sampled_distance_to_successors(
    #                     x_edges,
    #                     y_edges,
    #                     cell,
    #                     transition_system,
    #                     delta,
    #                     num_samples=500)
    #                 if np.abs(delta - delta_sat) <= 1e-1:
    #                     delta = max(delta, delta_sat)
    #                     break
    #                 delta = max(delta, delta_sat)

    #             if delta > min_delta:
    #                 min_delta = delta
                
    #             if count % 100 == 0:
    #                 print(f"    > Iter. {count} for shape={shape}")
    #             count += 1

    #     algorithm_runtime = time.process_time() - algorithm_start

    #     calc_runtime.append(algorithm_runtime)
    #     min_deltas.append(min_delta)

    #     print(
    #         f"shape={shape}, cpu_runtime={algorithm_runtime:.4f}s, min_delta={min_delta:.2f}"
    #     )

    # num_states = [s[0] * s[1] for s in shapes]

    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # axes[0].plot(num_states, calc_runtime, marker="o")
    # axes[0].set_xlabel("# abstract states")
    # axes[0].set_ylabel("Runtime (CPU seconds)")
    # axes[0].grid(True)

    # axes[1].plot(num_states, min_deltas, marker="o")
    # axes[1].set_xlabel("# abstract states")
    # axes[1].set_ylabel("$\hat{d}^{\leftarrow}(\hat{s}, s)$")
    # axes[1].grid(True)

    # fig.tight_layout()
    # plt.show()
        