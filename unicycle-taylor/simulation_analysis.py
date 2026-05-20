# =====================================================================
# Description: all essential functions for evaluating simulation
# metrics between the concrete and abstract systems
# =====================================================================

# =====================================================================
# Libraries for these analyses
# =====================================================================

import numpy as np
import unicycle_system as us
import unicycle_abstraction as ua


# =====================================================================
# Meausrement function between concrete/abstract spaces
# =====================================================================

def shortest_distance_from_state_to_cell(
        edges,
        state,
        cell_index
        ):
    """
    Standardized way to measure the distance between a concrete and abstract
    state (shortest distance between concrete and concretized abstract state)
    """

    xlb, xub, ylb, yub, thetalb, thetaub = cell_bounds_from_index(edges,
                                                                  cell_index)
    x, y, theta = np.asarray(state)

    dx = max(xlb - x, 0.0, x - xub)
    dy = max(ylb - y, 0.0, y - yub)
    dtheta = shortest_circular_distance_to_interval(theta, thetalb, thetaub)

    return np.sqrt(dx**2 + dy**2 + dtheta**2)

def shortest_circular_distance_to_interval(
        theta,
        theta_lb,
        theta_ub
        ):
    """
    Shortest angular distance from theta to a theta interval on S^1.
    """

    theta = us.wrap_to_pi(theta)
    theta_lb = us.wrap_to_pi(theta_lb)
    theta_ub = us.wrap_to_pi(theta_ub)

    if theta_lb <= theta_ub:
        if theta_lb <= theta <= theta_ub:
            return 0.0
    else:
        if theta >= theta_lb or theta <= theta_ub:
            return 0.0

    dist_to_lb = abs(us.wrap_to_pi(theta - theta_lb))
    dist_to_ub = abs(us.wrap_to_pi(theta - theta_ub))
    return min(dist_to_lb, dist_to_ub)

def cell_bounds_from_index(
        edges,
        cell_index
        ):
    """
    Standardized way to measure the distance between a concrete and abstract
    state (shortest distance between concrete and concretized abstract state)
    """

    i, j, k = cell_index
    x_edges, y_edges, theta_edges = edges

    xlb = x_edges[i]
    xub = x_edges[i + 1]
    ylb = y_edges[j]
    yub = y_edges[j + 1]
    thetalb = theta_edges[k]
    thetaub = theta_edges[k + 1]

    return xlb, xub, ylb, yub, thetalb, thetaub


# =====================================================================
# Methods to sample from post-image of delta-close pre-image set
# =====================================================================

def sample_states_within_delta_of_cell(
        edges,
        cell_index,
        delta,
        num_samples,
        rng=None,
        clip_to_domain=True
        ):

    xlb, xub, ylb, yub, thetalb, thetaub = cell_bounds_from_index(
        edges,
        cell_index
    )

    rng = np.random.default_rng(rng)

    if delta < 0:
        raise ValueError("delta must be nonnegative")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")
    if num_samples == 0:
        return np.empty((0, 3))

    if clip_to_domain:
        x_sample_lb = max(edges[0][0], xlb - delta)
        x_sample_ub = min(edges[0][-1], xub + delta)

        y_sample_lb = max(edges[1][0], ylb - delta)
        y_sample_ub = min(edges[1][-1], yub + delta)
    else:
        x_sample_lb = xlb - delta
        x_sample_ub = xub + delta
        y_sample_lb = ylb - delta
        y_sample_ub = yub + delta

    theta_span = thetaub - thetalb
    expanded_theta_span = theta_span + 2.0 * delta
    sample_full_circle = expanded_theta_span >= 2.0 * np.pi

    samples = []

    while len(samples) < num_samples:
        batch_size = max(32, 2 * (num_samples - len(samples)))

        xs = rng.uniform(x_sample_lb, x_sample_ub, size=batch_size)
        ys = rng.uniform(y_sample_lb, y_sample_ub, size=batch_size)

        if sample_full_circle:
            thetas = rng.uniform(-np.pi, np.pi, size=batch_size)
        else:
            theta_offsets = rng.uniform(
                0.0,
                expanded_theta_span,
                size=batch_size
            )
            thetas = us.wrap_to_pi((thetalb - delta) + theta_offsets)

        batch = np.column_stack((xs, ys, thetas))

        for state in batch:
            if shortest_distance_from_state_to_cell(edges, state, cell_index) <= delta:
                samples.append(state)
                if len(samples) == num_samples:
                    break

    samples = np.asarray(samples)
    rng.shuffle(samples)
    return samples


# =====================================================================
# Methods to tightly approximate the upward simulation metric
# =====================================================================

def max_sampled_distance_to_successors(
        transition_system,
        edges,
        cell,
        delta=0.0,
        num_samples=1000,
        rng=None
        ):
    """
    Computes a candidate delta that constitutes a valid approx. simulation relation
    for a particular abstract state.
    """

    successor_cells = transition_system[cell]
    state_samples = sample_states_within_delta_of_cell(
        edges,
        cell,
        delta,
        num_samples,
        rng=rng
    )
    next_state_samples = np.array([us.cl_system(state) for state in state_samples])

    max_min_delta = 0.0
    for successor in successor_cells:
        min_delta = np.inf
        for state in next_state_samples:
            delta = shortest_distance_from_state_to_cell(edges, state, successor)
            if delta < min_delta:
                min_delta = delta
        if min_delta > max_min_delta:
            max_min_delta = min_delta
    
    return max_min_delta


    # return max((shortest_distance_from_state_to_cell(
    #                 edges,
    #                 state,
    #                 successor
    #             )
    #             for successor in successor_cells
    #             for state in next_state_samples),
    #            default=np.nan)

def compute_satisficing_delta(
        transition_system,
        edges,
        cell,
        delta=0.0,
        delta_iterations=500,
        num_samples=500,
        tol = 1e-1,
        rng=None
        ):
    """
    Employs fixed-point iteration to approximate the smallest possible delta that
    constitutes a valid delta-approximate simulation relation, i.e., the simulation
    metric, for a particular abstract state.
    """
    
    for _ in range(delta_iterations):
        delta_sat = max_sampled_distance_to_successors(
            transition_system,
            edges,
            cell,
            delta=delta,
            num_samples=num_samples,
            rng=rng
        )
        if np.abs(delta - delta_sat) <= tol:
            break
        # delta = max(delta, delta_sat)
        delta = delta_sat
        print(delta)

    return delta

def compute_satisficing_delta_smooth(
        transition_system,
        edges,
        cell,
        delta=0.0,
        delta_iterations=100,
        num_samples=500,
        tol = 1e-1,
        smoothness = 1.0,
        rng=23
        ):
    """
    Employs fixed-point iteration to approximate the smallest possible delta that
    constitutes a valid delta-approximate simulation relation, i.e., the simulation
    metric, for a particular abstract state.
    """
    
    for _ in range(delta_iterations):
        delta_sat = max_sampled_distance_to_successors(
            transition_system,
            edges,
            cell,
            delta=delta,
            num_samples=num_samples,
            rng=rng
        )

        if np.abs(delta - delta_sat) <= tol:
            break
        else:
            grad = smoothness * (delta_sat - delta)
            delta += grad
            # print(f"Delta: {delta:.3f}, Min. satisfying: {delta_sat:.3f}, Gradient: {grad:.3f}")

    return delta

def approx_upward_metric(
        transition_system,
        shape,
        edges,
        delta_iterations=500,
        num_samples=500,
        tol = 1e-1
        ):
    """
    Iterates over all cells and determines the largest satisficing delta
    """
    nstates_1, nstates_2, nstates_3 = shape

    min_delta = 0.0
    count = 0
    for i in range(nstates_1):
        for j in range(nstates_2):
            for k in range(nstates_3):
                cell = (i, j, k)
                delta = 0.0
                delta_sat = compute_satisficing_delta_smooth(transition_system,
                                                            edges,
                                                            cell,
                                                            delta,
                                                            delta_iterations=delta_iterations,
                                                            num_samples=num_samples,
                                                            tol=tol)
                if delta_sat > min_delta:
                    min_delta = delta_sat
                    print(f"Current min delta: {min_delta}")
                if count % 1 == 0:
                    print(f"Processed = {count}")
                count += 1

    return min_delta

# =====================================================================
# Methods to reformat the Kripke components output during abstraction
# =====================================================================

def make_transition_system_dict(
        kripke_states,
        kripke_transitions,
        nstates_1,
        nstates_2,
        nstates_3
        ):
    """
    Convert Kripke transitions given as (src, dst) edges into a dictionary
    mapping each cell (i, j, k) to its list of successor cells.
    """

    oob_state_id = kripke_states[-1]

    successor_dict = {
        ua.id_to_cell(state_id, nstates_1, nstates_2, nstates_3): []
        for state_id in kripke_states
        if state_id != oob_state_id
    }

    for src, dst in kripke_transitions:
        if src == oob_state_id or dst == oob_state_id:
            continue
        src_cell = ua.id_to_cell(src, nstates_1, nstates_2, nstates_3)
        dst_cell = ua.id_to_cell(dst, nstates_1, nstates_2, nstates_3)
        successor_dict[src_cell].append(dst_cell)

    for state_id in successor_dict:
        successor_dict[state_id].sort()

    return successor_dict


# =====================================================================
# Place to test above methods
# =====================================================================

if __name__ == "__main__":

    # Fixed abstraction and environment settings
    nstates_1 = 20
    nstates_2 = 20
    nstates_3 = 20
    shape = [nstates_1, nstates_2, nstates_3]
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])

    # Define the initial domain
    init_domain_lb = np.array([20.0, 0.0, 0.0])
    init_domain_ub = np.array([50.0, 40.0, 0.0])

    # Initialize the grid parameters
    x_edges = np.linspace(domain_lb[0], domain_ub[0], nstates_1+1)
    y_edges = np.linspace(domain_lb[1], domain_ub[1], nstates_2+1)
    theta_edges = np.linspace(domain_lb[2], domain_ub[2], nstates_3+1)
    edges = [x_edges, y_edges, theta_edges]

    # Build the initial Kripke components
    kripke_components = ua.build_abstraction(x_edges, y_edges, theta_edges, verbose=True)
    transition_system = make_transition_system_dict(kripke_components['kripke_states'],
                                                    kripke_components['kripke_transitions'],
                                                    nstates_1,
                                                    nstates_2,
                                                    nstates_3)

    # cell_id = (10, 10, 10)
    # upward_delta = compute_satisficing_delta_smooth(transition_system, edges,
    #                                                 cell_id,
    #                                                 delta = 0.0,
    #                                                 delta_iterations=100,
    #                                                 num_samples=100,
    #                                                 tol=1e-3)
    
    upward_delta = approx_upward_metric(transition_system,
                                        shape,
                                        edges,
                                        delta_iterations=50,
                                        num_samples=30,
                                        tol=1e-1)
    print(upward_delta)

    # max_min_delta = max_sampled_distance_to_successors(transition_system,
    #                                                   edges,
    #                                                   (2, 2, 2),
    #                                                   delta = 2.12,
    #                                                   num_samples=5000)
    # print(max_min_delta)


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
