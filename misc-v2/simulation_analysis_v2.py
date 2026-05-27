# =====================================================================
# Description: all essential functions for evaluating simulation
# metrics between the concrete and abstract systems
# =====================================================================

# =====================================================================
# Libraries for these analyses
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
import unicycle_system as us
import unicycle_abstraction as ua
from scipy.spatial import cKDTree


# =====================================================================
# Measurement function between a concrete subset and abstract state
# =====================================================================

def directed_hausdorff(
        edges,
        cell_index,
        state_samples
        ):
    """
    Approximate the directed Hausdorff distance from a cell to a sampled set.
    The cell is evaluated on a dense tensor grid, and each grid point is
    matched to its nearest sample in ``state_samples`` using the same Euclidean
    metric on ``(x, y, theta)`` used elsewhere in this module, with ``theta``
    unwrapped relative to the cell center.
    """

    state_samples = np.asarray(state_samples, dtype=float)
    xlb, xub, ylb, yub, thetalb, thetaub = cell_bounds_from_index(
        edges,
        cell_index
    )

    theta_mid = 0.5 * (thetalb + thetaub)
    unwrapped_samples = np.array(state_samples, copy=True)
    unwrapped_samples[:, 2] = theta_mid + np.array([
        us.wrap_to_pi(theta - theta_mid) for theta in unwrapped_samples[:, 2]
    ])

    tree = cKDTree(unwrapped_samples)

    grid_points_per_dim = 5
    x_grid = np.linspace(xlb, xub, grid_points_per_dim)
    y_grid = np.linspace(ylb, yub, grid_points_per_dim)
    theta_grid = np.linspace(thetalb, thetaub, grid_points_per_dim)
    query_points = np.stack(
        np.meshgrid(x_grid, y_grid, theta_grid, indexing='ij'),
        axis=-1
    ).reshape(-1, 3)

    nearest_distances, _ = tree.query(query_points)
    return float(np.max(nearest_distances))

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
# Methods to densely approximate the post-image of a concrete subset
# =====================================================================

def sample_states_from_cell_grid(
        edges,
        cell_index,
        grid_density
        ):
    """
    Return the states on a dense tensor grid inside a cell, including the
    cell boundaries.
    """

    xlb, xub, ylb, yub, thetalb, thetaub = cell_bounds_from_index(
        edges,
        cell_index
    )

    x_grid = np.linspace(xlb, xub, grid_density)
    y_grid = np.linspace(ylb, yub, grid_density)
    theta_grid = np.linspace(thetalb, thetaub, grid_density)
    grid_states = np.stack(
        np.meshgrid(x_grid, y_grid, theta_grid, indexing='ij'),
        axis=-1
    ).reshape(-1, 3)

    return grid_states




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
    nstates_1 = 120
    nstates_2 = 120
    nstates_3 = 120
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
    


    # Smoketest
    cell_id = (60, 60, 60)
    next_cells = transition_system[cell_id]

    states = sample_states_from_cell_grid(edges, cell_id, 20)
    next_states = np.array([us.cl_system(state) for state in states])

    hausdorff_distances = [directed_hausdorff(edges, cell, next_states) for cell in next_cells]
    hausdorff_distances = np.array(hausdorff_distances)

    max_hausdorff_distance = np.max(hausdorff_distances)

    # Plots
    xlb, xub, ylb, yub, _, _ = cell_bounds_from_index(edges, cell_id)

    fig, ax = plt.subplots()

    ax.scatter(states[:, 0], states[:, 1], color='tab:blue', s=12, label='Samples')
    ax.scatter(next_states[:, 0], next_states[:, 1], color='tab:red', s=12, label='Next states')

    ax.add_patch(
        plt.Rectangle(
            (xlb, ylb),
            xub - xlb,
            yub - ylb,
            fill=False,
            edgecolor='black',
            linewidth=2,
            label='Initial cell'
        )
    )

    for idx, successor_cell in enumerate(next_cells):
        succ_xlb, succ_xub, succ_ylb, succ_yub, _, _ = cell_bounds_from_index(
            edges,
            successor_cell
        )
        ax.add_patch(
            plt.Rectangle(
                (succ_xlb, succ_ylb),
                succ_xub - succ_xlb,
                succ_yub - succ_ylb,
                fill=False,
                edgecolor='tab:green',
                linewidth=1,
                alpha=0.8,
                label='Successor cells' if idx == 0 else None
            )
        )

    

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Max Hausdorff distance = {max_hausdorff_distance:.4f}')
    ax.legend()
    plt.show()
    
