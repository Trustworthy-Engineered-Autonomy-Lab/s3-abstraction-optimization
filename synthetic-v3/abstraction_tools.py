# Libraries
import numpy as np

def build_abstraction(
        x_edges,
        y_edges
        ):
    
    # Initialize Kripike structure components
    nstates_1 = len(x_edges)
    nstates_2 = len(y_edges)
    n_kripke_states = nstates_1 * nstates_2 + 1 # includes out of bounds state
    oob_state_id = n_kripke_states - 1
    kripke_states = list(range(n_kripke_states))  # last state is out-of-bounds
    kripke_transitions = set()
    kripke_labels = {}

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

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