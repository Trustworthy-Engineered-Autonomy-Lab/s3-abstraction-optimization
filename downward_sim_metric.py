# Libraries
import numpy as np



def dynamics(state):
    A = np.array([[0.8, -0.3],
                   [0.3,  0.8]])
    x_star = np.array([5.0, 5.0])
    state = np.asarray(state)
    return (state - x_star) @ A.T + x_star


def build_abstraction(
        shape,
        domain_lb,
        domain_ub
        ):
    
    nstates_1, nstates_2 = shape
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
            next_cells = intersecting_cells_from_aabb(x_edges, y_edges, next_verts)

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


def cells_within_distance_of_state(
        x_edges,
        y_edges,
        state,
        epsilon
        ):

    nstates_1 = len(x_edges) - 1
    nstates_2 = len(y_edges) - 1

    return [(i, j)
            for i in range(nstates_1)
            for j in range(nstates_2)
            if shortest_distance_from_state_to_cell(
                x_edges,
                y_edges,
                state,
                (i, j)
            ) <= epsilon]


def min_distance_to_successor_cells(
        x_edges,
        y_edges,
        state,
        epsilon,
        transition_system
        ):
    epsilon_cells = cells_within_distance_of_state(x_edges, y_edges, state, epsilon)
    all_successors = []
    for epsilon_cell in epsilon_cells:
        all_successors.extend(transition_system[epsilon_cell])

    next_state = dynamics(state)
    return min((shortest_distance_from_state_to_cell(
                    x_edges,
                    y_edges,
                    next_state,
                    succ
                )
                for succ in all_successors),
               default=np.inf)



# Helper to find intersecting cells from AABB in y-space
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



if __name__ == "__main__":

    # Build a simple nondeterminsitic transition system of the dynamical system
    domain_lb = [0, 0]
    domain_ub = [10, 10]
    shape = [10, 10]
    nstates_1, nstates_2 = shape
    x_edges = np.linspace(domain_lb[0], domain_ub[0], nstates_1 + 1)
    y_edges = np.linspace(domain_lb[1], domain_ub[1], nstates_2 + 1)
    transition_system = build_abstraction(shape,
                                          domain_lb,
                                          domain_ub)


    # Initialize a random concrete state and fix an epsilon
    state = np.array([3.3, 3.3])
    epsilon = 0.0

    min_dist = min_distance_to_successor_cells(
        x_edges,
        y_edges,
        state,
        epsilon,
        transition_system
    )

    print(min_dist)




