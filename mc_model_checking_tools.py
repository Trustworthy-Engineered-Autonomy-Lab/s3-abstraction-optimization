# Libraries
import numpy as np
import pyModelChecking as pmc
import grid_plot_tools as gpt
import pyModelChecking.CTL as CTL
from pyModelChecking.CTL import A, E, G, F, Imply
import torch
import torch.nn as nn
from pathlib import Path


# MountainCar open-loop dynamics
def mc_ol_dynamics(state, action):
    p, v = state
    v_next = v + 0.001*(action - 1) - 0.0025*np.cos(3*p)
    v_next = np.clip(v_next, -0.07, 0.07)
    p_next = p + v_next
    p_next = np.clip(p_next, -1.2, 0.6)
    if p_next <= -1.2 and v_next < 0:
        v_next = 0.0
    return np.array([p_next, v_next])

# DQN policy
class DQN(nn.Module):

    # Initialize the neural network
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), # Input layer
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Output layer
        )

    def forward(self, x):
        return self.net(x)

# Lazy-load cached policy (avoid re-instantiation)
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_POLICY_CACHE = None
_POLICY_PATH = Path(__file__).resolve().parent / 'policy.pth'

def get_policy():
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        net = DQN(2, 3, 128).to(_DEVICE)
        state_dict = torch.load(str(_POLICY_PATH), map_location=_DEVICE)
        net.load_state_dict(state_dict)
        net.eval()
        _POLICY_CACHE = net
    return _POLICY_CACHE

def policy_action(state):
    net = get_policy()
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        q = net(s)
        return int(q.argmax(dim=1).item())

# Closed-loop dynamics using greedy policy (no need to pass network each call)
def mc_cl_dynamics(x):
    x_np = np.asarray(x, dtype=np.float64)
    a_idx = policy_action(x_np)
    return mc_ol_dynamics(x_np, a_idx)


# Helper to find intersecting cells from AABB in y-space
def intersecting_cells_from_y_aabb(y1_params, y2_params, y1_min, y1_max, y2_min, y2_max):
    """Sound over-approx: return all (i,j) whose Y-cell rectangle overlaps the given AABB.
    """
    y1_params = np.asarray(y1_params, dtype=float)
    y2_params = np.asarray(y2_params, dtype=float)
    n1 = len(y1_params) - 1
    n2 = len(y2_params) - 1

    # If completely outside the overall grid bounds, no in-grid successors.
    grid_y1_min, grid_y1_max = float(y1_params[0]), float(y1_params[-1])
    grid_y2_min, grid_y2_max = float(y2_params[0]), float(y2_params[-1])
    if y1_max < grid_y1_min or y1_min > grid_y1_max or y2_max < grid_y2_min or y2_min > grid_y2_max:
        return []

    # Use side="right" for both ends.
    # This avoids the empty-set edge case when y_min == y_max equals a grid line:
    #   side="left" would assign the max to the lower cell, potentially giving i_hi < i_lo.
    i_lo = int(np.searchsorted(y1_params, y1_min, side="right") - 1)
    i_hi = int(np.searchsorted(y1_params, y1_max, side="right") - 1)
    j_lo = int(np.searchsorted(y2_params, y2_min, side="right") - 1)
    j_hi = int(np.searchsorted(y2_params, y2_max, side="right") - 1)

    i_lo = max(0, min(n1 - 1, i_lo))
    i_hi = max(0, min(n1 - 1, i_hi))
    j_lo = max(0, min(n2 - 1, j_lo))
    j_hi = max(0, min(n2 - 1, j_hi))

    if i_hi < i_lo or j_hi < j_lo:
        return []

    return [(i, j) for i in range(i_lo, i_hi + 1) for j in range(j_lo, j_hi + 1)]


# Helper to check if point is in valid y domain
def point_in_parallelogram(p, verts, tol=1e-9):
    """Return True iff point p is inside/on the parallelogram defined by verts (CCW order)."""
    verts = np.asarray(verts, dtype=float)
    p = np.asarray(p, dtype=float)

    v0 = verts[0]
    e1 = verts[1] - v0
    e2 = verts[-1] - v0
    E = np.column_stack([e1, e2])  # 2x2

    # Solve p = v0 + a*e1 + b*e2
    ab = np.linalg.solve(E, p - v0)
    a, b = float(ab[0]), float(ab[1])
    return (-tol <= a <= 1.0 + tol) and (-tol <= b <= 1.0 + tol)


# Goal state labeling function
def is_goal_state(x_space_vertices):
    for v in x_space_vertices:
        if v[0] < 0.5:
            return False
    return True


def make_kripke_from_params(y1_params, y2_params, M):

    # Initialize Kripke structure parameters
    nstates_1 = len(y1_params) - 1
    nstates_2 = len(y2_params) - 1
    n_kripke_states = nstates_1 * nstates_2

    # Initialize Kripike structure components
    kripke_states = list(range(n_kripke_states))
    kripke_transitions = set()
    kripke_labels = {}

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Precompute transformation inverse
    M = np.asarray(M, dtype=float)
    invM = np.linalg.inv(M)

    # Loop through each abstract state
    for i in range(nstates_1):
        y1_lo, y1_hi = y1_params[i], y1_params[i+1]
        for j in range(nstates_2):
            y2_lo, y2_hi = y2_params[j], y2_params[j+1]
            corners = np.array([
                [y1_lo, y2_lo],
                [y1_lo, y2_hi],
                [y1_hi, y2_hi],
                [y1_hi, y2_lo]])

            # Compute y-space image
            x_corners = (M @ corners.T).T
            x_next = np.array([mc_cl_dynamics(x_corner) for x_corner in x_corners])
            y_next = (invM @ x_next.T).T

            # Check label of current cell
            label = ['goal'] if is_goal_state(x_corners) else ['safe']

            # Identify successor cells
            y1_min, y1_max = float(y_next[:, 0].min()), float(y_next[:, 0].max())
            y2_min, y2_max = float(y_next[:, 1].min()), float(y_next[:, 1].max())
            succ_cells = intersecting_cells_from_y_aabb(y1_params, y2_params, y1_min, y1_max, y2_min, y2_max)

            # Allocate relations to Kripke structure components
            src = cell_state_id(i, j)
            for (ip, jp) in succ_cells:
                dst = cell_state_id(ip, jp)
                kripke_transitions.add((src, dst))

            # # Force self loop if no in-grid successors and no OOB transition
            # if (not succ_cells):
            #     kripke_transitions.add((src, src))

            kripke_labels[src] = label
            # print(f"{src}: {label[0]} cell ({i},{j}) -> {len(succ_cells)} in-grid successors" + (" + OOB" if hits_oob else ""))

    # Define initial states (in bounds, non-goal states)
    initial_states = [s for s in kripke_states]

    # Make the Kripke structure
    kripke_structure = pmc.Kripke(S=kripke_states,
                                  S0=initial_states,
                                  R=list(kripke_transitions),
                                  L=kripke_labels)
    
    return kripke_structure


def model_check_kripke(kripke_structure):

    phi = 'A (F goal)'  # Eventually reach goal
    sat = CTL.modelcheck(kripke_structure, phi)
    
    return sat


def check_ground_truth(y1_params, y2_params, M):

    # Initialize ground truth dictionary
    nstates_1 = len(y1_params) - 1
    nstates_2 = len(y2_params) - 1
    ground_truth_check = {}

    # ID labeling function
    def cell_state_id(i, j):
        return i * nstates_2 + j

    # Precompute transformation matrix
    M = np.asarray(M, dtype=float)

    # Loop through each abstract state
    for i in range(nstates_1):
        y1_lo, y1_hi = y1_params[i], y1_params[i+1]
        for j in range(nstates_2):
            y2_lo, y2_hi = y2_params[j], y2_params[j+1]

            # Transform corners to x-space
            corners = np.array([
                [y1_lo, y2_lo],
                [y1_lo, y2_hi],
                [y1_hi, y2_hi],
                [y1_hi, y2_lo]])
            x_verts = (M @ corners.T).T

            # Push vertices through dynamics until any fail or all succeed
            src = cell_state_id(i, j)
            max_steps = 10_000
            reached_terminal = False
            for _ in range(max_steps):

                if is_goal_state(x_verts):
                    print(f"Cell ({i},{j}): all goal reached")
                    ground_truth_check[src] = 'goal'
                    reached_terminal = True
                    break

                x_verts = np.array([mc_cl_dynamics(v) for v in x_verts])

            if not reached_terminal:
                ground_truth_check[src] = 'unk'

            # print(f"Cell ({i},{j}): " +
            #       ("OOB reached" if any_oob else
            #        "All goal reached" if all_goal else
            #        "Neither OOB nor all goal reached"))

    return ground_truth_check

if __name__ == "__main__":

    K = pmc.Kripke(S=[0, 1, 3],
                R=[(0, 2), (2, 2), (0, 1), (1, 0), (3, 2)],
                L={1: ['p', 'q'], 2: ['p', 'q'], 3: ['q']})

    print(K.states())
    print(K.transitions())

def false_negative_rate(true_safe_states, checked_safe_states):
    false_negative_states = {s for s in true_safe_states if s not in checked_safe_states}
    denom = len(true_safe_states)
    fnr = (len(false_negative_states) / denom) if denom > 0 else float('nan')
    return fnr, false_negative_states
