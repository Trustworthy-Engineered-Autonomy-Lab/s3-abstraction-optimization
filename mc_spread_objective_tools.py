# Libraries
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import jax
import jax.numpy as jnp
from grid_plot_tools import get_yspace_bounds

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





def make_lines_from_gaps(u, lo, hi, min_gap=0.0):
    """
    u: (n_internal,) unconstrained
    Returns (n_internal+2,) including endpoints [lo, ..., hi]
    """
    gaps = jax.nn.softplus(u) + min_gap  # positive
    # normalize so sum(gaps) = hi-lo
    gaps = gaps * ((hi - lo) / (jnp.sum(gaps) + 1e-12))
    internal = lo + jnp.cumsum(gaps)[:-1]
    return jnp.concatenate([jnp.array([lo]), internal, jnp.array([hi])])

def trans_matrix(theta, a1, a2, h):
    """M = H @ S @ R where s1=exp(a1), s2=exp(a2)."""
    theta = jnp.asarray(theta)
    a1 = jnp.asarray(a1)
    a2 = jnp.asarray(a2)
    h = jnp.asarray(h)

    s1 = jnp.exp(a1)
    s2 = jnp.exp(a2)

    c, s = jnp.cos(theta), jnp.sin(theta)
    dtype = jnp.result_type(theta, a1, a2, h)

    R = jnp.array([[c, -s],
                   [s,  c]], dtype=dtype)
    S = jnp.array([[s1, 0.0],
                   [0.0, s2]], dtype=dtype)
    H = jnp.array([[1.0, h],
                   [0.0, 1.0]], dtype=dtype)
    return H @ S @ R

def extract_grid_params(params, *, n1_internal, n2_internal,
                          x1_min, x1_max, x2_min, x2_max,
                          min_gap=0.0):
    """
    Returns:
      M_np   : (2,2) numpy array
      y1_vals: (n1_internal+1,) numpy array of y1 grid line locations
      y2_vals: (n2_internal+1,) numpy array of y2 grid line locations
    """
    # unpack params (JAX arrays)
    u1 = params[:n1_internal]
    u2 = params[n1_internal:n1_internal+n2_internal]
    theta, a1, a2, h = params[n1_internal+n2_internal:]

    # Build M (NOTE: this assumes trans_matrix expects log-scales a1,a2 and exponentiates inside)
    M = trans_matrix(theta, a1, a2, h)

    # Convert M to numpy for your plotting + get_yspace_bounds
    M_np = np.array(jax.device_get(M))

    # Induced Y-domain from current M
    bounds_y, _ = get_yspace_bounds(
        M_np,
        x1_min, x1_max,
        x2_min, x2_max
    )
    y1_lo, y1_hi = bounds_y["y1"]
    y2_lo, y2_hi = bounds_y["y2"]

    # Convert gap-params -> actual y-line locations
    y1_vals = make_lines_from_gaps(u1, y1_lo, y1_hi, min_gap=min_gap)
    y2_vals = make_lines_from_gaps(u2, y2_lo, y2_hi, min_gap=min_gap)

    # Convert to numpy for plotting
    y1_vals_np = np.array(jax.device_get(y1_vals))
    y2_vals_np = np.array(jax.device_get(y2_vals))

    return M_np, y1_vals_np, y2_vals_np