# Libraries
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import jax
import jax.numpy as jnp
from grid_plot_tools import get_yspace_bounds


# ----------------------------
# Differentiable (JAX) pieces
# ----------------------------

_JAX_POLICY_CACHE = None


def _load_dqn_params_jax():
    """Load DQN weights from policy.pth into JAX arrays (cached)."""
    global _JAX_POLICY_CACHE
    if _JAX_POLICY_CACHE is not None:
        return _JAX_POLICY_CACHE

    state_dict = torch.load(str(_POLICY_PATH), map_location="cpu")
    # Keys (verified): net.0.{weight,bias}, net.2.{weight,bias}, net.4.{weight,bias}
    params = {
        "W1": jnp.asarray(state_dict["net.0.weight"].detach().cpu().numpy()),
        "b1": jnp.asarray(state_dict["net.0.bias"].detach().cpu().numpy()),
        "W2": jnp.asarray(state_dict["net.2.weight"].detach().cpu().numpy()),
        "b2": jnp.asarray(state_dict["net.2.bias"].detach().cpu().numpy()),
        "W3": jnp.asarray(state_dict["net.4.weight"].detach().cpu().numpy()),
        "b3": jnp.asarray(state_dict["net.4.bias"].detach().cpu().numpy()),
    }
    _JAX_POLICY_CACHE = params
    return params


def dqn_q_values_jax(state: jnp.ndarray) -> jnp.ndarray:
    """Return Q-values for MountainCar DQN policy.

    state: (...,2)
    returns: (...,3)
    """
    p = _load_dqn_params_jax()
    x = jnp.asarray(state)
    h1 = jax.nn.relu(x @ p["W1"].T + p["b1"])
    h2 = jax.nn.relu(h1 @ p["W2"].T + p["b2"])
    q = h2 @ p["W3"].T + p["b3"]
    return q


def policy_action_soft_jax(state: jnp.ndarray, *, temperature: float = 0.1) -> jnp.ndarray:
    """Differentiable policy: expected action under softmax(Q/T).

    Returns a continuous action in [0,2].
    """
    q = dqn_q_values_jax(state)
    pi = jax.nn.softmax(q / temperature, axis=-1)
    actions = jnp.asarray([0.0, 1.0, 2.0], dtype=pi.dtype)
    return jnp.sum(pi * actions, axis=-1)


def _smooth_clip(x: jnp.ndarray, lo: float, hi: float, sharpness: float = 20.0) -> jnp.ndarray:
    """Smooth approximation to clip using tanh."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return mid + half * jnp.tanh(sharpness * (x - mid) / (half + 1e-12))


def mc_ol_dynamics_soft_jax(
    state: jnp.ndarray,
    action_cont: jnp.ndarray,
    *,
    clip_sharpness: float = 20.0,
    wall_sharpness: float = 200.0,
) -> jnp.ndarray:
    """Differentiable MountainCar open-loop dynamics with continuous action in [0,2].

    state: (...,2) (p,v)
    action_cont: (...,)
    returns: (...,2)
    """
    p = state[..., 0]
    v = state[..., 1]

    v_next = v + 0.001 * (action_cont - 1.0) - 0.0025 * jnp.cos(3.0 * p)
    v_next = _smooth_clip(v_next, -0.07, 0.07, sharpness=clip_sharpness)

    p_next = p + v_next
    p_next = _smooth_clip(p_next, -1.2, 0.6, sharpness=clip_sharpness)

    # Smooth variant of: if p_next <= -1.2 and v_next < 0: v_next = 0
    hit_left = jax.nn.sigmoid(wall_sharpness * (-1.2 - p_next))
    going_left = jax.nn.sigmoid(wall_sharpness * (-v_next))
    v_next = v_next * (1.0 - hit_left * going_left)

    return jnp.stack([p_next, v_next], axis=-1)


def mc_cl_dynamics_soft_jax(
    state: jnp.ndarray,
    *,
    temperature: float = 0.1,
    clip_sharpness: float = 20.0,
    wall_sharpness: float = 200.0,
) -> jnp.ndarray:
    a = policy_action_soft_jax(state, temperature=temperature)
    return mc_ol_dynamics_soft_jax(
        state,
        a,
        clip_sharpness=clip_sharpness,
        wall_sharpness=wall_sharpness,
    )


def softmax_tau(x, tau):
    return tau * jax.scipy.special.logsumexp(x / tau, axis=-1)


def softmin_tau(x, tau):
    return -tau * jax.scipy.special.logsumexp(-x / tau, axis=-1)


def _soft_overlap_count(
    interval_min: jnp.ndarray,
    interval_max: jnp.ndarray,
    edges: jnp.ndarray,
    *,
    beta: float = 200.0,
):
    """Soft count of how many 1D bins [e_i,e_{i+1}] overlap [min,max].

    Returns a value in [0, nbins] with gradients.
    """
    edges = jnp.asarray(edges)
    left = edges[:-1]
    right = edges[1:]

    # overlap if right >= min and left <= max
    s1 = jax.nn.sigmoid(beta * (right - interval_min))
    s2 = jax.nn.sigmoid(beta * (interval_max - left))
    return jnp.sum(s1 * s2)


def diff_successor_count_objective(
    params,
    *,
    y1_lo,
    y1_hi,
    y2_lo,
    y2_hi,
    n1_internal,
    n2_internal,
    min_gap=0.0,
    temperature: float = 0.1,
    horizon: int = 1,
    tau_bbox: float = 0.02,
    beta_overlap: float = 200.0,
    weight_mode: str = "uniform",
    weight_power: float = 1.0,
):
    """Option A: differentiable proxy for successor count.

    This targets what drives spurious transitions in your model checker:
    successors are computed from the y-space AABB of mapped-back corner images.

    Proxy used here:
    - compute soft y-min/y-max per axis over the 4 corners
    - softly count how many y-bins overlap that interval
    - successor_count ≈ count_y1 * count_y2
    """
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    theta, a1, a2, h = params[n1_internal + n2_internal :]

    M = trans_matrix(theta, a1, a2, h)
    invM = jnp.linalg.inv(M)

    y1 = make_lines_from_gaps(u1, y1_lo, y1_hi, min_gap=min_gap)
    y2 = make_lines_from_gaps(u2, y2_lo, y2_hi, min_gap=min_gap)

    y1a, y1b = y1[:-1], y1[1:]
    y2a, y2b = y2[:-1], y2[1:]

    Y00 = jnp.stack(jnp.meshgrid(y1a, y2a, indexing="ij"), axis=-1)
    Y01 = jnp.stack(jnp.meshgrid(y1a, y2b, indexing="ij"), axis=-1)
    Y11 = jnp.stack(jnp.meshgrid(y1b, y2b, indexing="ij"), axis=-1)
    Y10 = jnp.stack(jnp.meshgrid(y1b, y2a, indexing="ij"), axis=-1)
    Ycorn = jnp.stack([Y00, Y01, Y11, Y10], axis=2)  # (n1,n2,4,2)

    Xcorn = (Ycorn @ M.T)

    def one_step(x):
        return mc_cl_dynamics_soft_jax(x, temperature=temperature)

    X = Xcorn.reshape(-1, 2)
    for _ in range(int(horizon)):
        X = jax.vmap(one_step)(X)
    Xnext = X.reshape(Xcorn.shape)

    # Map back to y
    Ynext = Xnext @ invM.T

    # soft bbox in y
    y_max = softmax_tau(Ynext, tau_bbox)  # (n1,n2,2)
    y_min = softmin_tau(Ynext, tau_bbox)  # (n1,n2,2)

    # vectorize overlap counts over all cells
    def cell_count(ymin: jnp.ndarray, ymax: jnp.ndarray):
        c1 = _soft_overlap_count(ymin[0], ymax[0], y1, beta=beta_overlap)
        c2 = _soft_overlap_count(ymin[1], ymax[1], y2, beta=beta_overlap)
        return c1 * c2

    counts = jax.vmap(jax.vmap(cell_count))(y_min, y_max)  # (n1,n2)

    if weight_mode == "uniform":
        return jnp.mean(counts)
    if weight_mode == "cell_area_y":
        cell_area = (y1b - y1a)[:, None] * (y2b - y2a)[None, :]
        weights = jnp.power(cell_area + 1e-12, weight_power)
        return jnp.sum(weights * counts) / (jnp.sum(weights) + 1e-12)

    raise ValueError(f"Unknown weight_mode: {weight_mode}")


def diff_objective(
    params,
    *,
    y1_lo,
    y1_hi,
    y2_lo,
    y2_hi,
    n1_internal,
    n2_internal,
    tau=0.05,
    min_gap=0.0,
    temperature: float = 0.1,
    horizon: int = 1,
    measure_space: str = "x",
    weight_mode: str = "uniform",
    weight_power: float = 1.0,
):
    """Differentiable spread objective for MountainCar closed-loop dynamics.

    Mirrors the synthetic objective: build grid in y-space, map to x-space by M,
    push corners through closed-loop dynamics, and measure (smooth) bbox area.

    Parameters
    ----------
    horizon: int
        Number of closed-loop steps to propagate corners (>=1). Larger horizon
        tends to reduce degeneracy but increases compute.
    weight_mode: "uniform" | "cell_area_y"
        If "cell_area_y", weights each cell by its current y-space area.
    """
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    theta, a1, a2, h = params[n1_internal + n2_internal :]

    M = trans_matrix(theta, a1, a2, h)

    y1 = make_lines_from_gaps(u1, y1_lo, y1_hi, min_gap=min_gap)
    y2 = make_lines_from_gaps(u2, y2_lo, y2_hi, min_gap=min_gap)

    y1a, y1b = y1[:-1], y1[1:]
    y2a, y2b = y2[:-1], y2[1:]

    Y00 = jnp.stack(jnp.meshgrid(y1a, y2a, indexing="ij"), axis=-1)
    Y01 = jnp.stack(jnp.meshgrid(y1a, y2b, indexing="ij"), axis=-1)
    Y11 = jnp.stack(jnp.meshgrid(y1b, y2b, indexing="ij"), axis=-1)
    Y10 = jnp.stack(jnp.meshgrid(y1b, y2a, indexing="ij"), axis=-1)
    Ycorn = jnp.stack([Y00, Y01, Y11, Y10], axis=2)  # (n1,n2,4,2)

    Xcorn = (Ycorn @ M.T)

    # Propagate corners through differentiable closed-loop.
    def one_step(x):
        return mc_cl_dynamics_soft_jax(x, temperature=temperature)

    X = Xcorn.reshape(-1, 2)
    for _ in range(int(horizon)):
        X = jax.vmap(one_step)(X)
    Xnext = X.reshape(Xcorn.shape)

    if measure_space == "x":
        Z = Xnext
    elif measure_space == "y":
        invM = jnp.linalg.inv(M)
        # y = invM @ x  (with x as row-vectors => x @ invM.T)
        Z = Xnext @ invM.T
    else:
        raise ValueError(f"Unknown measure_space: {measure_space}")

    xmax = softmax_tau(Z, tau)
    xmin = softmin_tau(Z, tau)
    w = xmax - xmin
    spread = w[..., 0] * w[..., 1]  # (n1,n2)

    if weight_mode == "uniform":
        return jnp.mean(spread)
    if weight_mode == "cell_area_y":
        cell_area = (y1b - y1a)[:, None] * (y2b - y2a)[None, :]
        weights = jnp.power(cell_area + 1e-12, weight_power)
        return jnp.sum(weights * spread) / (jnp.sum(weights) + 1e-12)

    raise ValueError(f"Unknown weight_mode: {weight_mode}")

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