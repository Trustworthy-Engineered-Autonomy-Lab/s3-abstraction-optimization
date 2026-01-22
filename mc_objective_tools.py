# Libraries
import jax
import jax.numpy as jnp
import numpy as np
import itertools
import torch
from pathlib import Path

'''Transformation and dynamics for synthetic objective experiments'''

COARSENESS_FACTOR = 5.0
_JAX_POLICY_CACHE = None
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_POLICY_CACHE = None
_POLICY_PATH = Path(__file__).resolve().parent / 'policy.pth'

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

    v_next = v + COARSENESS_FACTOR * 0.001 * (action_cont - 1.0) - 0.0025 * jnp.cos(3.0 * p)
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




def dynamics():
    pass



def rot_matrix(theta):
    theta = jnp.asarray(theta)
    c, s = jnp.cos(theta), jnp.sin(theta)
    dtype = jnp.result_type(theta)
    return jnp.array([[c, -s], [s, c]], dtype=dtype)


''' Objective functions and helpers '''

# Compute sum of image AABB areas
def image_area(
    params,
    *,
    x_domain,
    n1_internal,
    n2_internal,
):
    """Sum of post-image AABB areas for each abstract cell.

    JAX-compatible and differentiable w.r.t. `params`.
    `n1_internal` and `n2_internal` are treated as static (Python ints).
    """

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    # theta = params[n1_internal + n2_internal]

    y1_lo, y1_hi, y2_lo, y2_hi = x_domain
    # M = rot_matrix(theta)

    # Convert gap-params -> actual y-line locations
    y1_params = make_lines_from_gaps(u1, y1_lo, y1_hi)
    y2_params = make_lines_from_gaps(u2, y2_lo, y2_hi)

    # Build all cells' corners: (n1, n2, 4, 2)
    y1_los = y1_params[:-1]
    y1_his = y1_params[1:]
    y2_los = y2_params[:-1]
    y2_his = y2_params[1:]

    n1 = y1_los.shape[0]
    n2 = y2_los.shape[0]
    y1_lo_grid = jnp.broadcast_to(y1_los[:, None], (n1, n2))
    y1_hi_grid = jnp.broadcast_to(y1_his[:, None], (n1, n2))
    y2_lo_grid = jnp.broadcast_to(y2_los[None, :], (n1, n2))
    y2_hi_grid = jnp.broadcast_to(y2_his[None, :], (n1, n2))

    corners = jnp.stack(
        [
            jnp.stack([y1_lo_grid, y2_lo_grid], axis=-1),
            jnp.stack([y1_lo_grid, y2_hi_grid], axis=-1),
            jnp.stack([y1_hi_grid, y2_hi_grid], axis=-1),
            jnp.stack([y1_hi_grid, y2_lo_grid], axis=-1),
        ],
        axis=-2,
    )

    # Map y -> x: x = M @ y  (with row vectors: y @ M.T)
    # x_corners = corners @ M.T
    x_next = mc_cl_dynamics_soft_jax(corners)

    x1 = x_next[..., 0]
    x2 = x_next[..., 1]
    x1_lo_post = jnp.min(x1, axis=-1)
    x1_hi_post = jnp.max(x1, axis=-1)
    x2_lo_post = jnp.min(x2, axis=-1)
    x2_hi_post = jnp.max(x2, axis=-1)
    img_area = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post)
    return jnp.sum(img_area)

# Compute sum of image AABB areas divided by respective parent AABB areas
def image_area_over_parent(
    params,
    *,
    x_domain,
    n1_internal,
    n2_internal,
):

    # Unpack arguments
    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    y1_lo, y1_hi, y2_lo, y2_hi = x_domain

    # Convert gap-params -> actual y-line locations
    y1_params = make_lines_from_gaps(u1, y1_lo, y1_hi)
    y2_params = make_lines_from_gaps(u2, y2_lo, y2_hi)

    # Build all cells' corners: (n1, n2, 4, 2)
    y1_los = y1_params[:-1]
    y1_his = y1_params[1:]
    y2_los = y2_params[:-1]
    y2_his = y2_params[1:]

    n1 = y1_los.shape[0]
    n2 = y2_los.shape[0]
    y1_lo_grid = jnp.broadcast_to(y1_los[:, None], (n1, n2))
    y1_hi_grid = jnp.broadcast_to(y1_his[:, None], (n1, n2))
    y2_lo_grid = jnp.broadcast_to(y2_los[None, :], (n1, n2))
    y2_hi_grid = jnp.broadcast_to(y2_his[None, :], (n1, n2))

    corners = jnp.stack(
        [
            jnp.stack([y1_lo_grid, y2_lo_grid], axis=-1),
            jnp.stack([y1_lo_grid, y2_hi_grid], axis=-1),
            jnp.stack([y1_hi_grid, y2_hi_grid], axis=-1),
            jnp.stack([y1_hi_grid, y2_lo_grid], axis=-1),
        ],
        axis=-2,
    )
    x_next = mc_cl_dynamics_soft_jax(corners)

    # Compute parent AABB areas
    x1 = corners[..., 0]
    x2 = corners[..., 1]
    x1_lo_pre = jnp.min(x1, axis=-1)
    x1_hi_pre = jnp.max(x1, axis=-1)
    x2_lo_pre = jnp.min(x2, axis=-1)
    x2_hi_pre = jnp.max(x2, axis=-1)
    parent_area = (x1_hi_pre - x1_lo_pre) * (x2_hi_pre - x2_lo_pre)

    # Compute image AABB areas
    x1 = x_next[..., 0]
    x2 = x_next[..., 1]
    x1_lo_post = jnp.min(x1, axis=-1)
    x1_hi_post = jnp.max(x1, axis=-1)
    x2_lo_post = jnp.min(x2, axis=-1)
    x2_hi_post = jnp.max(x2, axis=-1)
    img_area = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post)

    # Divide image areas by parent areas
    img_area_over_parent = img_area / (parent_area + 1e-12)

    return jnp.sum(img_area_over_parent)


# Compute the proportion of image/parent intersection area to total parent area
def intersection_over_area(
    params,
    *,
    x_domain,
    n1_internal,
    n2_internal,
):

    # Unpack arguments
    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    y1_lo, y1_hi, y2_lo, y2_hi = x_domain

    # Convert gap-params -> actual y-line locations
    y1_params = make_lines_from_gaps(u1, y1_lo, y1_hi)
    y2_params = make_lines_from_gaps(u2, y2_lo, y2_hi)

    # Build all cells' corners: (n1, n2, 4, 2)
    y1_los = y1_params[:-1]
    y1_his = y1_params[1:]
    y2_los = y2_params[:-1]
    y2_his = y2_params[1:]

    n1 = y1_los.shape[0]
    n2 = y2_los.shape[0]
    y1_lo_grid = jnp.broadcast_to(y1_los[:, None], (n1, n2))
    y1_hi_grid = jnp.broadcast_to(y1_his[:, None], (n1, n2))
    y2_lo_grid = jnp.broadcast_to(y2_los[None, :], (n1, n2))
    y2_hi_grid = jnp.broadcast_to(y2_his[None, :], (n1, n2))

    corners = jnp.stack(
        [
            jnp.stack([y1_lo_grid, y2_lo_grid], axis=-1),
            jnp.stack([y1_lo_grid, y2_hi_grid], axis=-1),
            jnp.stack([y1_hi_grid, y2_hi_grid], axis=-1),
            jnp.stack([y1_hi_grid, y2_lo_grid], axis=-1),
        ],
        axis=-2,
    )
    x_next = mc_cl_dynamics_soft_jax(corners)

    # Parent AABB
    x1_pre = corners[..., 0]
    x2_pre = corners[..., 1]
    x1_lo_pre = jnp.min(x1_pre, axis=-1)
    x1_hi_pre = jnp.max(x1_pre, axis=-1)
    x2_lo_pre = jnp.min(x2_pre, axis=-1)
    x2_hi_pre = jnp.max(x2_pre, axis=-1)
    parent_area = (x1_hi_pre - x1_lo_pre) * (x2_hi_pre - x2_lo_pre)

    # Image AABB
    x1_post = x_next[..., 0]
    x2_post = x_next[..., 1]
    x1_lo_post = jnp.min(x1_post, axis=-1)
    x1_hi_post = jnp.max(x1_post, axis=-1)
    x2_lo_post = jnp.min(x2_post, axis=-1)
    x2_hi_post = jnp.max(x2_post, axis=-1)

    # Intersection AABB(parent, image)
    l1 = jnp.maximum(0.0, jnp.minimum(x1_hi_pre, x1_hi_post) - jnp.maximum(x1_lo_pre, x1_lo_post))
    l2 = jnp.maximum(0.0, jnp.minimum(x2_hi_pre, x2_hi_post) - jnp.maximum(x2_lo_pre, x2_lo_post))
    intersection_area = l1 * l2

    # IoA: intersection area / parent area (per cell), summed over all cells
    ioa = intersection_area / (parent_area + 1e-12)
    return jnp.sum(ioa)



# Compute the propotion of image/parent intersection are to image/parent union area
def intersection_over_union(
    params,
    *,
    x_domain,
    n1_internal,
    n2_internal,
):

    # Unpack arguments
    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    y1_lo, y1_hi, y2_lo, y2_hi = x_domain

    # Convert gap-params -> actual y-line locations
    y1_params = make_lines_from_gaps(u1, y1_lo, y1_hi)
    y2_params = make_lines_from_gaps(u2, y2_lo, y2_hi)

    # Build all cells' corners: (n1, n2, 4, 2)
    y1_los = y1_params[:-1]
    y1_his = y1_params[1:]
    y2_los = y2_params[:-1]
    y2_his = y2_params[1:]

    n1 = y1_los.shape[0]
    n2 = y2_los.shape[0]
    y1_lo_grid = jnp.broadcast_to(y1_los[:, None], (n1, n2))
    y1_hi_grid = jnp.broadcast_to(y1_his[:, None], (n1, n2))
    y2_lo_grid = jnp.broadcast_to(y2_los[None, :], (n1, n2))
    y2_hi_grid = jnp.broadcast_to(y2_his[None, :], (n1, n2))

    corners = jnp.stack(
        [
            jnp.stack([y1_lo_grid, y2_lo_grid], axis=-1),
            jnp.stack([y1_lo_grid, y2_hi_grid], axis=-1),
            jnp.stack([y1_hi_grid, y2_hi_grid], axis=-1),
            jnp.stack([y1_hi_grid, y2_lo_grid], axis=-1),
        ],
        axis=-2,
    )
    x_next = mc_cl_dynamics_soft_jax(corners)

    # Parent AABB
    x1_pre = corners[..., 0]
    x2_pre = corners[..., 1]
    x1_lo_pre = jnp.min(x1_pre, axis=-1)
    x1_hi_pre = jnp.max(x1_pre, axis=-1)
    x2_lo_pre = jnp.min(x2_pre, axis=-1)
    x2_hi_pre = jnp.max(x2_pre, axis=-1)
    parent_area = (x1_hi_pre - x1_lo_pre) * (x2_hi_pre - x2_lo_pre)

    # Image AABB
    x1_post = x_next[..., 0]
    x2_post = x_next[..., 1]
    x1_lo_post = jnp.min(x1_post, axis=-1)
    x1_hi_post = jnp.max(x1_post, axis=-1)
    x2_lo_post = jnp.min(x2_post, axis=-1)
    x2_hi_post = jnp.max(x2_post, axis=-1)
    image_area = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post)

    # Intersection AABB(parent, image)
    l1 = jnp.maximum(0.0, jnp.minimum(x1_hi_pre, x1_hi_post) - jnp.maximum(x1_lo_pre, x1_lo_post))
    l2 = jnp.maximum(0.0, jnp.minimum(x2_hi_pre, x2_hi_post) - jnp.maximum(x2_lo_pre, x2_lo_post))
    intersection_area = l1 * l2

    # IoU: intersection / union, where union = parent + image - intersection
    union_area = parent_area + image_area - intersection_area
    iou = intersection_area / (union_area + 1e-12)
    return jnp.sum(iou)




# def objective_function(params, *, y_domain, n1_internal, n2_internal):

#     u1 = params[:n1_internal]
#     u2 = params[n1_internal : n1_internal + n2_internal]
#     theta, a1, a2, h = params[n1_internal + n2_internal :]

#     # Unpack parameters
#     y1_lo, y1_hi, y2_lo, y2_hi = y_domain
#     y1_params = make_lines_from_gaps(u1, y1_lo, y1_hi)
#     y2_params = make_lines_from_gaps(u2, y2_lo, y2_hi)

#     # Initialize
#     nstates_1 = len(y1_params) - 1
#     nstates_2 = len(y2_params) - 1
#     n_states = nstates_1 * nstates_2

#     # Precompute transformation inverse
#     M = np.asarray(M, dtype=float)


#     invM = np.linalg.inv(M)

#     # Loop through each abstract state
#     sum_image_area = 0.0
#     sum_int_over_area = 0.0
#     for i in range(nstates_1):
#         y1_lo, y1_hi = y1_params[i], y1_params[i+1]
#         for j in range(nstates_2):
#             y2_lo, y2_hi = y2_params[j], y2_params[j+1]
#             corners = np.array([
#                 [y1_lo, y2_lo],
#                 [y1_lo, y2_hi],
#                 [y1_hi, y2_hi],
#                 [y1_hi, y2_lo]])

#             # Compute y-space image
#             x_corners = (M @ corners.T).T
#             x_next = dynamics(x_corners)

#             # Compute dimensions
#             x1_lo_pre, x1_hi_pre = float(x_corners[:, 0].min()), float(x_corners[:, 0].max())
#             x2_lo_pre, x2_hi_pre = float(x_corners[:, 1].min()), float(x_corners[:, 1].max())
#             x1_lo_post, x1_hi_post = float(x_next[:, 0].min()), float(x_next[:, 0].max())
#             x2_lo_post, x2_hi_post = float(x_next[:, 1].min()), float(x_next[:, 1].max())

#             # Compute image AABB area
#             img_area = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post)
#             sum_image_area += img_area

#             # Compute IoA
#             l1 = max(0, min(x1_hi_pre, x1_hi_post) - max(x1_lo_pre, x1_lo_post))
#             l2 = max(0, min(x2_hi_pre, x2_hi_post) - max(x2_lo_pre, x2_lo_post))
#             int_area = (l1 * l2)
#             sum_int_over_area += int_area / img_area
            





''' Utility functions for synthetic objective experiments'''

def order_vertices_ccw(verts):
    verts = np.asarray(verts, dtype=float)
    c = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
    return verts[np.argsort(angles)]

# Compute bound in y-space
def get_yspace_bounds(M, x1_min, x1_max, x2_min, x2_max):

    M = np.asarray(M, dtype=float)

    B = np.linalg.inv(M)

    # Four corners of the x-rectangle
    corners_x = np.array([
        [c1, c2]
        for c1, c2 in itertools.product([x1_min, x1_max], [x2_min, x2_max])
    ], dtype=float)

    # Map corners back to y-space: y = M^{-1} x
    verts_y = (B @ corners_x.T).T  # shape (4,2)
    verts_y = order_vertices_ccw(verts_y)

    y1_min, y1_max = float(verts_y[:, 0].min()), float(verts_y[:, 0].max())
    y2_min, y2_max = float(verts_y[:, 1].min()), float(verts_y[:, 1].max())

    bounds = {"y1": (y1_min, y1_max), "y2": (y2_min, y2_max)}
    return bounds, verts_y

def make_lines_from_gaps(u, lo, hi):
    u = jnp.asarray(u)
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)

    # Positive gaps that sum to (hi-lo). Works with unconstrained u.
    gaps = jax.nn.softplus(u)
    total = jnp.sum(gaps)
    gaps = gaps * ((hi - lo) / (total + 1e-12))
    internal = lo + jnp.cumsum(gaps)[:-1]
    return jnp.concatenate([jnp.array([lo]), internal, jnp.array([hi])])

def extract_grid_params(params, n1_internal, n2_internal, x1_min, x1_max, x2_min, x2_max):

    # unpack params (JAX arrays)
    u1 = params[:n1_internal]
    u2 = params[n1_internal:n1_internal+n2_internal]

    # Convert gap-params -> actual y-line locations
    x1_vals = make_lines_from_gaps(u1, x1_min, x1_max)
    x2_vals = make_lines_from_gaps(u2, x2_min, x2_max)

    # Convert to numpy for plotting
    x1_vals = np.array(jax.device_get(x1_vals))
    x2_vals = np.array(jax.device_get(x2_vals))

    return x1_vals, x2_vals



# def extract_grid_params(params, *, n1_internal, n2_internal,
#                           x1_min, x1_max, x2_min, x2_max,
#                           min_gap=0.0):

#     # unpack params (JAX arrays)
#     u1 = params[:n1_internal]
#     u2 = params[n1_internal:n1_internal+n2_internal]
#     theta, a1, a2, h = params[n1_internal+n2_internal:]

#     # Build M (NOTE: this assumes trans_matrix expects log-scales a1,a2 and exponentiates inside)
#     M = trans_matrix(theta, a1, a2, h)

#     # Convert M to numpy for your plotting + get_yspace_bounds
#     M_np = np.array(jax.device_get(M))

#     # Induced Y-domain from current M
#     bounds_y, _ = get_yspace_bounds(
#         M_np,
#         x1_min, x1_max,
#         x2_min, x2_max
#     )
#     y1_lo, y1_hi = bounds_y["y1"]
#     y2_lo, y2_hi = bounds_y["y2"]

#     # Convert gap-params -> actual y-line locations
#     y1_vals = make_lines_from_gaps(u1, y1_lo, y1_hi, min_gap=min_gap)
#     y2_vals = make_lines_from_gaps(u2, y2_lo, y2_hi, min_gap=min_gap)

#     # Convert to numpy for plotting
#     y1_vals_np = np.array(jax.device_get(y1_vals))
#     y2_vals_np = np.array(jax.device_get(y2_vals))

#     return M_np, y1_vals_np, y2_vals_np