# Libraries
import jax
import jax.numpy as jnp
import numpy as np
import itertools

''' Fast Dynamics '''


''' JAX-compatible Dynamics '''

def wrap_to_pi(angle):
    angle = jnp.asarray(angle)
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

def unicycle_dynamics(state, control, control_bound = np.pi/4):

    state = jnp.asarray(state)
    control = jnp.asarray(control)

    # Unicycle model parameters
    delta_t = jnp.asarray(0.5, dtype=state.dtype)
    velocity = jnp.asarray(5.0, dtype=state.dtype)
    control_bound = jnp.asarray(control_bound, dtype=state.dtype)
    pose_x, pose_y, theta = state

    # Apply control bounds (heading rate of change)
    control = jnp.clip(control, -control_bound, control_bound)

    # Update the state
    next_pose_x = pose_x + (delta_t * velocity * jnp.cos(theta))
    next_pose_y = pose_y + (delta_t * velocity * jnp.sin(theta))
    next_theta = wrap_to_pi(theta + (delta_t * control))  # normalize angle to [-pi, pi]

    return jnp.stack([next_pose_x, next_pose_y, next_theta])

def state_controller(
    state,
    *,
    goal_center,
    obstacle_centers,
    obstacle_radii,
    # gains / shaping
    k_goal=1.0,
    k_rep=8.0,
    alpha=0.6,
    # heading control
    k_theta=2.5,
    omega_max=np.pi/4,
    # numerical smoothing
    eps=1e-6,
):
    """
    Deterministic smooth controller for Dubins/unicycle:
      1) Build desired planar direction v(p) = v_att + v_rep
      2) Convert to desired heading theta_d
      3) Apply smooth saturated turn rate: omega = omega_max * tanh(k_theta * e_theta)
    """

    state = jnp.asarray(state)
    goal_center = jnp.asarray(goal_center, dtype=state.dtype)
    obstacle_centers = jnp.asarray(obstacle_centers, dtype=state.dtype)
    obstacle_radii = jnp.asarray(obstacle_radii, dtype=state.dtype)

    k_goal = jnp.asarray(k_goal, dtype=state.dtype)
    k_rep = jnp.asarray(k_rep, dtype=state.dtype)
    alpha = jnp.asarray(alpha, dtype=state.dtype)
    k_theta = jnp.asarray(k_theta, dtype=state.dtype)
    omega_max = jnp.asarray(omega_max, dtype=state.dtype)
    eps = jnp.asarray(eps, dtype=state.dtype)

    px, py, theta = state
    p = jnp.stack([px, py])

    # Attractive component (toward goal)
    v_att = k_goal * (goal_center - p)

    # Repulsive component (sum over discs)
    # obstacle_centers: [N, 2], obstacle_radii: [N]
    diff = p[None, :] - obstacle_centers
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + eps)  # smoothed distance to center
    clearance = dist - obstacle_radii

    # Smooth activation: stronger when near obstacle, decays with clearance
    w = jnp.exp(-alpha * clearance)

    # Direction away from obstacle with smoothing in denom
    denom = (dist**3 + eps)[:, None]
    v_rep = jnp.sum((k_rep * w)[:, None] * diff / denom, axis=0)

    v = v_att + v_rep
    v_norm = jnp.linalg.norm(v)

    def _omega_when_ok(_):
        theta_d = jnp.arctan2(v[1], v[0])
        e_theta = wrap_to_pi(theta_d - theta)
        return omega_max * jnp.tanh(k_theta * e_theta)

    omega = jax.lax.cond(v_norm < jnp.asarray(1e-9, dtype=state.dtype), lambda _: jnp.asarray(0.0, dtype=state.dtype), _omega_when_ok, operand=None)
    return omega

# Unicylce system with controller in the loop
def cl_unicycle_dynamics(state):
    state = jnp.asarray(state)

    obs_center = jnp.array([25.0, 25.0], dtype=state.dtype)
    obs_radius = jnp.asarray(5.0, dtype=state.dtype)
    goal_center = jnp.array([40.0, 20.0], dtype=state.dtype)

    control_input = state_controller(
        state,
        goal_center=goal_center,
        obstacle_centers=jnp.stack([obs_center], axis=0),
        obstacle_radii=jnp.stack([obs_radius], axis=0),
        k_goal=0.3,
        k_rep=300.0,
        alpha=0.1,
        k_theta=2.0,
        omega_max=jnp.pi / 4,
    )
    return unicycle_dynamics(state, control_input)


''' Objective functions and helpers '''

# Compute sum of image AABB volume
def image_volume(
    params,
    *,
    domain,
    n1_internal,
    n2_internal,
    n3_internal,
):
    """Sum of post-image AABB volume for each abstract cell.

    JAX-compatible and differentiable w.r.t. `params`.
    `n1_internal`, `n2_internal`, `n3_internal` are treated as static (Python ints).
    """
    
    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    u3 = params[n1_internal + n2_internal : n1_internal + n2_internal + n3_internal]

    x1_lo, x1_hi, x2_lo, x2_hi, x3_lo, x3_hi = domain

    # Convert gap-params -> actual y-line locations
    x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
    x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)
    x3_params = make_lines_from_gaps(u3, x3_lo, x3_hi)

    # Build all cells' corners: (n1, n2, 4, 2)
    x1_los = x1_params[:-1]
    x1_his = x1_params[1:]
    x2_los = x2_params[:-1]
    x2_his = x2_params[1:]
    x3_los = x3_params[:-1]
    x3_his = x3_params[1:]

    n1 = x1_los.shape[0]
    n2 = x2_los.shape[0]
    n3 = x3_los.shape[0]
    x1_lo_grid = jnp.broadcast_to(x1_los[:, None, None], (n1, n2, n3))
    x1_hi_grid = jnp.broadcast_to(x1_his[:, None, None], (n1, n2, n3))
    x2_lo_grid = jnp.broadcast_to(x2_los[None, :, None], (n1, n2, n3))
    x2_hi_grid = jnp.broadcast_to(x2_his[None, :, None], (n1, n2, n3))
    x3_lo_grid = jnp.broadcast_to(x3_los[None, None, :], (n1, n2, n3))
    x3_hi_grid = jnp.broadcast_to(x3_his[None, None, :], (n1, n2, n3))

    # All 8 corners per cell: (n1, n2, n3, 8, 3)
    corners = jnp.stack(
        [
            jnp.stack([x1_lo_grid, x2_lo_grid, x3_lo_grid], axis=-1),
            jnp.stack([x1_lo_grid, x2_lo_grid, x3_hi_grid], axis=-1),
            jnp.stack([x1_lo_grid, x2_hi_grid, x3_lo_grid], axis=-1),
            jnp.stack([x1_lo_grid, x2_hi_grid, x3_hi_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_lo_grid, x3_lo_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_lo_grid, x3_hi_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_hi_grid, x3_lo_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_hi_grid, x3_hi_grid], axis=-1),
        ],
        axis=-2,
    )

    flat_corners = corners.reshape((-1, 3))
    flat_next = jax.vmap(cl_unicycle_dynamics)(flat_corners)
    x_next = flat_next.reshape(corners.shape)

    x1 = x_next[..., 0]
    x2 = x_next[..., 1]
    x3 = x_next[..., 2]
    x1_lo_post = jnp.min(x1, axis=-1)
    x1_hi_post = jnp.max(x1, axis=-1)
    x2_lo_post = jnp.min(x2, axis=-1)
    x2_hi_post = jnp.max(x2, axis=-1)
    x3_lo_post = jnp.min(x3, axis=-1)
    x3_hi_post = jnp.max(x3, axis=-1)
    img_volume = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post) * (x3_hi_post - x3_lo_post)

    return jnp.sum(img_volume)

def image_volume_over_parent(
    params,
    *,
    domain,
    n1_internal,
    n2_internal,
    n3_internal,
):
    """Sum over cells of (image AABB volume / parent AABB volume).

    JAX-compatible and differentiable w.r.t. `params`.
    """

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    u3 = params[n1_internal + n2_internal : n1_internal + n2_internal + n3_internal]

    x1_lo, x1_hi, x2_lo, x2_hi, x3_lo, x3_hi = domain

    # Convert gap-params -> actual grid line locations
    x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
    x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)
    x3_params = make_lines_from_gaps(u3, x3_lo, x3_hi)

    # Build all cells' corners: (n1, n2, n3, 8, 3)
    x1_los = x1_params[:-1]
    x1_his = x1_params[1:]
    x2_los = x2_params[:-1]
    x2_his = x2_params[1:]
    x3_los = x3_params[:-1]
    x3_his = x3_params[1:]

    n1 = x1_los.shape[0]
    n2 = x2_los.shape[0]
    n3 = x3_los.shape[0]

    x1_lo_grid = jnp.broadcast_to(x1_los[:, None, None], (n1, n2, n3))
    x1_hi_grid = jnp.broadcast_to(x1_his[:, None, None], (n1, n2, n3))
    x2_lo_grid = jnp.broadcast_to(x2_los[None, :, None], (n1, n2, n3))
    x2_hi_grid = jnp.broadcast_to(x2_his[None, :, None], (n1, n2, n3))
    x3_lo_grid = jnp.broadcast_to(x3_los[None, None, :], (n1, n2, n3))
    x3_hi_grid = jnp.broadcast_to(x3_his[None, None, :], (n1, n2, n3))

    corners = jnp.stack(
        [
            jnp.stack([x1_lo_grid, x2_lo_grid, x3_lo_grid], axis=-1),
            jnp.stack([x1_lo_grid, x2_lo_grid, x3_hi_grid], axis=-1),
            jnp.stack([x1_lo_grid, x2_hi_grid, x3_lo_grid], axis=-1),
            jnp.stack([x1_lo_grid, x2_hi_grid, x3_hi_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_lo_grid, x3_lo_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_lo_grid, x3_hi_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_hi_grid, x3_lo_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_hi_grid, x3_hi_grid], axis=-1),
        ],
        axis=-2,
    )

    # Push corners through closed-loop dynamics
    flat_corners = corners.reshape((-1, 3))
    flat_next = jax.vmap(cl_unicycle_dynamics)(flat_corners)
    x_next = flat_next.reshape(corners.shape)

    # Parent AABB volumes (for axis-aligned cubes this equals cell volume)
    parent_volume = (x1_hi_grid - x1_lo_grid) * (x2_hi_grid - x2_lo_grid) * (x3_hi_grid - x3_lo_grid)

    # Image AABB volumes
    x1_post = x_next[..., 0]
    x2_post = x_next[..., 1]
    x3_post = x_next[..., 2]
    x1_lo_post = jnp.min(x1_post, axis=-1)
    x1_hi_post = jnp.max(x1_post, axis=-1)
    x2_lo_post = jnp.min(x2_post, axis=-1)
    x2_hi_post = jnp.max(x2_post, axis=-1)
    x3_lo_post = jnp.min(x3_post, axis=-1)
    x3_hi_post = jnp.max(x3_post, axis=-1)
    img_volume = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post) * (x3_hi_post - x3_lo_post)

    ratio = img_volume / (parent_volume + 1e-12)
    return jnp.sum(ratio)


# # Compute the proportion of image/parent intersection area to total parent area
# def intersection_over_area(
#     params,
#     *,
#     x_domain,
#     n1_internal,
#     n2_internal,
# ):

#     # Unpack arguments
#     params = jnp.asarray(params)
#     u1 = params[:n1_internal]
#     u2 = params[n1_internal : n1_internal + n2_internal]
#     y1_lo, y1_hi, y2_lo, y2_hi = x_domain

#     # Convert gap-params -> actual y-line locations
#     y1_params = make_lines_from_gaps(u1, y1_lo, y1_hi)
#     y2_params = make_lines_from_gaps(u2, y2_lo, y2_hi)

#     # Build all cells' corners: (n1, n2, 4, 2)
#     y1_los = y1_params[:-1]
#     y1_his = y1_params[1:]
#     y2_los = y2_params[:-1]
#     y2_his = y2_params[1:]

#     n1 = y1_los.shape[0]
#     n2 = y2_los.shape[0]
#     y1_lo_grid = jnp.broadcast_to(y1_los[:, None], (n1, n2))
#     y1_hi_grid = jnp.broadcast_to(y1_his[:, None], (n1, n2))
#     y2_lo_grid = jnp.broadcast_to(y2_los[None, :], (n1, n2))
#     y2_hi_grid = jnp.broadcast_to(y2_his[None, :], (n1, n2))

#     corners = jnp.stack(
#         [
#             jnp.stack([y1_lo_grid, y2_lo_grid], axis=-1),
#             jnp.stack([y1_lo_grid, y2_hi_grid], axis=-1),
#             jnp.stack([y1_hi_grid, y2_hi_grid], axis=-1),
#             jnp.stack([y1_hi_grid, y2_lo_grid], axis=-1),
#         ],
#         axis=-2,
#     )
#     x_next = dynamics(corners)

#     # Parent AABB
#     x1_pre = corners[..., 0]
#     x2_pre = corners[..., 1]
#     x1_lo_pre = jnp.min(x1_pre, axis=-1)
#     x1_hi_pre = jnp.max(x1_pre, axis=-1)
#     x2_lo_pre = jnp.min(x2_pre, axis=-1)
#     x2_hi_pre = jnp.max(x2_pre, axis=-1)
#     parent_area = (x1_hi_pre - x1_lo_pre) * (x2_hi_pre - x2_lo_pre)

#     # Image AABB
#     x1_post = x_next[..., 0]
#     x2_post = x_next[..., 1]
#     x1_lo_post = jnp.min(x1_post, axis=-1)
#     x1_hi_post = jnp.max(x1_post, axis=-1)
#     x2_lo_post = jnp.min(x2_post, axis=-1)
#     x2_hi_post = jnp.max(x2_post, axis=-1)

#     # Intersection AABB(parent, image)
#     l1 = jnp.maximum(0.0, jnp.minimum(x1_hi_pre, x1_hi_post) - jnp.maximum(x1_lo_pre, x1_lo_post))
#     l2 = jnp.maximum(0.0, jnp.minimum(x2_hi_pre, x2_hi_post) - jnp.maximum(x2_lo_pre, x2_lo_post))
#     intersection_area = l1 * l2

#     # IoA: intersection area / parent area (per cell), summed over all cells
#     ioa = intersection_area / (parent_area + 1e-12)
#     return jnp.sum(ioa)



# # Compute the propotion of image/parent intersection are to image/parent union area
# def intersection_over_union(
#     params,
#     *,
#     x_domain,
#     n1_internal,
#     n2_internal,
# ):

#     # Unpack arguments
#     params = jnp.asarray(params)
#     u1 = params[:n1_internal]
#     u2 = params[n1_internal : n1_internal + n2_internal]
#     y1_lo, y1_hi, y2_lo, y2_hi = x_domain

#     # Convert gap-params -> actual y-line locations
#     y1_params = make_lines_from_gaps(u1, y1_lo, y1_hi)
#     y2_params = make_lines_from_gaps(u2, y2_lo, y2_hi)

#     # Build all cells' corners: (n1, n2, 4, 2)
#     y1_los = y1_params[:-1]
#     y1_his = y1_params[1:]
#     y2_los = y2_params[:-1]
#     y2_his = y2_params[1:]

#     n1 = y1_los.shape[0]
#     n2 = y2_los.shape[0]
#     y1_lo_grid = jnp.broadcast_to(y1_los[:, None], (n1, n2))
#     y1_hi_grid = jnp.broadcast_to(y1_his[:, None], (n1, n2))
#     y2_lo_grid = jnp.broadcast_to(y2_los[None, :], (n1, n2))
#     y2_hi_grid = jnp.broadcast_to(y2_his[None, :], (n1, n2))

#     corners = jnp.stack(
#         [
#             jnp.stack([y1_lo_grid, y2_lo_grid], axis=-1),
#             jnp.stack([y1_lo_grid, y2_hi_grid], axis=-1),
#             jnp.stack([y1_hi_grid, y2_hi_grid], axis=-1),
#             jnp.stack([y1_hi_grid, y2_lo_grid], axis=-1),
#         ],
#         axis=-2,
#     )
#     x_next = dynamics(corners)

#     # Parent AABB
#     x1_pre = corners[..., 0]
#     x2_pre = corners[..., 1]
#     x1_lo_pre = jnp.min(x1_pre, axis=-1)
#     x1_hi_pre = jnp.max(x1_pre, axis=-1)
#     x2_lo_pre = jnp.min(x2_pre, axis=-1)
#     x2_hi_pre = jnp.max(x2_pre, axis=-1)
#     parent_area = (x1_hi_pre - x1_lo_pre) * (x2_hi_pre - x2_lo_pre)

#     # Image AABB
#     x1_post = x_next[..., 0]
#     x2_post = x_next[..., 1]
#     x1_lo_post = jnp.min(x1_post, axis=-1)
#     x1_hi_post = jnp.max(x1_post, axis=-1)
#     x2_lo_post = jnp.min(x2_post, axis=-1)
#     x2_hi_post = jnp.max(x2_post, axis=-1)
#     image_area = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post)

#     # Intersection AABB(parent, image)
#     l1 = jnp.maximum(0.0, jnp.minimum(x1_hi_pre, x1_hi_post) - jnp.maximum(x1_lo_pre, x1_lo_post))
#     l2 = jnp.maximum(0.0, jnp.minimum(x2_hi_pre, x2_hi_post) - jnp.maximum(x2_lo_pre, x2_lo_post))
#     intersection_area = l1 * l2

#     # IoU: intersection / union, where union = parent + image - intersection
#     union_area = parent_area + image_area - intersection_area
#     iou = intersection_area / (union_area + 1e-12)
#     return jnp.sum(iou)


''' Utility functions for synthetic objective experiments'''

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

def extract_grid_params(params, n1_internal, n2_internal, n3_internal, domain):

    x1_min, x1_max, x2_min, x2_max, x3_min, x3_max = domain

    # unpack params (JAX arrays)
    u1 = params[:n1_internal]
    u2 = params[n1_internal:n1_internal+n2_internal]
    u3 = params[n1_internal+n2_internal:n1_internal+n2_internal+n3_internal]

    # Convert gap-params -> actual y-line locations
    x1_vals = make_lines_from_gaps(u1, x1_min, x1_max)
    x2_vals = make_lines_from_gaps(u2, x2_min, x2_max)
    x3_vals = make_lines_from_gaps(u3, x3_min, x3_max)

    # Convert to numpy for plotting
    x1_vals = np.array(jax.device_get(x1_vals))
    x2_vals = np.array(jax.device_get(x2_vals))
    x3_vals = np.array(jax.device_get(x3_vals))

    return x1_vals, x2_vals, x3_vals
