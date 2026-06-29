# =====================================================================
# Description: contains all differentiable objective functions used for
# training the parameters of the state abstraction
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import jax
import jax.numpy as jnp
import numpy as np
from unicycle_system import cl_system_jax, wrap_to_pi_jax
import unicycle_system_jax as usj
import itertools

# =====================================================================
# Objective functions
# =====================================================================

def succ_estimate(params,
                *,
                shape,
                domain_lb,
                domain_ub,
                ):
    
    L = usj.estimate_lipschitz_array(domain_lb, domain_ub)
    A = np.eye(3) + L

    n1_internal, n2_internal, n3_internal = shape
    x1_lo, x2_lo, x3_lo = domain_lb
    x1_hi, x2_hi, x3_hi = domain_ub

    gap1 = params[0]
    gap2 = params[1]
    gap3 = params[2]

def image_volume(
    params,
    *,
    args
    ):
    """
    Sum of post-image AABB volume for each abstract cell.
    """

    n1_internal, n2_internal, n3_internal = args['shape']
    x1_lo, x2_lo, x3_lo = args['domain_lb']
    x1_hi, x2_hi, x3_hi = args['domain_ub']
    
    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    u3 = params[n1_internal + n2_internal : n1_internal + n2_internal + n3_internal]

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
    flat_next = jax.vmap(cl_system_jax)(flat_corners)
    x_next = flat_next.reshape(corners.shape)

    x1 = x_next[..., 0]
    x2 = x_next[..., 1]
    x3 = x_next[..., 2]
    x1_lo_post = jnp.min(x1, axis=-1)
    x1_hi_post = jnp.max(x1, axis=-1)
    x2_lo_post = jnp.min(x2, axis=-1)
    x2_hi_post = jnp.max(x2, axis=-1)
    theta_span = theta_min_arc_length(x3)
    img_volume = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post) * theta_span

    return jnp.sum(img_volume)

def image_volume_over_parent(
    params,
    *,
    args
    ):
    """Sum over cells of (image AABB volume / parent AABB volume).
    """

    n1_internal, n2_internal, n3_internal = args['shape']
    x1_lo, x2_lo, x3_lo = args['domain_lb']
    x1_hi, x2_hi, x3_hi = args['domain_ub']

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    u3 = params[n1_internal + n2_internal : n1_internal + n2_internal + n3_internal]

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
    flat_next = jax.vmap(cl_system_jax)(flat_corners)
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
    theta_span = theta_min_arc_length(x3_post)
    img_volume = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post) * theta_span

    ratio = img_volume / (parent_volume + 1e-12)
    return jnp.sum(ratio)

def succ_bound(
    params,
    *,
    args,
    p=10.0 # soft min smoothing factor
    ):
    """Derived, smooth upper bound to successor count.
    """
    
    n1_internal, n2_internal, n3_internal = args['shape']
    x1_lo, x2_lo, x3_lo = args['domain_lb']
    x1_hi, x2_hi, x3_hi = args['domain_ub']
    L = args['L']

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    u3 = params[n1_internal + n2_internal : n1_internal + n2_internal + n3_internal]

    x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
    x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)
    x3_params = make_lines_from_gaps(u3, x3_lo, x3_hi)

    gap1 = jnp.diff(x1_params)
    gap2 = jnp.diff(x2_params)
    gap3 = jnp.diff(x3_params)

    L = jnp.asarray(L)
    p = jnp.asarray(p, dtype=params.dtype)
    eps = jnp.asarray(1e-12, dtype=params.dtype)

    def soft_min_gap(gaps):
        return jnp.power(jnp.sum(jnp.power(gaps + eps, -p)), -1.0 / p)

    eta = jnp.stack([
        soft_min_gap(gap1),
        soft_min_gap(gap2),
        soft_min_gap(gap3),
    ])

    gap_grid = jnp.stack(
        jnp.meshgrid(gap1, gap2, gap3, indexing="ij"),
        axis=-1,
    )
    diam = jnp.einsum("...j,kj->...k", gap_grid, L)
    prod = jnp.prod(2.0 + diam / (eta + eps), axis=-1)

    return jnp.mean(prod)

def inflated_image_volume(
    params,
    *,
    shape,
    domain_lb,
    domain_ub,
    ):
    """
    Sum of post-image AABB volume inflated with Lagrange bound for each abstract cell.
    """

    n1_internal, n2_internal, n3_internal = shape
    x1_lo, x2_lo, x3_lo = domain_lb
    x1_hi, x2_hi, x3_hi = domain_ub
    
    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]
    u3 = params[n1_internal + n2_internal : n1_internal + n2_internal + n3_internal]

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
    flat_next = jax.vmap(cl_system_jax)(flat_corners)
    x_next = flat_next.reshape(corners.shape)

    x1 = x_next[..., 0]
    x2 = x_next[..., 1]
    x3 = x_next[..., 2]
    x1_lo_post = jnp.min(x1, axis=-1)
    x1_hi_post = jnp.max(x1, axis=-1)
    x2_lo_post = jnp.min(x2, axis=-1)
    x2_hi_post = jnp.max(x2, axis=-1)
    theta_span = theta_min_arc_length(x3)
    img_volume = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post) * theta_span

    return jnp.sum(img_volume)


# =====================================================================
# JAX-compatible helper methods
# =====================================================================

def theta_min_arc_length(thetas, *, eps=1e-12):
    """
    Length of the minimal circular arc covering thetas.
    """

    th = wrap_to_pi_jax(thetas)
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=th.dtype)
    eps = jnp.asarray(eps, dtype=th.dtype)

    # Map to [0, 2pi) and sort for stable circular gap computation.
    u = jnp.sort(th + jnp.pi, axis=-1)
    n = u.shape[-1]

    # Largest gap between consecutive points on the circle.
    u_ext = jnp.concatenate([u, u[..., :1] + two_pi], axis=-1)
    gaps = jnp.diff(u_ext, axis=-1)
    k = jnp.argmax(gaps, axis=-1)

    start_idx = (k + 1) % n
    end_idx = k
    start_u = jnp.take_along_axis(u, start_idx[..., None], axis=-1)[..., 0]
    end_u = jnp.take_along_axis(u, end_idx[..., None], axis=-1)[..., 0]

    # Complement of the largest gap.
    arc_len = jnp.mod(end_u - start_u, two_pi)
    arc_len = jnp.where(arc_len >= two_pi - eps, two_pi, arc_len)
    return arc_len

def make_lines_from_gaps(u, lo, hi):
    """
    Converts gap parameters to explicit hyperplane locations.
    """
    u = jnp.asarray(u)
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)

    # Positive gaps that sum to (hi-lo). Works with unconstrained u.
    gaps = jax.nn.softplus(u)
    total = jnp.sum(gaps)
    gaps = gaps * ((hi - lo) / (total + 1e-12))
    internal = lo + jnp.cumsum(gaps)[:-1]
    return jnp.concatenate([jnp.array([lo]), internal, jnp.array([hi])])

def extract_grid_params(params, shape, domain_lb, domain_ub):

    n1_internal, n2_internal, n3_internal = shape
    x1_min, x2_min, x3_min = domain_lb
    x1_max, x2_max, x3_max = domain_ub

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
