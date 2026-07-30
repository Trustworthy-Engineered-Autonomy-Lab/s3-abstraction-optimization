# =====================================================================
# Description: contains all differentiable objective functions used for
# training the parameters of the state abstraction
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import jax
import jax.numpy as jnp
import numpy as np
from synthetic_system import dynamics_jax

XSTAR = np.array([5.0, 5.0])


# =====================================================================
# Objective functions
# =====================================================================


def image_area(
    params,
    *,
    args
    ):
    """
    Sum of post-image AABB areas for each abstract cell.
    """

    n1_internal, n2_internal = args['shape']
    x1_lo, x2_lo = args['domain_lb']
    x1_hi, x2_hi = args['domain_ub']

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]

    # Convert gap-params -> actual y-line locations
    x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
    x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)

    # Build all cells' corners: (n1, n2, 4, 2)
    x1_los = x1_params[:-1]
    x1_his = x1_params[1:]
    x2_los = x2_params[:-1]
    x2_his = x2_params[1:]

    n1 = x1_los.shape[0]
    n2 = x2_los.shape[0]
    x1_lo_grid = jnp.broadcast_to(x1_los[:, None], (n1, n2))
    x1_hi_grid = jnp.broadcast_to(x1_his[:, None], (n1, n2))
    x2_lo_grid = jnp.broadcast_to(x2_los[None, :], (n1, n2))
    x2_hi_grid = jnp.broadcast_to(x2_his[None, :], (n1, n2))

    corners = jnp.stack(
        [
            jnp.stack([x1_lo_grid, x2_lo_grid], axis=-1),
            jnp.stack([x1_lo_grid, x2_hi_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_hi_grid], axis=-1),
            jnp.stack([x1_hi_grid, x2_lo_grid], axis=-1),
        ],
        axis=-2,
    )
    x_next = dynamics_jax(corners, x_star=XSTAR)

    x1 = x_next[..., 0]
    x2 = x_next[..., 1]
    x1_lo_post = jnp.min(x1, axis=-1)
    x1_hi_post = jnp.max(x1, axis=-1)
    x2_lo_post = jnp.min(x2, axis=-1)
    x2_hi_post = jnp.max(x2, axis=-1)
    img_area = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post)
    return jnp.sum(img_area)


def epsilon_1_bound(
    params,
    *,
    args,
    ):
    """
    Upper-bound to epsilon-1 on the value-1 function.
    """

    n1_internal, n2_internal = args['shape']
    x1_lo, x2_lo = args['domain_lb']
    x1_hi, x2_hi = args['domain_ub']

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]

    x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
    x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)

    x1_los = x1_params[:-1]
    x1_his = x1_params[1:]
    x2_los = x2_params[:-1]
    x2_his = x2_params[1:]

    x1_lo_grid, x2_lo_grid = jnp.meshgrid(x1_los, x2_los, indexing="ij")
    x1_hi_grid, x2_hi_grid = jnp.meshgrid(x1_his, x2_his, indexing="ij")

    lower_bounds = jnp.stack([x1_lo_grid, x2_lo_grid], axis=-1)
    upper_bounds = jnp.stack([x1_hi_grid, x2_hi_grid], axis=-1)
    corners = jnp.stack(
        [
            lower_bounds,
            jnp.stack([x1_lo_grid, x2_hi_grid], axis=-1),
            upper_bounds,
            jnp.stack([x1_hi_grid, x2_lo_grid], axis=-1),
        ],
        axis=-2,
    )

    centroids = 0.5 * (lower_bounds + upper_bounds)
    next_corners = dynamics_jax(corners, x_star=XSTAR)
    next_lower_bounds = jnp.min(next_corners, axis=-2)
    next_upper_bounds = jnp.max(next_corners, axis=-2)
    reachable_centroids = 0.5 * (next_lower_bounds + next_upper_bounds)

    radii = 0.5 * jnp.linalg.norm(
        next_upper_bounds - next_lower_bounds,
        axis=-1,
    )
    center_offsets = (
        dynamics_jax(centroids, x_star=XSTAR) - reachable_centroids
    )
    norm_epsilon = jnp.asarray(1e-12, dtype=params.dtype)
    center_differences = jnp.sqrt(
        jnp.sum(jnp.square(center_offsets), axis=-1) + norm_epsilon
    )
    return jnp.sum(radii + center_differences)


def epsilon_H_bound(
    params,
    *,
    args,
    ):
    """
    Sum a smooth finite-horizon epsilon bound over all abstract cells.

    The center of each cell is used as the concrete witness. Its enclosing
    AABB is propagated through the dynamics and inflated after every step.
    A temperature-scaled log-sum-exp smoothly aggregates the per-step bounds.
    """

    horizon = args['horizon']
    temp = args['temp']
    inflation_coef = args['inflation_coef']

    n1_internal, n2_internal = args['shape']
    x1_lo, x2_lo = args['domain_lb']
    x1_hi, x2_hi = args['domain_ub']

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]

    x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
    x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)

    x1_los = x1_params[:-1]
    x1_his = x1_params[1:]
    x2_los = x2_params[:-1]
    x2_his = x2_params[1:]

    x1_lo_grid, x2_lo_grid = jnp.meshgrid(x1_los, x2_los, indexing="ij")
    x1_hi_grid, x2_hi_grid = jnp.meshgrid(x1_his, x2_his, indexing="ij")
    lower_bounds = jnp.stack([x1_lo_grid, x2_lo_grid], axis=-1)
    upper_bounds = jnp.stack([x1_hi_grid, x2_hi_grid], axis=-1)

    def aabb_corners(lower, upper):
        return jnp.stack(
            [
                lower,
                jnp.stack([lower[..., 0], upper[..., 1]], axis=-1),
                upper,
                jnp.stack([upper[..., 0], lower[..., 1]], axis=-1),
            ],
            axis=-2,
        )

    witnesses = 0.5 * (lower_bounds + upper_bounds)
    corners = aabb_corners(lower_bounds, upper_bounds)
    inflation_coef = jnp.asarray(inflation_coef, dtype=params.dtype)
    temp = jnp.asarray(temp, dtype=params.dtype)
    norm_epsilon = jnp.asarray(1e-12, dtype=params.dtype)

    def rollout_step(carry, _):
        witness, current_corners = carry
        witness = dynamics_jax(witness, x_star=XSTAR)
        next_corners = dynamics_jax(current_corners, x_star=XSTAR)
        next_lower_bounds = jnp.min(next_corners, axis=-2)
        next_upper_bounds = jnp.max(next_corners, axis=-2)

        reach_centroids = 0.5 * (next_lower_bounds + next_upper_bounds)
        reach_radii = 0.5 * jnp.linalg.norm(
            next_upper_bounds - next_lower_bounds,
            axis=-1,
        )
        witness_offsets = witness - reach_centroids
        witness_differences = jnp.sqrt(
            jnp.sum(jnp.square(witness_offsets), axis=-1) + norm_epsilon
        )
        step_bounds = reach_radii + witness_differences

        inflated_lower_bounds = next_lower_bounds - inflation_coef
        inflated_upper_bounds = next_upper_bounds + inflation_coef
        inflated_corners = aabb_corners(
            inflated_lower_bounds,
            inflated_upper_bounds,
        )
        return (witness, inflated_corners), step_bounds

    _, step_bounds = jax.lax.scan(
        rollout_step,
        (witnesses, corners),
        xs=None,
        length=horizon,
    )
    per_cell_bounds = temp * jax.scipy.special.logsumexp(
        step_bounds / temp,
        axis=0,
    )
    return jnp.sum(per_cell_bounds)



def upward_proxy(
    params,
    *,
    args,
    ):
    """
    LSE over a smooth finite-horizon epsilon bound.
    """

    horizon = args['horizon']
    temp_in = args['temp_in']
    temp_out = args['temp_out']
    inflation_coefs = args['inflation_coefs']

    n1_internal, n2_internal = args['shape']
    x1_lo, x2_lo = args['domain_lb']
    x1_hi, x2_hi = args['domain_ub']

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]

    x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
    x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)

    x1_los = x1_params[:-1]
    x1_his = x1_params[1:]
    x2_los = x2_params[:-1]
    x2_his = x2_params[1:]

    x1_lo_grid, x2_lo_grid = jnp.meshgrid(x1_los, x2_los, indexing="ij")
    x1_hi_grid, x2_hi_grid = jnp.meshgrid(x1_his, x2_his, indexing="ij")
    lower_bounds = jnp.stack([x1_lo_grid, x2_lo_grid], axis=-1)
    upper_bounds = jnp.stack([x1_hi_grid, x2_hi_grid], axis=-1)

    def aabb_corners(lower, upper):
        return jnp.stack(
            [
                lower,
                jnp.stack([lower[..., 0], upper[..., 1]], axis=-1),
                upper,
                jnp.stack([upper[..., 0], lower[..., 1]], axis=-1),
            ],
            axis=-2,
        )

    witnesses = 0.5 * (lower_bounds + upper_bounds)
    corners = aabb_corners(lower_bounds, upper_bounds)
    inflation_coefs = jnp.asarray(inflation_coefs, dtype=params.dtype)
    temp_in = jnp.asarray(temp_in, dtype=params.dtype)
    temp_out = jnp.asarray(temp_out, dtype=params.dtype)
    norm_epsilon = jnp.asarray(1e-12, dtype=params.dtype)

    def rollout_step(carry, _):
        witness, current_corners = carry
        witness = dynamics_jax(witness, x_star=XSTAR)
        next_corners = dynamics_jax(current_corners, x_star=XSTAR)
        next_lower_bounds = jnp.min(next_corners, axis=-2)
        next_upper_bounds = jnp.max(next_corners, axis=-2)

        reach_centroids = 0.5 * (next_lower_bounds + next_upper_bounds)
        reach_radii = 0.5 * jnp.linalg.norm(
            next_upper_bounds - next_lower_bounds,
            axis=-1,
        )
        witness_offsets = witness - reach_centroids
        witness_differences = jnp.sqrt(
            jnp.sum(jnp.square(witness_offsets), axis=-1) + norm_epsilon
        )
        step_bounds = reach_radii + witness_differences

        inflated_lower_bounds = next_lower_bounds - inflation_coefs
        inflated_upper_bounds = next_upper_bounds + inflation_coefs
        inflated_corners = aabb_corners(
            inflated_lower_bounds,
            inflated_upper_bounds,
        )
        return (witness, inflated_corners), step_bounds

    _, step_bounds = jax.lax.scan(
        rollout_step,
        (witnesses, corners),
        xs=None,
        length=horizon,
    )
    per_cell_bounds = temp_in * jax.scipy.special.logsumexp(
        step_bounds / temp_in,
        axis=0,
    )
    overall_bound = temp_out * jax.scipy.special.logsumexp(per_cell_bounds / temp_out)

    return overall_bound


# =====================================================================
# JAX-compatible helper methods
# =====================================================================

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

    n1_internal, n2_internal = shape
    x1_min, x2_min = domain_lb
    x1_max, x2_max = domain_ub

    # unpack params (JAX arrays)
    u1 = params[:n1_internal]
    u2 = params[n1_internal:n1_internal+n2_internal]

    # Convert gap-params to actual edge locations
    x1_vals = make_lines_from_gaps(u1, x1_min, x1_max)
    x2_vals = make_lines_from_gaps(u2, x2_min, x2_max)

    # Convert to numpy for plotting
    x1_vals = np.array(jax.device_get(x1_vals))
    x2_vals = np.array(jax.device_get(x2_vals))

    return x1_vals, x2_vals
