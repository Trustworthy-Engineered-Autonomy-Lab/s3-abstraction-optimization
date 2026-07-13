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

_cl_system_hessian = jax.hessian(cl_system_jax)
_cl_system_jacobian = jax.jacfwd(cl_system_jax)
_batched_cl_system_hessian = jax.vmap(_cl_system_hessian)
_batched_cl_system_jacobian = jax.vmap(_cl_system_jacobian)
_corner_signs = jnp.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [-1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0, -1.0],
        [ 1.0,  1.0,  1.0],
    ],
    dtype=jnp.float32,
)

# =====================================================================
# Objective functions
# =====================================================================

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


def taylor_remainder(
    params,
    *,
    args,
    batch_size=4096,
    ):
    """
    Sum the per-cell Taylor remainder proxy with batched JAX Hessians.
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

    x1_widths = jnp.diff(x1_params)
    x2_widths = jnp.diff(x2_params)
    x3_widths = jnp.diff(x3_params)

    x1_centers = 0.5 * (x1_params[:-1] + x1_params[1:])
    x2_centers = 0.5 * (x2_params[:-1] + x2_params[1:])
    x3_centers = 0.5 * (x3_params[:-1] + x3_params[1:])

    centroids = jnp.stack(
        jnp.meshgrid(x1_centers, x2_centers, x3_centers, indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))
    half_spans = 0.5 * jnp.stack(
        jnp.meshgrid(x1_widths, x2_widths, x3_widths, indexing="ij"),
        axis=-1,
    )
    max_displacements = jnp.linalg.norm(half_spans, axis=-1).reshape((-1,))

    batch_size = int(batch_size)
    num_cells = centroids.shape[0]
    pad = (-num_cells) % batch_size
    padded_num_cells = num_cells + pad

    centroids = jnp.pad(centroids, ((0, pad), (0, 0)))
    max_displacements = jnp.pad(max_displacements, (0, pad))
    valid_mask = jnp.arange(padded_num_cells) < num_cells

    centroid_batches = centroids.reshape((-1, batch_size, 3))
    displacement_batches = max_displacements.reshape((-1, batch_size))
    mask_batches = valid_mask.reshape((-1, batch_size))

    def accumulate_batch(total, batch):
        centroid_batch, displacement_batch, mask_batch = batch
        hessians = _batched_cl_system_hessian(centroid_batch)
        eigvals = jnp.linalg.eigvalsh(hessians)
        spec_norms = jnp.max(jnp.abs(eigvals), axis=-1)
        spec_norms = jnp.max(spec_norms, axis=-1)
        contributions = 0.5 * spec_norms * displacement_batch
        contributions = jnp.where(mask_batch, contributions, jnp.zeros_like(contributions))
        return total + jnp.sum(contributions), None

    total_taylor_remainder, _ = jax.lax.scan(
        accumulate_batch,
        jnp.asarray(0.0, dtype=params.dtype),
        (centroid_batches, displacement_batches, mask_batches),
    )

    return total_taylor_remainder


def noninflated_image_volume(
    params,
    *,
    args,
    batch_size=4096,
    ):
    """
    Sum of all linearized AABB image volumes pre-inflation.
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

    x1_widths = jnp.diff(x1_params)
    x2_widths = jnp.diff(x2_params)
    x3_widths = jnp.diff(x3_params)

    x1_centers = 0.5 * (x1_params[:-1] + x1_params[1:])
    x2_centers = 0.5 * (x2_params[:-1] + x2_params[1:])
    x3_centers = 0.5 * (x3_params[:-1] + x3_params[1:])

    centroids = jnp.stack(
        jnp.meshgrid(x1_centers, x2_centers, x3_centers, indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))
    half_spans = 0.5 * jnp.stack(
        jnp.meshgrid(x1_widths, x2_widths, x3_widths, indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))

    batch_size = int(batch_size)
    num_cells = centroids.shape[0]
    pad = (-num_cells) % batch_size
    padded_num_cells = num_cells + pad

    centroids = jnp.pad(centroids, ((0, pad), (0, 0)))
    half_spans = jnp.pad(half_spans, ((0, pad), (0, 0)))
    valid_mask = jnp.arange(padded_num_cells) < num_cells

    centroid_batches = centroids.reshape((-1, batch_size, 3))
    half_span_batches = half_spans.reshape((-1, batch_size, 3))
    mask_batches = valid_mask.reshape((-1, batch_size))

    def accumulate_batch(total, batch):
        centroid_batch, half_span_batch, mask_batch = batch
        jacobians = _batched_cl_system_jacobian(centroid_batch)
        centers_next = jax.vmap(cl_system_jax)(centroid_batch)

        corner_offsets = half_span_batch[:, None, :] * _corner_signs[None, :, :]
        linearized_next_verts = centers_next[:, None, :] + jnp.einsum(
            "nij,nvj->nvi",
            jacobians,
            corner_offsets,
        )

        next_lower_bounds = jnp.min(linearized_next_verts, axis=1)
        next_upper_bounds = jnp.max(linearized_next_verts, axis=1)
        side_lengths = next_upper_bounds - next_lower_bounds
        volumes = jnp.prod(side_lengths, axis=-1)
        volumes = jnp.where(mask_batch, volumes, jnp.zeros_like(volumes))
        return total + jnp.sum(volumes), None

    total_volume, _ = jax.lax.scan(
        accumulate_batch,
        jnp.asarray(0.0, dtype=params.dtype),
        (centroid_batches, half_span_batches, mask_batches),
    )

    return total_volume


def epsilon_1_bound(
    params,
    *,
    args,
    batch_size=4096,
    ):
    """
    Sum the linearized one-step epsilon bound over all abstract cells.

    Each cell is propagated through the first-order model at its centroid.
    The returned bound is computed before Taylor-remainder inflation because
    the abstraction's interval-arithmetic remainder implementation is not
    JAX-differentiable. This function otherwise follows the same linearized
    AABB construction as ``unicycle_abstraction.build_abstraction``.
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

    x1_widths = jnp.diff(x1_params)
    x2_widths = jnp.diff(x2_params)
    x3_widths = jnp.diff(x3_params)

    x1_centers = 0.5 * (x1_params[:-1] + x1_params[1:])
    x2_centers = 0.5 * (x2_params[:-1] + x2_params[1:])
    x3_centers = 0.5 * (x3_params[:-1] + x3_params[1:])

    centroids = jnp.stack(
        jnp.meshgrid(x1_centers, x2_centers, x3_centers, indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))
    half_spans = 0.5 * jnp.stack(
        jnp.meshgrid(x1_widths, x2_widths, x3_widths, indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))

    batch_size = int(batch_size)
    num_cells = centroids.shape[0]
    pad = (-num_cells) % batch_size
    padded_num_cells = num_cells + pad

    centroids = jnp.pad(centroids, ((0, pad), (0, 0)))
    half_spans = jnp.pad(half_spans, ((0, pad), (0, 0)))
    valid_mask = jnp.arange(padded_num_cells) < num_cells

    centroid_batches = centroids.reshape((-1, batch_size, 3))
    half_span_batches = half_spans.reshape((-1, batch_size, 3))
    mask_batches = valid_mask.reshape((-1, batch_size))
    norm_epsilon = jnp.asarray(1e-12, dtype=params.dtype)

    def accumulate_batch(total, batch):
        centroid_batch, half_span_batch, mask_batch = batch
        jacobians = _batched_cl_system_jacobian(centroid_batch)
        centers_next = jax.vmap(cl_system_jax)(centroid_batch)

        corner_offsets = half_span_batch[:, None, :] * _corner_signs[None, :, :]
        linearized_next_verts = centers_next[:, None, :] + jnp.einsum(
            "nij,nvj->nvi",
            jacobians,
            corner_offsets,
        )
        next_lower_bounds = jnp.min(linearized_next_verts, axis=1)
        next_upper_bounds = jnp.max(linearized_next_verts, axis=1)
        next_centroids = 0.5 * (next_lower_bounds + next_upper_bounds)

        aabb_radii = 0.5 * jnp.linalg.norm(
            next_upper_bounds - next_lower_bounds,
            axis=-1,
        )
        center_offsets = centers_next - next_centroids
        # The linearized image is centered at f(centroid), so this offset is
        # normally zero. The smooth upper norm avoids NaN derivatives there.
        center_differences = jnp.sqrt(
            jnp.sum(jnp.square(center_offsets), axis=-1) + norm_epsilon
        )
        contributions = aabb_radii + center_differences
        contributions = jnp.where(
            mask_batch,
            contributions,
            jnp.zeros_like(contributions),
        )
        return total + jnp.sum(contributions), None

    epsilon_sum, _ = jax.lax.scan(
        accumulate_batch,
        jnp.asarray(0.0, dtype=params.dtype),
        (centroid_batches, half_span_batches, mask_batches),
    )
    return epsilon_sum


# def noninflated_image_volume(
#     params,
#     *,
#     args,
#     ):
#     """
#     Sum of all linearized AABB image volumes pre-inflation.
#     """

#     n1_internal, n2_internal, n3_internal = args['shape']
#     x1_lo, x2_lo, x3_lo = args['domain_lb']
#     x1_hi, x2_hi, x3_hi = args['domain_ub']
    
#     params = jnp.asarray(params)
#     u1 = params[:n1_internal]
#     u2 = params[n1_internal : n1_internal + n2_internal]
#     u3 = params[n1_internal + n2_internal : n1_internal + n2_internal + n3_internal]

#     # Convert gap-params -> actual y-line locations
#     x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
#     x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)
#     x3_params = make_lines_from_gaps(u3, x3_lo, x3_hi)

#     # Iterate over cells
#     total_volume = 0.0
#     count = 0
#     for i in range(n1_internal):
#         x1_lo, x1_hi = x1_params[i], x1_params[i+1]
#         for j in range(n2_internal):
#             x2_lo, x2_hi = x2_params[j], x2_params[j+1]
#             for k in range(n3_internal):
#                 x3_lo, x3_hi = x3_params[k], x3_params[k+1]

#                 lower_bounds = np.array([x1_lo, x2_lo, x3_lo])
#                 upper_bounds = np.array([x1_hi, x2_hi, x3_hi])
#                 all_verts = [list(combo) for combo in product(*zip(lower_bounds, upper_bounds))]
#                 all_verts = np.array(all_verts)

#                 # Compute cell centroid and evaluate Jacobian there; compute post-image AABB
#                 centroid = (lower_bounds + upper_bounds) / 2.0
#                 J = uss.jacobian(centroid)
#                 f_center = uss.cl_system_numeric(centroid)
#                 linearized_next_verts = np.array([
#                     uss.linear_cl_system(vert, centroid, J=J, f_center=f_center)
#                     for vert in all_verts
#                 ])
#                 next_lower_bounds = linearized_next_verts.min(axis=0)
#                 next_upper_bounds = linearized_next_verts.max(axis=0)

#                 # Compute Taylor remainder and inflate AABB
#                 R_lo, R_hi = uss.taylor_remainder(lb=lower_bounds,
#                                                   ub=upper_bounds)
#                 next_lower_bounds += R_lo
#                 next_upper_bounds += R_hi

#                 side_lengths = next_upper_bounds - next_lower_bounds
#                 volume = side_lengths[0]*side_lengths[1]*side_lengths[2]
#                 total_volume += volume

#                 if count % 1000 == 0:
#                     print(f"iter={count}, total_volume={total_volume}")
#                 count += 1

#     return total_volume
                

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
