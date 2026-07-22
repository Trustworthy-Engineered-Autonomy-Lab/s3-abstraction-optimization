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
from mountain_car_system import (
    MAX_POSITION,
    MAX_VELOCITY,
    MIN_POSITION,
    MIN_VELOCITY,
    cl_system_numeric,
    cl_system_jax,
    interval_cl_system,
    interval_cl_system_jax,
)


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
    x_next = cl_system_jax(corners)

    x1 = x_next[..., 0]
    x2 = x_next[..., 1]
    x1_lo_post = jnp.min(x1, axis=-1)
    x1_hi_post = jnp.max(x1, axis=-1)
    x2_lo_post = jnp.min(x2, axis=-1)
    x2_hi_post = jnp.max(x2, axis=-1)
    img_area = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post)
    return jnp.sum(img_area)



_DEFAULT_WITNESS_POINTS = np.array(
    [
        [0.5, 0.5],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=float,
)


def _logmeanexp_numpy(values, temperature, *, sign=1.0, axis=None):
    """Stable normalized LSE; ``sign=-1`` gives a normalized soft minimum."""

    values = np.asarray(values, dtype=float)
    if temperature <= 0.0:
        reducer = np.max if sign > 0.0 else np.min
        return reducer(values, axis=axis)
    scaled = sign * values / temperature
    maximum = np.max(scaled, axis=axis, keepdims=True)
    count = values.size if axis is None else values.shape[axis]
    result = maximum + np.log(
        np.mean(np.exp(scaled - maximum), axis=axis, keepdims=True)
    )
    result = sign * temperature * result
    if axis is None:
        return float(np.asarray(result).reshape(-1)[0])
    return np.squeeze(result, axis=axis)


def _soft_cell_faces_numpy(value, edges, temperature):
    """Smoothly select the left/right faces of the cell containing ``value``."""

    edges = np.asarray(edges, dtype=float)
    if temperature <= 0.0:
        index = int(np.searchsorted(edges, value, side="right") - 1)
        index = max(0, min(len(edges) - 2, index))
        return edges[index], edges[index + 1]
    left = edges[:-1]
    right = edges[1:]
    distance = np.maximum(np.maximum(left - value, value - right), 0.0)
    logits = -distance / temperature
    weights = np.exp(logits - np.max(logits))
    weights /= np.sum(weights)
    return float(weights @ left), float(weights @ right)


def _soft_cell_faces_jax(values, edges, temperature):
    """Batched differentiable counterpart of ``_soft_cell_faces_numpy``."""

    if temperature <= 0.0:
        indices = jnp.searchsorted(edges, values, side="right") - 1
        indices = jnp.clip(indices, 0, edges.shape[0] - 2)
        return edges[indices], edges[indices + 1]
    left = edges[:-1]
    right = edges[1:]
    distance = jnp.maximum(
        jnp.maximum(left[None, :] - values[:, None], values[:, None] - right[None, :]),
        0.0,
    )
    weights = jax.nn.softmax(-distance / temperature, axis=1)
    return weights @ left, weights @ right


def _snap_box_jax(lower, upper, x1_edges, x2_edges, snap_temperatures):
    """Return soft outer box and the two far faces used by box distance."""

    first_x_left, first_x_right = _soft_cell_faces_jax(
        lower[:, 0], x1_edges, snap_temperatures[0]
    )
    last_x_left, last_x_right = _soft_cell_faces_jax(
        upper[:, 0], x1_edges, snap_temperatures[0]
    )
    first_y_left, first_y_right = _soft_cell_faces_jax(
        lower[:, 1], x2_edges, snap_temperatures[1]
    )
    last_y_left, last_y_right = _soft_cell_faces_jax(
        upper[:, 1], x2_edges, snap_temperatures[1]
    )
    outer_lower = jnp.stack([first_x_left, first_y_left], axis=-1)
    outer_upper = jnp.stack([last_x_right, last_y_right], axis=-1)
    far_lower = jnp.stack([last_x_left, last_y_left], axis=-1)
    far_upper = jnp.stack([first_x_right, first_y_right], axis=-1)
    return outer_lower, outer_upper, far_lower, far_upper


def _logsumexp_numpy(values, temperature, *, axis=None):
    """Stable, unnormalized temperature-scaled log-sum-exp."""

    values = np.asarray(values, dtype=float)
    if temperature <= 0.0:
        raise ValueError("LSE temperatures must be positive")
    scaled = values / temperature
    maximum = np.max(scaled, axis=axis, keepdims=True)
    result = temperature * (
        maximum
        + np.log(np.sum(np.exp(scaled - maximum), axis=axis, keepdims=True))
    )
    if axis is None:
        return float(np.asarray(result).reshape(-1)[0])
    return np.squeeze(result, axis=axis)


def _weighted_norm_numpy(values, weights, norm_order):
    return np.linalg.norm(
        np.asarray(values) * np.asarray(weights),
        ord=norm_order,
        axis=-1,
    )


def _weighted_norm_jax(values, weights, norm_order):
    weighted = values * weights
    if norm_order == 1.0:
        return jnp.sum(jnp.abs(weighted), axis=-1)
    if norm_order == 2.0:
        return jnp.sqrt(jnp.sum(jnp.square(weighted), axis=-1))
    if np.isinf(norm_order):
        return jnp.max(jnp.abs(weighted), axis=-1)
    return jnp.linalg.norm(weighted, ord=norm_order, axis=-1)


def upward_proxy_bruteforce(params, *, args):
    """NumPy reference for the paper-formalized upward proxy.

    For every quantization cell, this propagates the cell centroid and a soft
    reachable AABB.  At step ``k`` it evaluates

        radius(Reach(A_k)) + ||f^k(x_c) - centroid(Reach(A_k))||,

    then applies an unnormalized LSE over steps and an unnormalized LSE over
    cells, matching Equation (13).
    """

    horizon = int(args['horizon'])
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    number_of_scores = horizon + 1
    temp_in = float(args['temp_in'])
    temp_out = float(args['temp_out'])
    if temp_in <= 0.0 or temp_out <= 0.0:
        raise ValueError("temp_in and temp_out must be positive")
    propagation = args.get('propagation', 'interval')
    if propagation not in {'corners', 'interval'}:
        raise ValueError("propagation must be 'corners' or 'interval'")
    inflation = np.asarray(
        args.get('inflation_coefs', [0.0, 0.0]), dtype=float
    )
    coordinate_weights = np.asarray(
        args.get('coordinate_weights', [1.0, 1.0]), dtype=float
    )
    norm_order = float(args.get('norm_order', 2.0))

    n1, n2 = args['shape']
    domain_lower = np.asarray(args['domain_lb'], dtype=float)
    domain_upper = np.asarray(args['domain_ub'], dtype=float)
    snap_temperatures = np.asarray(
        args.get(
            'snap_temperatures',
            (domain_upper - domain_lower) / (20.0 * np.asarray([n1, n2])),
        ),
        dtype=float,
    )
    x1_edges, x2_edges = extract_grid_params(
        params, (n1, n2), domain_lower, domain_upper
    )

    cell_values = []
    for i in range(n1):
        for j in range(n2):
            lower = np.array([x1_edges[i], x2_edges[j]], dtype=float)
            upper = np.array([x1_edges[i + 1], x2_edges[j + 1]], dtype=float)
            witness = 0.5 * (lower + upper)
            step_scores = []
            for _ in range(number_of_scores):
                witness = cl_system_numeric(witness)
                if propagation == 'interval':
                    raw_lower, raw_upper = interval_cl_system(lower, upper)
                else:
                    corners = np.array(
                        [[lower[0], lower[1]], [lower[0], upper[1]],
                         [upper[0], upper[1]], [upper[0], lower[1]]]
                    )
                    images = np.stack([cl_system_numeric(x) for x in corners])
                    raw_lower = np.min(images, axis=0)
                    raw_upper = np.max(images, axis=0)

                raw_lower = np.maximum(domain_lower, raw_lower - inflation)
                raw_upper = np.minimum(domain_upper, raw_upper + inflation)
                first_faces = [
                    _soft_cell_faces_numpy(
                        raw_lower[d], edges, snap_temperatures[d]
                    )
                    for d, edges in enumerate((x1_edges, x2_edges))
                ]
                last_faces = [
                    _soft_cell_faces_numpy(
                        raw_upper[d], edges, snap_temperatures[d]
                    )
                    for d, edges in enumerate((x1_edges, x2_edges))
                ]
                lower = np.array([face[0] for face in first_faces])
                upper = np.array([face[1] for face in last_faces])

                reach_centroid = 0.5 * (lower + upper)
                reach_radius = 0.5 * _weighted_norm_numpy(
                    upper - lower, coordinate_weights, norm_order
                )
                witness_offset = _weighted_norm_numpy(
                    witness - reach_centroid, coordinate_weights, norm_order
                )
                step_scores.append(reach_radius + witness_offset)

            cell_values.append(
                _logsumexp_numpy(step_scores, temp_in, axis=0)
            )

    return _logsumexp_numpy(cell_values, temp_out, axis=0)


def upward_proxy(params, *, args, batch_size=4096):
    """Batched JAX implementation of Equation (13)'s upward proxy.

    Reachability may be changed with ``propagation`` and
    ``snap_temperatures``, but the abstract step score and both smoothing
    operations deliberately follow the paper formulation.
    """

    horizon = int(args['horizon'])
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    number_of_scores = horizon + 1
    temp_in_value = float(args['temp_in'])
    temp_out_value = float(args['temp_out'])
    if temp_in_value <= 0.0 or temp_out_value <= 0.0:
        raise ValueError("temp_in and temp_out must be positive")
    propagation = args.get('propagation', 'interval')
    if propagation not in {'corners', 'interval'}:
        raise ValueError("propagation must be 'corners' or 'interval'")
    norm_order = float(args.get('norm_order', 2.0))

    n1, n2 = args['shape']
    x1_lo, x2_lo = args['domain_lb']
    x1_hi, x2_hi = args['domain_ub']
    params = jnp.asarray(params)
    x1_edges = make_lines_from_gaps(params[:n1], x1_lo, x1_hi)
    x2_edges = make_lines_from_gaps(params[n1:n1 + n2], x2_lo, x2_hi)
    x1_widths = jnp.diff(x1_edges)
    x2_widths = jnp.diff(x2_edges)
    x1_centers = 0.5 * (x1_edges[:-1] + x1_edges[1:])
    x2_centers = 0.5 * (x2_edges[:-1] + x2_edges[1:])
    centroids = jnp.stack(
        jnp.meshgrid(x1_centers, x2_centers, indexing='ij'), axis=-1
    ).reshape((-1, 2))
    half_spans = 0.5 * jnp.stack(
        jnp.meshgrid(x1_widths, x2_widths, indexing='ij'), axis=-1
    ).reshape((-1, 2))

    batch_size = int(batch_size)
    num_cells = centroids.shape[0]
    pad = (-num_cells) % batch_size
    padded_num_cells = num_cells + pad
    centroids = jnp.pad(centroids, ((0, pad), (0, 0)))
    half_spans = jnp.pad(half_spans, ((0, pad), (0, 0)))
    valid_mask = jnp.arange(padded_num_cells) < num_cells
    centroid_batches = centroids.reshape((-1, batch_size, 2))
    half_span_batches = half_spans.reshape((-1, batch_size, 2))
    mask_batches = valid_mask.reshape((-1, batch_size))

    domain_lower = jnp.asarray(args['domain_lb'], dtype=params.dtype)
    domain_upper = jnp.asarray(args['domain_ub'], dtype=params.dtype)
    inflation = jnp.asarray(
        args.get('inflation_coefs', [0.0, 0.0]), dtype=params.dtype
    )
    coordinate_weights = jnp.asarray(
        args.get('coordinate_weights', [1.0, 1.0]), dtype=params.dtype
    )
    domain_span = (
        np.asarray(args['domain_ub'], dtype=float)
        - np.asarray(args['domain_lb'], dtype=float)
    )
    snap_temperature_array = np.asarray(
        args.get(
            'snap_temperatures',
            domain_span / np.asarray([20.0 * n1, 20.0 * n2]),
        ),
        dtype=float,
    )
    if snap_temperature_array.shape != (2,) or np.any(
        snap_temperature_array < 0.0
    ):
        raise ValueError("snap_temperatures must be two nonnegative values")
    snap_temperatures = tuple(float(x) for x in snap_temperature_array)
    corner_signs = jnp.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        dtype=params.dtype,
    )

    def evaluate_batch(unused, batch):
        centroid_batch, half_span_batch, mask_batch = batch
        initial_lower = centroid_batch - half_span_batch
        initial_upper = centroid_batch + half_span_batch

        def rollout_step(carry, _):
            lower, upper, witness = carry
            next_witness = cl_system_jax(witness)
            if propagation == 'interval':
                raw_lower, raw_upper = interval_cl_system_jax(lower, upper)
            else:
                corners = (
                    0.5 * (lower + upper)[:, None, :]
                    + 0.5 * (upper - lower)[:, None, :]
                    * corner_signs[None, :, :]
                )
                images = cl_system_jax(corners)
                raw_lower = jnp.min(images, axis=1)
                raw_upper = jnp.max(images, axis=1)

            raw_lower = jnp.maximum(domain_lower, raw_lower - inflation)
            raw_upper = jnp.minimum(domain_upper, raw_upper + inflation)
            next_lower, next_upper, _, _ = _snap_box_jax(
                raw_lower,
                raw_upper,
                x1_edges,
                x2_edges,
                snap_temperatures,
            )
            reach_centroid = 0.5 * (next_lower + next_upper)
            reach_radius = 0.5 * _weighted_norm_jax(
                next_upper - next_lower,
                coordinate_weights,
                norm_order,
            )
            witness_offset = _weighted_norm_jax(
                next_witness - reach_centroid,
                coordinate_weights,
                norm_order,
            )
            step_score = reach_radius + witness_offset
            return (next_lower, next_upper, next_witness), step_score

        _, step_scores = jax.lax.scan(
            rollout_step,
            (initial_lower, initial_upper, centroid_batch),
            xs=None,
            length=number_of_scores,
        )
        per_cell = temp_in_value * jax.scipy.special.logsumexp(
            step_scores / temp_in_value, axis=0
        )
        per_cell = jnp.where(mask_batch, per_cell, -jnp.inf)
        return unused, per_cell

    _, per_cell_batches = jax.lax.scan(
        evaluate_batch,
        jnp.asarray(0.0, dtype=params.dtype),
        (centroid_batches, half_span_batches, mask_batches),
    )
    return temp_out_value * jax.scipy.special.logsumexp(
        per_cell_batches / temp_out_value
    )


def simulation_aligned_proxy_bruteforce(params, *, args):
    """NumPy-loop reference for :func:`simulation_aligned_proxy`."""

    horizon = int(args['horizon'])
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    temp_in = float(args['temp_in'])
    temp_out = float(args['temp_out'])
    temp_witness = float(args.get('temp_witness', 0.0))
    metric_temp = float(args.get('metric_temp', 0.0))
    mean_weight = float(args.get('mean_weight', 0.0))
    propagation = args.get('propagation', 'corners')
    inflation = np.asarray(args.get('inflation_coefs', [0.0, 0.0]), dtype=float)
    coordinate_weights = np.asarray(
        args.get('coordinate_weights', [1.0, 1.0]), dtype=float
    )
    witness_points = np.asarray(
        args.get('witness_points', _DEFAULT_WITNESS_POINTS), dtype=float
    )

    n1, n2 = args['shape']
    domain_lower = np.asarray(args['domain_lb'], dtype=float)
    domain_upper = np.asarray(args['domain_ub'], dtype=float)
    snap_temperatures = np.asarray(
        args.get(
            'snap_temperatures',
            (domain_upper - domain_lower) / (20.0 * np.asarray([n1, n2])),
        ),
        dtype=float,
    )
    x1_edges, x2_edges = extract_grid_params(
        params, (n1, n2), domain_lower, domain_upper
    )

    cell_values = []
    for i in range(n1):
        for j in range(n2):
            lower = np.array([x1_edges[i], x2_edges[j]], dtype=float)
            upper = np.array([x1_edges[i + 1], x2_edges[j + 1]], dtype=float)
            witnesses = lower + witness_points * (upper - lower)
            per_step = []
            for _ in range(horizon):
                witnesses = np.stack([cl_system_numeric(x) for x in witnesses])
                if propagation == 'interval':
                    raw_lower, raw_upper = interval_cl_system(lower, upper)
                elif propagation == 'corners':
                    corners = np.array(
                        [[lower[0], lower[1]], [lower[0], upper[1]],
                         [upper[0], upper[1]], [upper[0], lower[1]]]
                    )
                    images = np.stack([cl_system_numeric(x) for x in corners])
                    raw_lower = np.min(images, axis=0)
                    raw_upper = np.max(images, axis=0)
                else:
                    raise ValueError("propagation must be 'corners' or 'interval'")
                raw_lower = np.maximum(domain_lower, raw_lower - inflation)
                raw_upper = np.minimum(domain_upper, raw_upper + inflation)

                first_faces = [
                    _soft_cell_faces_numpy(raw_lower[d], edges, snap_temperatures[d])
                    for d, edges in enumerate((x1_edges, x2_edges))
                ]
                last_faces = [
                    _soft_cell_faces_numpy(raw_upper[d], edges, snap_temperatures[d])
                    for d, edges in enumerate((x1_edges, x2_edges))
                ]
                lower = np.array([face[0] for face in first_faces])
                upper = np.array([face[1] for face in last_faces])
                far_upper = np.array([face[1] for face in first_faces])
                far_lower = np.array([face[0] for face in last_faces])
                residual = np.maximum(
                    np.maximum(
                        far_lower[None, :] - witnesses,
                        witnesses - far_upper[None, :],
                    ),
                    0.0,
                ) * coordinate_weights
                per_step.append(
                    _logmeanexp_numpy(residual, metric_temp, axis=1)
                    if metric_temp > 0.0
                    else np.max(residual, axis=1)
                )

            per_witness = _logmeanexp_numpy(
                np.stack(per_step), temp_in, axis=0
            )
            cell_values.append(
                _logmeanexp_numpy(
                    per_witness, temp_witness, sign=-1.0, axis=0
                )
            )

    cell_values = np.asarray(cell_values)
    smooth_max = _logmeanexp_numpy(cell_values, temp_out, axis=0)
    return (1.0 - mean_weight) * smooth_max + mean_weight * np.mean(cell_values)


def simulation_aligned_proxy(
    params,
    *,
    args,
    batch_size=4096,
    ):
    """Smooth surrogate aligned directly with the graph simulation metric.

    The abstract reachable box is propagated either by corner images or by
    neural-network interval bounds, then softly snapped to the parameterized
    grid.  At each time, the loss is the worst distance from synchronized
    concrete witnesses to the reachable *cells*, matching
    ``mountain_car_simulation_analysis``.  A soft minimum over witness
    candidates approximates the metric's inner minimization; horizon and cell
    maxima use normalized LSEs.  ``mean_weight`` optionally blends the cell
    mean into the outer worst-case objective.  This is an empirical ablation,
    not the paper-formalized ``upward_proxy``.
    """

    horizon = int(args['horizon'])
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    temp_in_value = float(args['temp_in'])
    temp_out_value = float(args['temp_out'])
    temp_witness_value = float(args.get('temp_witness', 0.0))
    metric_temp_value = float(args.get('metric_temp', 0.0))
    mean_weight = float(args.get('mean_weight', 0.0))
    if not 0.0 <= mean_weight <= 1.0:
        raise ValueError("mean_weight must lie in [0, 1]")
    propagation = args.get('propagation', 'corners')
    if propagation not in {'corners', 'interval'}:
        raise ValueError("propagation must be 'corners' or 'interval'")
    inflation_coefs = args.get('inflation_coefs', [0.0, 0.0])

    n1_internal, n2_internal = args['shape']
    x1_lo, x2_lo = args['domain_lb']
    x1_hi, x2_hi = args['domain_ub']

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]

    x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
    x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)

    x1_widths = jnp.diff(x1_params)
    x2_widths = jnp.diff(x2_params)
    x1_centers = 0.5 * (x1_params[:-1] + x1_params[1:])
    x2_centers = 0.5 * (x2_params[:-1] + x2_params[1:])

    centroids = jnp.stack(
        jnp.meshgrid(x1_centers, x2_centers, indexing="ij"),
        axis=-1,
    ).reshape((-1, 2))
    half_spans = 0.5 * jnp.stack(
        jnp.meshgrid(x1_widths, x2_widths, indexing="ij"),
        axis=-1,
    ).reshape((-1, 2))

    batch_size = int(batch_size)
    num_cells = centroids.shape[0]
    pad = (-num_cells) % batch_size
    padded_num_cells = num_cells + pad

    centroids = jnp.pad(centroids, ((0, pad), (0, 0)))
    half_spans = jnp.pad(half_spans, ((0, pad), (0, 0)))
    valid_mask = jnp.arange(padded_num_cells) < num_cells

    centroid_batches = centroids.reshape((-1, batch_size, 2))
    half_span_batches = half_spans.reshape((-1, batch_size, 2))
    mask_batches = valid_mask.reshape((-1, batch_size))
    inflation_coefs = jnp.asarray(inflation_coefs, dtype=params.dtype)
    coordinate_weights = jnp.asarray(
        args.get('coordinate_weights', [1.0, 1.0]), dtype=params.dtype
    )
    witness_points = jnp.asarray(
        args.get('witness_points', _DEFAULT_WITNESS_POINTS), dtype=params.dtype
    )
    if witness_points.ndim != 2 or witness_points.shape[1] != 2:
        raise ValueError("witness_points must have shape (K, 2)")
    domain_lower = jnp.asarray(args['domain_lb'], dtype=params.dtype)
    domain_upper = jnp.asarray(args['domain_ub'], dtype=params.dtype)
    domain_span_numpy = (
        np.asarray(args['domain_ub'], dtype=float)
        - np.asarray(args['domain_lb'], dtype=float)
    )
    default_snap_temperatures = domain_span_numpy / np.asarray(
        [20.0 * n1_internal, 20.0 * n2_internal], dtype=float
    )
    snap_temperatures_array = np.asarray(
        args.get('snap_temperatures', default_snap_temperatures),
        dtype=float,
    )
    if snap_temperatures_array.shape != (2,) or np.any(snap_temperatures_array < 0.0):
        raise ValueError("snap_temperatures must be two nonnegative values")
    snap_temperatures = tuple(float(value) for value in snap_temperatures_array)
    corner_signs = jnp.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=params.dtype,
    )

    def evaluate_batch(unused, batch):
        centroid_batch, half_span_batch, mask_batch = batch
        initial_lower = centroid_batch - half_span_batch
        initial_upper = centroid_batch + half_span_batch
        witnesses = (
            initial_lower[:, None, :]
            + witness_points[None, :, :]
            * (initial_upper - initial_lower)[:, None, :]
        )

        def rollout_step(carry, _):
            current_lower, current_upper, current_witnesses = carry
            next_witnesses = cl_system_jax(current_witnesses)
            if propagation == 'interval':
                raw_lower, raw_upper = interval_cl_system_jax(
                    current_lower, current_upper
                )
            else:
                current_corners = (
                    0.5 * (current_lower + current_upper)[:, None, :]
                    + 0.5 * (current_upper - current_lower)[:, None, :]
                    * corner_signs[None, :, :]
                )
                next_corners = cl_system_jax(current_corners)
                raw_lower = jnp.min(next_corners, axis=1)
                raw_upper = jnp.max(next_corners, axis=1)

            raw_lower = jnp.maximum(domain_lower, raw_lower - inflation_coefs)
            raw_upper = jnp.minimum(domain_upper, raw_upper + inflation_coefs)
            outer_lower, outer_upper, far_lower, far_upper = _snap_box_jax(
                raw_lower,
                raw_upper,
                x1_params,
                x2_params,
                snap_temperatures,
            )
            residual = jnp.maximum(
                jnp.maximum(
                    far_lower[:, None, :] - next_witnesses,
                    next_witnesses - far_upper[:, None, :],
                ),
                0.0,
            ) * coordinate_weights[None, None, :]
            if metric_temp_value > 0.0:
                step_values = metric_temp_value * (
                    jax.scipy.special.logsumexp(
                        residual / metric_temp_value, axis=-1
                    ) - jnp.log(jnp.asarray(2.0, dtype=params.dtype))
                )
            else:
                step_values = jnp.max(residual, axis=-1)
            return (
                outer_lower,
                outer_upper,
                next_witnesses,
            ), step_values

        _, step_values = jax.lax.scan(
            rollout_step,
            (initial_lower, initial_upper, witnesses),
            xs=None,
            length=horizon,
        )
        if temp_in_value > 0.0:
            per_witness_values = temp_in_value * (
                jax.scipy.special.logsumexp(
                    step_values / temp_in_value, axis=0
                ) - jnp.log(jnp.asarray(float(horizon), dtype=params.dtype))
            )
        else:
            per_witness_values = jnp.max(step_values, axis=0)
        if witness_points.shape[0] == 1:
            per_cell_values = per_witness_values[:, 0]
        elif temp_witness_value > 0.0:
            per_cell_values = -temp_witness_value * (
                jax.scipy.special.logsumexp(
                    -per_witness_values / temp_witness_value, axis=1
                ) - jnp.log(
                    jnp.asarray(float(witness_points.shape[0]), dtype=params.dtype)
                )
            )
        else:
            per_cell_values = jnp.min(per_witness_values, axis=1)
        per_cell_values = jnp.where(
            mask_batch,
            per_cell_values,
            -jnp.inf,
        )
        return unused, per_cell_values

    _, per_cell_value_batches = jax.lax.scan(
        evaluate_batch,
        jnp.asarray(0.0, dtype=params.dtype),
        (centroid_batches, half_span_batches, mask_batches),
    )
    flat_values = per_cell_value_batches.reshape(-1)
    if temp_out_value > 0.0:
        smooth_max = temp_out_value * (
            jax.scipy.special.logsumexp(flat_values / temp_out_value)
            - jnp.log(jnp.asarray(float(num_cells), dtype=params.dtype))
        )
    else:
        smooth_max = jnp.max(flat_values)
    mean_value = jnp.sum(jnp.where(valid_mask, flat_values, 0.0)) / num_cells
    return (1.0 - mean_weight) * smooth_max + mean_weight * mean_value

def epsilon_sum(
    params,
    *,
    args,
    batch_size=4096,
    ):
    """
    Batched JAX implementation of the finite-horizon upward proxy.
    """

    horizon = args['horizon']
    inflation_coefs = args['inflation_coefs']
    temp_in = args['temp_in']
    temp_out = args['temp_out']

    n1_internal, n2_internal = args['shape']
    x1_lo, x2_lo = args['domain_lb']
    x1_hi, x2_hi = args['domain_ub']

    params = jnp.asarray(params)
    u1 = params[:n1_internal]
    u2 = params[n1_internal : n1_internal + n2_internal]

    x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
    x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)

    x1_widths = jnp.diff(x1_params)
    x2_widths = jnp.diff(x2_params)
    x1_centers = 0.5 * (x1_params[:-1] + x1_params[1:])
    x2_centers = 0.5 * (x2_params[:-1] + x2_params[1:])

    centroids = jnp.stack(
        jnp.meshgrid(x1_centers, x2_centers, indexing="ij"),
        axis=-1,
    ).reshape((-1, 2))
    half_spans = 0.5 * jnp.stack(
        jnp.meshgrid(x1_widths, x2_widths, indexing="ij"),
        axis=-1,
    ).reshape((-1, 2))

    batch_size = int(batch_size)
    num_cells = centroids.shape[0]
    pad = (-num_cells) % batch_size
    padded_num_cells = num_cells + pad

    centroids = jnp.pad(centroids, ((0, pad), (0, 0)))
    half_spans = jnp.pad(half_spans, ((0, pad), (0, 0)))
    valid_mask = jnp.arange(padded_num_cells) < num_cells

    centroid_batches = centroids.reshape((-1, batch_size, 2))
    half_span_batches = half_spans.reshape((-1, batch_size, 2))
    mask_batches = valid_mask.reshape((-1, batch_size))
    inflation_coefs = jnp.asarray(inflation_coefs, dtype=params.dtype)
    temp_in = jnp.asarray(temp_in, dtype=params.dtype)
    temp_out = jnp.asarray(temp_out, dtype=params.dtype)
    norm_epsilon = jnp.asarray(1e-12, dtype=params.dtype)
    corner_signs = jnp.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=params.dtype,
    )

    def evaluate_batch(unused, batch):
        centroid_batch, half_span_batch, mask_batch = batch
        initial_corners = (
            centroid_batch[:, None, :]
            + half_span_batch[:, None, :] * corner_signs[None, :, :]
        )

        def rollout_step(carry, _):
            witness, current_corners = carry
            witness = cl_system_jax(witness)
            next_corners = cl_system_jax(current_corners)
            next_lower_bounds = jnp.min(next_corners, axis=1)
            next_upper_bounds = jnp.max(next_corners, axis=1)
            reach_centroids = 0.5 * (
                next_lower_bounds + next_upper_bounds
            )

            image_spans = next_upper_bounds - next_lower_bounds
            reach_radii = 0.5 * jnp.sqrt(
                jnp.sum(jnp.square(image_spans), axis=-1) + norm_epsilon
            )
            witness_offsets = witness - reach_centroids
            witness_differences = jnp.sqrt(
                jnp.sum(jnp.square(witness_offsets), axis=-1)
                + norm_epsilon
            )
            step_values = reach_radii + witness_differences

            inflated_half_spans = 0.5 * image_spans + inflation_coefs
            inflated_corners = (
                reach_centroids[:, None, :]
                + inflated_half_spans[:, None, :]
                * corner_signs[None, :, :]
            )
            return (witness, inflated_corners), step_values

        carry = (centroid_batch, initial_corners)
        step_value_list = []
        for _ in range(horizon):
            carry, values = rollout_step(carry, None)
            step_value_list.append(values)
        step_values = jnp.stack(step_value_list, axis=0)

        per_cell_values = temp_in * jax.scipy.special.logsumexp(
            step_values / temp_in,
            axis=0,
        )
        per_cell_values = jnp.where(
            mask_batch,
            per_cell_values,
            jnp.zeros_like(per_cell_values),
        )
        return unused, per_cell_values

    _, per_cell_value_batches = jax.lax.scan(
        evaluate_batch,
        jnp.asarray(0.0, dtype=params.dtype),
        (centroid_batches, half_span_batches, mask_batches),
    )
    return jnp.sum(per_cell_value_batches)


# def epsilon_H_bound(
#     params,
#     *,
#     args,
#     ):
#     """
#     Sum a smooth finite-horizon epsilon bound over all abstract cells.
#     """

#     horizon = args['horizon']
#     temp = args['temp']
#     inflation_coefs = args['inflation_coefs']

#     n1_internal, n2_internal = args['shape']
#     x1_lo, x2_lo = args['domain_lb']
#     x1_hi, x2_hi = args['domain_ub']

#     params = jnp.asarray(params)
#     u1 = params[:n1_internal]
#     u2 = params[n1_internal : n1_internal + n2_internal]

#     x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
#     x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)

#     x1_los = x1_params[:-1]
#     x1_his = x1_params[1:]
#     x2_los = x2_params[:-1]
#     x2_his = x2_params[1:]

#     x1_lo_grid, x2_lo_grid = jnp.meshgrid(x1_los, x2_los, indexing="ij")
#     x1_hi_grid, x2_hi_grid = jnp.meshgrid(x1_his, x2_his, indexing="ij")
#     lower_bounds = jnp.stack([x1_lo_grid, x2_lo_grid], axis=-1)
#     upper_bounds = jnp.stack([x1_hi_grid, x2_hi_grid], axis=-1)

#     def aabb_corners(lower, upper):
#         return jnp.stack(
#             [
#                 lower,
#                 jnp.stack([lower[..., 0], upper[..., 1]], axis=-1),
#                 upper,
#                 jnp.stack([upper[..., 0], lower[..., 1]], axis=-1),
#             ],
#             axis=-2,
#         )

#     witnesses = 0.5 * (lower_bounds + upper_bounds)
#     corners = aabb_corners(lower_bounds, upper_bounds)
#     inflation_coefs = jnp.asarray(inflation_coefs, dtype=params.dtype)
#     temp = jnp.asarray(temp, dtype=params.dtype)
#     norm_epsilon = jnp.asarray(1e-12, dtype=params.dtype)

#     def rollout_step(carry, _):
#         witness, current_corners = carry
#         witness = cl_system_jax(witness)
#         next_corners = cl_system_jax(current_corners)
#         next_lower_bounds = jnp.min(next_corners, axis=-2)
#         next_upper_bounds = jnp.max(next_corners, axis=-2)

#         reach_centroids = 0.5 * (next_lower_bounds + next_upper_bounds)
#         reach_spans = next_upper_bounds - next_lower_bounds
#         reach_radii = 0.5 * jnp.sqrt(
#             jnp.sum(jnp.square(reach_spans), axis=-1) + norm_epsilon
#         )
#         witness_offsets = witness - reach_centroids
#         witness_differences = jnp.sqrt(
#             jnp.sum(jnp.square(witness_offsets), axis=-1) + norm_epsilon
#         )
#         step_bounds = reach_radii + witness_differences

#         inflated_lower_bounds = next_lower_bounds - inflation_coefs
#         inflated_upper_bounds = next_upper_bounds + inflation_coefs
#         inflated_corners = aabb_corners(
#             inflated_lower_bounds,
#             inflated_upper_bounds,
#         )
#         return (witness, inflated_corners), step_bounds

#     _, step_bounds = jax.lax.scan(
#         rollout_step,
#         (witnesses, corners),
#         xs=None,
#         length=horizon,
#     )
#     per_cell_bounds = temp * jax.scipy.special.logsumexp(
#         step_bounds / temp,
#         axis=0,
#     )
#     return jnp.sum(per_cell_bounds)

# def epsilon_H_bound(
#         params,
#         *,
#         args
#     ):

#     horizon = args['horizon']
#     temp = args['temp']
#     inflation_coef = args['inflation_coef']

#     n1_internal, n2_internal = args['shape']
#     x1_lo, x2_lo = args['domain_lb']
#     x1_hi, x2_hi = args['domain_ub']

#     params = jnp.asarray(params)
#     u1 = params[:n1_internal]
#     u2 = params[n1_internal : n1_internal + n2_internal]

#     x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
#     x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)

#     epsilon_sum = 0.0
#     for i in range(n1_internal):
#         x_lo, x_hi = x1_params[i], x1_params[i+1]
#         for j in range(n2_internal):
#             y_lo, y_hi = x2_params[j], x2_params[j+1]

#             # Specify cell domain and corners
#             lower_bounds = np.array([x_lo, y_lo])
#             upper_bounds = np.array([x_hi, y_hi])
#             all_verts = np.array([
#                 [x1_lo, x2_lo],
#                 [x1_hi, x2_lo],
#                 [x1_lo, x2_hi],
#                 [x1_hi, x2_hi]
#             ])
#             xk = (lower_bounds + upper_bounds) / 2.0

#             # Iterate over horizon
#             s_values = []
#             for _ in range(horizon):
                
#                 # Push the concrete witness through dynamics
#                 xk = dynamics_jax(xk, x_star=XSTAR)

#                 # Determine reachable AABB
#                 all_verts = np.array([
#                     dynamics_jax(v, x_star=XSTAR) for v in all_verts 
#                 ])
#                 next_lower_bounds = all_verts.min(axis=0)
#                 next_upper_bounds = all_verts.max(axis=0)

#                 # Reachable AABB dimensions
#                 reach_centroid = (next_upper_bounds + next_lower_bounds) / 2.0
#                 reach_radius = 0.5 * np.linalg.norm(next_upper_bounds - next_lower_bounds)

#                 # s_value calculation
#                 witness_diff = np.linalg.norm(xk - reach_centroid)
#                 sk = reach_radius + witness_diff
#                 s_values.append(sk)

#                 # Inflate AABB and update all vertices
#                 inflation_bound = inflation_coef*np.ones_like(next_lower_bounds)
#                 next_upper_bounds += inflation_bound
#                 next_lower_bounds -= inflation_bound
#                 all_verts = np.array([
#                     [next_lower_bounds[0], next_upper_bounds[0]],
#                     [next_lower_bounds[1], next_upper_bounds[0]],
#                     [next_lower_bounds[0], next_upper_bounds[1]],
#                     [next_lower_bounds[1], next_upper_bounds[1]]
#                 ])

#             # Log-sum-exp over s-values
#             s_values = np.asarray(s_values)
#             p = temp * np.log(np.sum(np.exp(s_values / temp)))
#             epsilon_sum += p
    
#     return epsilon_sum

# def image_volume(
#     params,
#     *,
#     args
#     ):
#     """
#     Sum of post-image AABB volume for each abstract cell.
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

#     # Build all cells' corners: (n1, n2, 4, 2)
#     x1_los = x1_params[:-1]
#     x1_his = x1_params[1:]
#     x2_los = x2_params[:-1]
#     x2_his = x2_params[1:]
#     x3_los = x3_params[:-1]
#     x3_his = x3_params[1:]

#     n1 = x1_los.shape[0]
#     n2 = x2_los.shape[0]
#     n3 = x3_los.shape[0]
#     x1_lo_grid = jnp.broadcast_to(x1_los[:, None, None], (n1, n2, n3))
#     x1_hi_grid = jnp.broadcast_to(x1_his[:, None, None], (n1, n2, n3))
#     x2_lo_grid = jnp.broadcast_to(x2_los[None, :, None], (n1, n2, n3))
#     x2_hi_grid = jnp.broadcast_to(x2_his[None, :, None], (n1, n2, n3))
#     x3_lo_grid = jnp.broadcast_to(x3_los[None, None, :], (n1, n2, n3))
#     x3_hi_grid = jnp.broadcast_to(x3_his[None, None, :], (n1, n2, n3))

#     # All 8 corners per cell: (n1, n2, n3, 8, 3)
#     corners = jnp.stack(
#         [
#             jnp.stack([x1_lo_grid, x2_lo_grid, x3_lo_grid], axis=-1),
#             jnp.stack([x1_lo_grid, x2_lo_grid, x3_hi_grid], axis=-1),
#             jnp.stack([x1_lo_grid, x2_hi_grid, x3_lo_grid], axis=-1),
#             jnp.stack([x1_lo_grid, x2_hi_grid, x3_hi_grid], axis=-1),
#             jnp.stack([x1_hi_grid, x2_lo_grid, x3_lo_grid], axis=-1),
#             jnp.stack([x1_hi_grid, x2_lo_grid, x3_hi_grid], axis=-1),
#             jnp.stack([x1_hi_grid, x2_hi_grid, x3_lo_grid], axis=-1),
#             jnp.stack([x1_hi_grid, x2_hi_grid, x3_hi_grid], axis=-1),
#         ],
#         axis=-2,
#     )

#     flat_corners = corners.reshape((-1, 3))
#     flat_next = jax.vmap(cl_system_jax)(flat_corners)
#     x_next = flat_next.reshape(corners.shape)

#     x1 = x_next[..., 0]
#     x2 = x_next[..., 1]
#     x3 = x_next[..., 2]
#     x1_lo_post = jnp.min(x1, axis=-1)
#     x1_hi_post = jnp.max(x1, axis=-1)
#     x2_lo_post = jnp.min(x2, axis=-1)
#     x2_hi_post = jnp.max(x2, axis=-1)
#     theta_span = theta_min_arc_length(x3)
#     img_volume = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post) * theta_span

#     return jnp.sum(img_volume)

# def image_volume_over_parent(
#     params,
#     *,
#     args
#     ):
#     """Sum over cells of (image AABB volume / parent AABB volume).
#     """

#     n1_internal, n2_internal, n3_internal = args['shape']
#     x1_lo, x2_lo, x3_lo = args['domain_lb']
#     x1_hi, x2_hi, x3_hi = args['domain_ub']

#     params = jnp.asarray(params)
#     u1 = params[:n1_internal]
#     u2 = params[n1_internal : n1_internal + n2_internal]
#     u3 = params[n1_internal + n2_internal : n1_internal + n2_internal + n3_internal]

#     # Convert gap-params -> actual grid line locations
#     x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
#     x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)
#     x3_params = make_lines_from_gaps(u3, x3_lo, x3_hi)

#     # Build all cells' corners: (n1, n2, n3, 8, 3)
#     x1_los = x1_params[:-1]
#     x1_his = x1_params[1:]
#     x2_los = x2_params[:-1]
#     x2_his = x2_params[1:]
#     x3_los = x3_params[:-1]
#     x3_his = x3_params[1:]

#     n1 = x1_los.shape[0]
#     n2 = x2_los.shape[0]
#     n3 = x3_los.shape[0]

#     x1_lo_grid = jnp.broadcast_to(x1_los[:, None, None], (n1, n2, n3))
#     x1_hi_grid = jnp.broadcast_to(x1_his[:, None, None], (n1, n2, n3))
#     x2_lo_grid = jnp.broadcast_to(x2_los[None, :, None], (n1, n2, n3))
#     x2_hi_grid = jnp.broadcast_to(x2_his[None, :, None], (n1, n2, n3))
#     x3_lo_grid = jnp.broadcast_to(x3_los[None, None, :], (n1, n2, n3))
#     x3_hi_grid = jnp.broadcast_to(x3_his[None, None, :], (n1, n2, n3))

#     corners = jnp.stack(
#         [
#             jnp.stack([x1_lo_grid, x2_lo_grid, x3_lo_grid], axis=-1),
#             jnp.stack([x1_lo_grid, x2_lo_grid, x3_hi_grid], axis=-1),
#             jnp.stack([x1_lo_grid, x2_hi_grid, x3_lo_grid], axis=-1),
#             jnp.stack([x1_lo_grid, x2_hi_grid, x3_hi_grid], axis=-1),
#             jnp.stack([x1_hi_grid, x2_lo_grid, x3_lo_grid], axis=-1),
#             jnp.stack([x1_hi_grid, x2_lo_grid, x3_hi_grid], axis=-1),
#             jnp.stack([x1_hi_grid, x2_hi_grid, x3_lo_grid], axis=-1),
#             jnp.stack([x1_hi_grid, x2_hi_grid, x3_hi_grid], axis=-1),
#         ],
#         axis=-2,
#     )

#     # Push corners through closed-loop dynamics
#     flat_corners = corners.reshape((-1, 3))
#     flat_next = jax.vmap(cl_system_jax)(flat_corners)
#     x_next = flat_next.reshape(corners.shape)

#     # Parent AABB volumes (for axis-aligned cubes this equals cell volume)
#     parent_volume = (x1_hi_grid - x1_lo_grid) * (x2_hi_grid - x2_lo_grid) * (x3_hi_grid - x3_lo_grid)

#     # Image AABB volumes
#     x1_post = x_next[..., 0]
#     x2_post = x_next[..., 1]
#     x3_post = x_next[..., 2]
#     x1_lo_post = jnp.min(x1_post, axis=-1)
#     x1_hi_post = jnp.max(x1_post, axis=-1)
#     x2_lo_post = jnp.min(x2_post, axis=-1)
#     x2_hi_post = jnp.max(x2_post, axis=-1)
#     theta_span = theta_min_arc_length(x3_post)
#     img_volume = (x1_hi_post - x1_lo_post) * (x2_hi_post - x2_lo_post) * theta_span

#     ratio = img_volume / (parent_volume + 1e-12)
#     return jnp.sum(ratio)

# def succ_bound(
#     params,
#     *,
#     args,
#     p=10.0 # soft min smoothing factor
#     ):
#     """Derived, smooth upper bound to successor count.
#     """
    
#     n1_internal, n2_internal, n3_internal = args['shape']
#     x1_lo, x2_lo, x3_lo = args['domain_lb']
#     x1_hi, x2_hi, x3_hi = args['domain_ub']
#     L = args['L']

#     params = jnp.asarray(params)
#     u1 = params[:n1_internal]
#     u2 = params[n1_internal : n1_internal + n2_internal]
#     u3 = params[n1_internal + n2_internal : n1_internal + n2_internal + n3_internal]

#     x1_params = make_lines_from_gaps(u1, x1_lo, x1_hi)
#     x2_params = make_lines_from_gaps(u2, x2_lo, x2_hi)
#     x3_params = make_lines_from_gaps(u3, x3_lo, x3_hi)

#     gap1 = jnp.diff(x1_params)
#     gap2 = jnp.diff(x2_params)
#     gap3 = jnp.diff(x3_params)

#     L = jnp.asarray(L)
#     p = jnp.asarray(p, dtype=params.dtype)
#     eps = jnp.asarray(1e-12, dtype=params.dtype)

#     def soft_min_gap(gaps):
#         return jnp.power(jnp.sum(jnp.power(gaps + eps, -p)), -1.0 / p)

#     eta = jnp.stack([
#         soft_min_gap(gap1),
#         soft_min_gap(gap2),
#         soft_min_gap(gap3),
#     ])

#     gap_grid = jnp.stack(
#         jnp.meshgrid(gap1, gap2, gap3, indexing="ij"),
#         axis=-1,
#     )
#     diam = jnp.einsum("...j,kj->...k", gap_grid, L)
#     prod = jnp.prod(2.0 + diam / (eta + eps), axis=-1)

#     return jnp.mean(prod)


# def taylor_remainder(
#     params,
#     *,
#     args,
#     batch_size=4096,
#     ):
#     """
#     Sum the per-cell Taylor remainder proxy with batched JAX Hessians.
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

#     x1_widths = jnp.diff(x1_params)
#     x2_widths = jnp.diff(x2_params)
#     x3_widths = jnp.diff(x3_params)

#     x1_centers = 0.5 * (x1_params[:-1] + x1_params[1:])
#     x2_centers = 0.5 * (x2_params[:-1] + x2_params[1:])
#     x3_centers = 0.5 * (x3_params[:-1] + x3_params[1:])

#     centroids = jnp.stack(
#         jnp.meshgrid(x1_centers, x2_centers, x3_centers, indexing="ij"),
#         axis=-1,
#     ).reshape((-1, 3))
#     half_spans = 0.5 * jnp.stack(
#         jnp.meshgrid(x1_widths, x2_widths, x3_widths, indexing="ij"),
#         axis=-1,
#     )
#     max_displacements = jnp.linalg.norm(half_spans, axis=-1).reshape((-1,))

#     batch_size = int(batch_size)
#     num_cells = centroids.shape[0]
#     pad = (-num_cells) % batch_size
#     padded_num_cells = num_cells + pad

#     centroids = jnp.pad(centroids, ((0, pad), (0, 0)))
#     max_displacements = jnp.pad(max_displacements, (0, pad))
#     valid_mask = jnp.arange(padded_num_cells) < num_cells

#     centroid_batches = centroids.reshape((-1, batch_size, 3))
#     displacement_batches = max_displacements.reshape((-1, batch_size))
#     mask_batches = valid_mask.reshape((-1, batch_size))

#     def accumulate_batch(total, batch):
#         centroid_batch, displacement_batch, mask_batch = batch
#         hessians = _batched_cl_system_hessian(centroid_batch)
#         eigvals = jnp.linalg.eigvalsh(hessians)
#         spec_norms = jnp.max(jnp.abs(eigvals), axis=-1)
#         spec_norms = jnp.max(spec_norms, axis=-1)
#         contributions = 0.5 * spec_norms * displacement_batch
#         contributions = jnp.where(mask_batch, contributions, jnp.zeros_like(contributions))
#         return total + jnp.sum(contributions), None

#     total_taylor_remainder, _ = jax.lax.scan(
#         accumulate_batch,
#         jnp.asarray(0.0, dtype=params.dtype),
#         (centroid_batches, displacement_batches, mask_batches),
#     )

#     return total_taylor_remainder


# def noninflated_image_volume(
#     params,
#     *,
#     args,
#     batch_size=4096,
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

#     x1_widths = jnp.diff(x1_params)
#     x2_widths = jnp.diff(x2_params)
#     x3_widths = jnp.diff(x3_params)

#     x1_centers = 0.5 * (x1_params[:-1] + x1_params[1:])
#     x2_centers = 0.5 * (x2_params[:-1] + x2_params[1:])
#     x3_centers = 0.5 * (x3_params[:-1] + x3_params[1:])

#     centroids = jnp.stack(
#         jnp.meshgrid(x1_centers, x2_centers, x3_centers, indexing="ij"),
#         axis=-1,
#     ).reshape((-1, 3))
#     half_spans = 0.5 * jnp.stack(
#         jnp.meshgrid(x1_widths, x2_widths, x3_widths, indexing="ij"),
#         axis=-1,
#     ).reshape((-1, 3))

#     batch_size = int(batch_size)
#     num_cells = centroids.shape[0]
#     pad = (-num_cells) % batch_size
#     padded_num_cells = num_cells + pad

#     centroids = jnp.pad(centroids, ((0, pad), (0, 0)))
#     half_spans = jnp.pad(half_spans, ((0, pad), (0, 0)))
#     valid_mask = jnp.arange(padded_num_cells) < num_cells

#     centroid_batches = centroids.reshape((-1, batch_size, 3))
#     half_span_batches = half_spans.reshape((-1, batch_size, 3))
#     mask_batches = valid_mask.reshape((-1, batch_size))

#     def accumulate_batch(total, batch):
#         centroid_batch, half_span_batch, mask_batch = batch
#         jacobians = _batched_cl_system_jacobian(centroid_batch)
#         centers_next = jax.vmap(cl_system_jax)(centroid_batch)

#         corner_offsets = half_span_batch[:, None, :] * _corner_signs[None, :, :]
#         linearized_next_verts = centers_next[:, None, :] + jnp.einsum(
#             "nij,nvj->nvi",
#             jacobians,
#             corner_offsets,
#         )

#         next_lower_bounds = jnp.min(linearized_next_verts, axis=1)
#         next_upper_bounds = jnp.max(linearized_next_verts, axis=1)
#         side_lengths = next_upper_bounds - next_lower_bounds
#         volumes = jnp.prod(side_lengths, axis=-1)
#         volumes = jnp.where(mask_batch, volumes, jnp.zeros_like(volumes))
#         return total + jnp.sum(volumes), None

#     total_volume, _ = jax.lax.scan(
#         accumulate_batch,
#         jnp.asarray(0.0, dtype=params.dtype),
#         (centroid_batches, half_span_batches, mask_batches),
#     )

#     return total_volume


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
