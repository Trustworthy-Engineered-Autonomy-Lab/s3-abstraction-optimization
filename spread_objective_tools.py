# Libraries
import numpy as np
import jax
import jax.numpy as jnp
from grid_plot_tools import get_yspace_bounds

CENTER = np.array([5.0, 5.0])

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

def dynamics(points_x: np.ndarray,
                     A: np.ndarray,
                     center: np.ndarray = CENTER) -> np.ndarray:
    """
    Vectorized version of: center + A @ (x - center)
    points_x: (N,2)
    """
    points_x = np.asarray(points_x, dtype=float)
    return center + (points_x - center) @ A.T

def cell_corners_from_lines(y1_lines: np.ndarray, y2_lines: np.ndarray) -> np.ndarray:
    """
    Build corners for every cell in the rectilinear grid defined by y1_lines, y2_lines.
    Returns corners_y with shape (Ncells, 4, 2) in consistent order:
      [ (y1_i,y2_j), (y1_i,y2_{j+1}), (y1_{i+1},y2_{j+1}), (y1_{i+1},y2_j) ]
    """
    y1 = np.asarray(y1_lines, dtype=float).copy()
    y2 = np.asarray(y2_lines, dtype=float).copy()
    y1.sort()
    y2.sort()

    n1 = len(y1) - 1
    n2 = len(y2) - 1
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Need at least 2 lines in each direction to form cells.")

    corners = []
    for i in range(n1):
        for j in range(n2):
            y1a, y1b = y1[i], y1[i+1]
            y2a, y2b = y2[j], y2[j+1]
            corners.append([
                [y1a, y2a],
                [y1a, y2b],
                [y1b, y2b],
                [y1b, y2a],
            ])
    return np.array(corners, dtype=float)

def spread_metric(points_x: np.ndarray, mode: str = "bbox_area") -> float:
    """
    points_x: (4,2) corners after propagation, in x-space.
    Returns a scalar spread. Pick a mode that matches what you mean by 'spread'.
    """
    P = np.asarray(points_x, dtype=float)
    if P.shape != (4, 2):
        raise ValueError("Expected points_x shape (4,2).")

    if mode in ("bbox_area", "bbox_diag", "bbox_sum"):
        mins = P.min(axis=0)
        maxs = P.max(axis=0)
        w = maxs - mins  # [width_x1, width_x2]
        if mode == "bbox_area":
            return float(w[0] * w[1])              # bbox area in x-space
        if mode == "bbox_diag":
            return float(np.linalg.norm(w))        # bbox diagonal length
        if mode == "bbox_sum":
            return float(w[0] + w[1])              # sum of side lengths

    if mode == "mean_pairwise":
        # Average Euclidean distance over all 6 pairs
        dsum = 0.0
        cnt = 0
        for a in range(4):
            for b in range(a+1, 4):
                dsum += np.linalg.norm(P[a] - P[b])
                cnt += 1
        return float(dsum / cnt)

    if mode == "trace_cov":
        # Trace of covariance ~ average squared spread around the mean
        mu = P.mean(axis=0)
        Q = P - mu
        cov = (Q.T @ Q) / 4.0
        return float(np.trace(cov))

    raise ValueError(f"Unknown spread mode: {mode}")

def quick_objective(y1_lines: np.ndarray,
                                  y2_lines: np.ndarray,
                                  theta: float, s1: float, s2: float, h: float,
                                  A: np.ndarray,
                                  center: np.ndarray = CENTER,
                                  spread_mode: str = "bbox_area",
                                  weight_mode: str = "uniform") -> float:
    """
    Average spread of propagated corners in x-space.

    weight_mode:
      - "uniform": equal weight per cell
      - "cell_area_y": weight by cell area in y-space (dy1*dy2)
      - "cell_area_x": weight by parallelogram area of the cell in x-space
    """
    M = trans_matrix(theta, s1, s2, h)

    # Build all cells in y
    corners_y = cell_corners_from_lines(y1_lines, y2_lines)  # (Ncells,4,2)

    total = 0.0
    wsum = 0.0

    y1 = np.sort(np.asarray(y1_lines, dtype=float))
    y2 = np.sort(np.asarray(y2_lines, dtype=float))
    n1 = len(y1) - 1
    n2 = len(y2) - 1

    idx = 0
    for i in range(n1):
        for j in range(n2):
            cy = corners_y[idx]  # (4,2)

            # Map to x: x = M y (no translation here)
            cx = (M @ cy.T).T  # (4,2)

            # Push corners through affine dynamics
            cx_next = dynamics(cx, A=A, center=center)  # (4,2)

            # Spread in x-space of the propagated corners
            s = spread_metric(cx_next, mode=spread_mode)

            # Weighting
            if weight_mode == "uniform":
                w = 1.0
            elif weight_mode == "cell_area_y":
                dy1 = y1[i+1] - y1[i]
                dy2 = y2[j+1] - y2[j]
                w = float(dy1 * dy2)
            elif weight_mode == "cell_area_x":
                # area of parallelogram spanned by two edges in x-space
                e1 = cx[3] - cx[0]
                e2 = cx[1] - cx[0]
                w = float(abs(np.linalg.det(np.stack([e1, e2], axis=1))))
            else:
                raise ValueError(f"Unknown weight_mode: {weight_mode}")

            total += w * s
            wsum += w
            idx += 1

    return float(total / (wsum + 1e-12))

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

def softmax_tau(x, tau):
    return tau * jax.scipy.special.logsumexp(x / tau, axis=-1)

def softmin_tau(x, tau):
    return -tau * jax.scipy.special.logsumexp(-x / tau, axis=-1)

def diff_objective(params, *, A, center, y1_lo, y1_hi, y2_lo, y2_hi,
              n1_internal, n2_internal, tau=0.05, min_gap=0.0):
    """
    params packs:
      u1: (n1_internal,)   -> y1 internal gaps
      u2: (n2_internal,)   -> y2 internal gaps
      theta, a1, a2, h: scalars
    """
    A = jnp.asarray(A)
    center = jnp.asarray(center)

    u1 = params[:n1_internal]
    u2 = params[n1_internal:n1_internal+n2_internal]
    theta, a1, a2, h = params[n1_internal+n2_internal:]

    M = trans_matrix(theta, a1, a2, h)

    y1 = make_lines_from_gaps(u1, y1_lo, y1_hi, min_gap=min_gap)
    y2 = make_lines_from_gaps(u2, y2_lo, y2_hi, min_gap=min_gap)

    # cell intervals
    y1a, y1b = y1[:-1], y1[1:]   # (n1,)
    y2a, y2b = y2[:-1], y2[1:]   # (n2,)

    # build corners (n1,n2,4,2) using broadcasting
    # order: (a,a), (a,b), (b,b), (b,a)
    Y00 = jnp.stack(jnp.meshgrid(y1a, y2a, indexing="ij"), axis=-1)  # (n1,n2,2)
    Y01 = jnp.stack(jnp.meshgrid(y1a, y2b, indexing="ij"), axis=-1)
    Y11 = jnp.stack(jnp.meshgrid(y1b, y2b, indexing="ij"), axis=-1)
    Y10 = jnp.stack(jnp.meshgrid(y1b, y2a, indexing="ij"), axis=-1)

    Ycorn = jnp.stack([Y00, Y01, Y11, Y10], axis=2)  # (n1,n2,4,2)

    # map to x: x = M y
    Xcorn = (Ycorn @ M.T)  # (n1,n2,4,2)

    # propagate: center + A (x-center)
    Xnext = center + (Xcorn - center) @ A.T

    # smooth bbox spread in x-space
    # max/min over the 4 corners, separately per dimension
    xmax = softmax_tau(Xnext, tau)     # (n1,n2,2)
    xmin = softmin_tau(Xnext, tau)     # (n1,n2,2)
    w = xmax - xmin                    # (n1,n2,2)

    # area spread
    spread = w[..., 0] * w[..., 1]     # (n1,n2)

    return jnp.mean(spread)

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

def _gaps_from_u(u, lo, hi, min_gap=0.0):
    """
    Returns the *scaled* gaps that sum to (hi-lo).
    This matches make_lines_from_gaps internally, but gives you the gaps explicitly.
    """
    gaps_raw = jax.nn.softplus(u) + min_gap          # positive
    scale = (hi - lo) / (jnp.sum(gaps_raw) + 1e-12)  # normalize total length
    return gaps_raw * scale                          # sums to hi-lo

def diff_objective_reg(params, *, A, center,
                       y1_lo, y1_hi, y2_lo, y2_hi,
                       n1_internal, n2_internal,
                       tau=0.05, min_gap=0.0,
                       # regularizer weights
                       lam_gap=1.0,
                       lam_det=10.0,
                       lam_cond=1e-2,
                       lam_shear=1e-1):
    """
    Same spread objective + regularizers to prevent degenerate grids.
    """
    A = jnp.asarray(A)
    center = jnp.asarray(center)

    u1 = params[:n1_internal]
    u2 = params[n1_internal:n1_internal+n2_internal]
    theta, a1, a2, h = params[n1_internal+n2_internal:]

    # Transformation
    M = trans_matrix(theta, a1, a2, h)

    # --- Build line locations (same as before) ---
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
    Xnext = center + (Xcorn - center) @ A.T

    xmax = softmax_tau(Xnext, tau)
    xmin = softmin_tau(Xnext, tau)
    w = xmax - xmin
    spread = w[..., 0] * w[..., 1]
    J_spread = jnp.mean(spread)

    # -----------------------
    # Regularizers (key part)
    # -----------------------

    # (1) Gap-uniformity penalty (prevents "pile-up" / giant cells)
    L1 = (y1_hi - y1_lo)
    L2 = (y2_hi - y2_lo)
    gaps1 = _gaps_from_u(u1, y1_lo, y1_hi, min_gap=min_gap)  # sums to L1
    gaps2 = _gaps_from_u(u2, y2_lo, y2_hi, min_gap=min_gap)  # sums to L2

    # Normalize so the penalty is scale-free
    g1 = gaps1 / (L1 + 1e-12)
    g2 = gaps2 / (L2 + 1e-12)
    target1 = 1.0 / n1_internal
    target2 = 1.0 / n2_internal
    R_gap = jnp.mean((g1 - target1)**2) + jnp.mean((g2 - target2)**2)

    # (2) Determinant / scale control: discourage collapsing or blowing up M
    # det(S) = exp(a1+a2) so keeping a1+a2 near 0 keeps det near 1
    R_det = (a1 + a2)**2

    # (3) Conditioning penalty: discourages needle-like/skewed transforms
    Minv = jnp.linalg.inv(M)
    R_cond = jnp.sum(M*M) + jnp.sum(Minv*Minv)  # ||M||_F^2 + ||M^{-1}||_F^2

    # (4) Shear penalty (optional but often stabilizes)
    R_shear = h*h

    J = (J_spread
         + lam_gap * R_gap
         + lam_det * R_det
         + lam_cond * R_cond
         + lam_shear * R_shear)

    return J