# Libraries
import jax
import jax.numpy as jnp

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

def softmax_tau_axis(x, tau, axis):
    return tau * jax.scipy.special.logsumexp(x / tau, axis=axis)

def softmin_tau_axis(x, tau, axis):
    return -tau * jax.scipy.special.logsumexp(-x / tau, axis=axis)

def softmax2(a, b, tau):
    # smooth max(a,b)
    a, b = jnp.broadcast_arrays(a, b)
    return tau * jax.scipy.special.logsumexp(jnp.stack([a, b], axis=-1) / tau, axis=-1)

def softmin2(a, b, tau):
    # smooth min(a,b)
    a, b = jnp.broadcast_arrays(a, b)
    return -tau * jax.scipy.special.logsumexp(-jnp.stack([a, b], axis=-1) / tau, axis=-1)

def gaps_from_u(u, lo, hi, min_gap=0.0):
    gaps_raw = jax.nn.softplus(u) + min_gap
    scale = (hi - lo) / (jnp.sum(gaps_raw) + 1e-12)
    return gaps_raw * scale  # sums to hi-lo

def soft_overlap_box_with_cells(src_lo, src_hi, cell_lo, cell_hi, tau_ov=0.02):
    """
    src_lo/src_hi: (N,2) bounding boxes
    cell_lo/cell_hi: (N,2) rectangles
    returns overlap_area: (N,N) soft overlap proxy
    """
    # Expand dims for pairwise (i,j)
    s_lo = src_lo[:, None, :]  # (N,1,2)
    s_hi = src_hi[:, None, :]
    c_lo = cell_lo[None, :, :] # (1,N,2)
    c_hi = cell_hi[None, :, :]

    # overlap length in dim d: max(0, min(hi) - max(lo))
    max_lo = softmax2(s_lo, c_lo, tau_ov)   # (N,N,2)
    min_hi = softmin2(s_hi, c_hi, tau_ov)   # (N,N,2)
    len_d  = jax.nn.softplus(min_hi - max_lo)  # (N,N,2), smooth ReLU

    return len_d[..., 0] * len_d[..., 1]  # (N,N)

def soft_adversarial_reach_objective(
    params, *,
    A, center,
    y1_lo, y1_hi, y2_lo, y2_hi,
    n1_internal, n2_internal,
    # goal / init definition (x-space balls, easy starter)
    goal_center=jnp.array([5.0, 5.0]), goal_radius=1.0,
    init_center=jnp.array([5.0, 5.0]), init_radius=8.0,
    # soft params
    tau_bb=0.03,    # for bbox (over corners)
    tau_ov=0.02,    # for overlap geometry
    tau_adv=0.15,   # for adversarial softmax over successors
    gamma=0.95,
    K=40,
    eps_w=1e-8,
    min_gap=0.0,
    # mild regularizers (optional, keep non-crazy but allow nonuniform)
    lam_det=5.0,
    lam_cond=1e-3,
    lam_gap_bounds=5.0,
    gap_ratio_max=5.0
):
    """
    Differentiable proxy for "eventually reach goal" under worst-case scheduler.

    Returns a scalar loss (lower is better).
    """
    A = jnp.asarray(A)
    center = jnp.asarray(center)
    goal_center = jnp.asarray(goal_center)
    init_center = jnp.asarray(init_center)

    # unpack params
    u1 = params[:n1_internal]
    u2 = params[n1_internal:n1_internal+n2_internal]
    theta, a1, a2, h = params[n1_internal+n2_internal:]

    # transform + inverse
    M = trans_matrix(theta, a1, a2, h)
    Minv = jnp.linalg.inv(M)

    # grid lines
    y1 = make_lines_from_gaps(u1, y1_lo, y1_hi, min_gap=min_gap)  # (n1+1,)
    y2 = make_lines_from_gaps(u2, y2_lo, y2_hi, min_gap=min_gap)  # (n2+1,)

    y1a, y1b = y1[:-1], y1[1:]   # (n1,)
    y2a, y2b = y2[:-1], y2[1:]   # (n2,)

    # corners per cell in y
    Y00 = jnp.stack(jnp.meshgrid(y1a, y2a, indexing="ij"), axis=-1)  # (n1,n2,2)
    Y01 = jnp.stack(jnp.meshgrid(y1a, y2b, indexing="ij"), axis=-1)
    Y11 = jnp.stack(jnp.meshgrid(y1b, y2b, indexing="ij"), axis=-1)
    Y10 = jnp.stack(jnp.meshgrid(y1b, y2a, indexing="ij"), axis=-1)
    Ycorn = jnp.stack([Y00, Y01, Y11, Y10], axis=2)                 # (n1,n2,4,2)

    # map to x, propagate in x, then map back to y
    Xcorn = Ycorn @ M.T                                             # (n1,n2,4,2)
    Xnext = center + (Xcorn - center) @ A.T                          # (n1,n2,4,2)
    Ynext = Xnext @ Minv.T                                           # (n1,n2,4,2)

    # bounding boxes of Ynext per source cell (smooth over the 4 corners)
    src_hi = softmax_tau_axis(Ynext, tau_bb, axis=2)                 # (n1,n2,2)
    src_lo = softmin_tau_axis(Ynext, tau_bb, axis=2)                 # (n1,n2,2)

    # target cell rectangles (lo/hi) in y for every cell
    # Build lo/hi arrays of shape (n1,n2,2)
    cell_lo = jnp.stack(jnp.meshgrid(y1a, y2a, indexing="ij"), axis=-1)
    cell_hi = jnp.stack(jnp.meshgrid(y1b, y2b, indexing="ij"), axis=-1)

    # flatten to (N,2)
    N = n1_internal * n2_internal
    src_lo_f = src_lo.reshape((N, 2))
    src_hi_f = src_hi.reshape((N, 2))
    cell_lo_f = cell_lo.reshape((N, 2))
    cell_hi_f = cell_hi.reshape((N, 2))

    # soft overlap area proxy (N,N)
    ov = soft_overlap_box_with_cells(src_lo_f, src_hi_f, cell_lo_f, cell_hi_f, tau_ov=tau_ov)

    # convert to soft successor weights W (row-stochastic)
    W = ov / (jnp.sum(ov, axis=1, keepdims=True) + eps_w)  # (N,N)

    # label goal + init using cell centers (in x-space)
    y1c = 0.5 * (y1a + y1b)
    y2c = 0.5 * (y2a + y2b)
    Yc = jnp.stack(jnp.meshgrid(y1c, y2c, indexing="ij"), axis=-1).reshape((N, 2))
    Xc = Yc @ M.T

    goal = (jnp.sum((Xc - goal_center)**2, axis=1) <= goal_radius**2).astype(jnp.float32)  # (N,)
    init = (jnp.sum((Xc - init_center)**2, axis=1) <= init_radius**2).astype(jnp.float32)  # (N,)
    init = init / (jnp.sum(init) + 1e-12)  # normalized weights

    # unrolled adversarial cost-to-go
    J = jnp.zeros((N,), dtype=jnp.float32)

    logW = jnp.log(W + eps_w)  # (N,N)
    not_goal = 1.0 - goal

    def bellman_step(Jcur):
        # worst-case successor: softmax over j of (J[j] + logW[i,j])
        adv = softmax_tau_axis(Jcur[None, :] + logW, tau_adv, axis=1)  # (N,)
        Jnew = not_goal * (1.0 + gamma * adv)  # keep goal at 0
        return Jnew

    for _ in range(K):
        J = bellman_step(J)

    # objective = expected cost from init distribution
    J_main = jnp.sum(init * J)

    # -------------------------
    # Regularizers (light-touch)
    # -------------------------

    # keep det near 1: det(S)=exp(a1+a2)
    R_det = (a1 + a2)**2

    # keep conditioning reasonable
    R_cond = jnp.sum(M*M) + jnp.sum(Minv*Minv)

    # prevent extreme gap ratios, but still allow nonuniform refinement
    g1 = gaps_from_u(u1, y1_lo, y1_hi, min_gap=min_gap)
    g2 = gaps_from_u(u2, y2_lo, y2_hi, min_gap=min_gap)
    g1m = jnp.mean(g1); g2m = jnp.mean(g2)
    r1 = g1 / (g1m + 1e-12)
    r2 = g2 / (g2m + 1e-12)
    # penalize ratios above gap_ratio_max (and below 1/gap_ratio_max)
    R_gap = jnp.mean(jax.nn.softplus(r1 - gap_ratio_max) + jax.nn.softplus((1.0/gap_ratio_max) - r1)) \
          + jnp.mean(jax.nn.softplus(r2 - gap_ratio_max) + jax.nn.softplus((1.0/gap_ratio_max) - r2))

    return J_main + lam_det * R_det + lam_cond * R_cond + lam_gap_bounds * R_gap
