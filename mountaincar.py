# Libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

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


# -----------------------------
# Differentiable (Torch) pieces
# -----------------------------

def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus for x>0."""
    return torch.log(torch.expm1(x))


def _smooth_clip(x: torch.Tensor, lo: float, hi: float, sharpness: float = 20.0) -> torch.Tensor:
    """Smooth approximation to clip using tanh."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return mid + half * torch.tanh(sharpness * (x - mid) / (half + 1e-12))


def mc_ol_dynamics_torch(state: torch.Tensor, action_cont: torch.Tensor) -> torch.Tensor:
    """Differentiable MountainCar open-loop dynamics with continuous action in [0,2].

    state: (...,2) tensor (p,v)
    action_cont: (...) tensor
    returns: (...,2)
    """
    p = state[..., 0]
    v = state[..., 1]

    v_next = v + 0.001 * (action_cont - 1.0) - 0.0025 * torch.cos(3.0 * p)
    v_next = _smooth_clip(v_next, -0.07, 0.07, sharpness=20.0)

    p_next = p + v_next
    p_next = _smooth_clip(p_next, -1.2, 0.6, sharpness=20.0)

    # Smooth-ish variant of: if p_next <= -1.2 and v_next < 0: v_next = 0
    hit_left = torch.sigmoid(200.0 * (-1.2 - p_next))
    going_left = torch.sigmoid(200.0 * (-v_next))
    v_next = v_next * (1.0 - hit_left * going_left)

    return torch.stack([p_next, v_next], dim=-1)


def policy_action_soft(state: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable closed-loop policy: expected action under softmax(Q/T)."""
    net = get_policy()
    q = net(state)
    pi = F.softmax(q / temperature, dim=-1)  # (...,3)
    actions = torch.tensor([0.0, 1.0, 2.0], device=state.device, dtype=state.dtype)
    return (pi * actions).sum(dim=-1)


def mc_cl_dynamics_soft(state: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    a = policy_action_soft(state, temperature=temperature)
    return mc_ol_dynamics_torch(state, a)


def _grid_edges_from_unconstrained(
    w: torch.Tensor,
    x_min: float,
    x_max: float,
    eps: float = 1e-6,
    delta_min: float = 0.0,
) -> torch.Tensor:
    """Parameterize a sorted grid with fixed endpoints via positive spacings.

    w has shape (n_lines-1,) representing spacings.
    """
    raw = F.softplus(w) + eps
    rng = float(x_max - x_min)

    if delta_min > 0.0:
        # Guarantee deltas >= delta_min exactly by allocating a fixed base length
        base = float(delta_min) * raw.numel()
        remaining = rng - base
        if remaining <= 0.0:
            raise ValueError(
                f"delta_min too large: need (n_lines-1)*delta_min < range, got {(raw.numel())}*{delta_min} >= {rng}"
            )
        raw_sum = raw.sum()
        deltas = delta_min + raw * (remaining / raw_sum)
    else:
        raw_sum = raw.sum()
        deltas = raw * (rng / raw_sum)

    edges = torch.cat([
        torch.tensor([x_min], device=w.device, dtype=w.dtype),
        x_min + torch.cumsum(deltas, dim=0),
    ])
    edges[-1] = x_max
    return edges


def _log_spacing_var(edges: torch.Tensor) -> torch.Tensor:
    """Scale-invariant uneven-spacing penalty: var(log(delta))."""
    deltas = torch.diff(edges).clamp_min(1e-12)
    return torch.var(torch.log(deltas), unbiased=False)


def _soft_bin_probs(x: torch.Tensor, edges: torch.Tensor, beta: float = 200.0) -> torch.Tensor:
    """Soft 1D binning against ordered edges.

    Returns probabilities over bins, shape (..., nbins) where nbins = len(edges)-1.
    """
    gate = torch.sigmoid(beta * (x[..., None] - edges[None, ...]))
    probs = gate[..., :-1] - gate[..., 1:]
    return probs.clamp_min(0.0)


def soft_successor_objective(
    edges_x: torch.Tensor,
    edges_y: torch.Tensor,
    temperature: float = 0.1,
    beta: float = 200.0,
    include_self: bool = True,
) -> torch.Tensor:
    """Differentiable proxy to average number of successor cells.

    Uses soft binning into destination cells + soft union over 4 mapped corners.
    """
    nbx = edges_x.numel() - 1
    nby = edges_y.numel() - 1
    n_cells = nbx * nby

    # Build all cells' corners in a vectorized way
    x0 = edges_x[:-1]
    x1 = edges_x[1:]
    y0 = edges_y[:-1]
    y1 = edges_y[1:]
    X0, Y0 = torch.meshgrid(x0, y0, indexing='ij')
    X1, _ = torch.meshgrid(x1, y0, indexing='ij')
    _, Y1 = torch.meshgrid(x0, y1, indexing='ij')

    # corners: (nbx,nby,4,2) -> (n_cells,4,2)
    corners = torch.stack([
        torch.stack([X0, Y0], dim=-1),
        torch.stack([X1, Y0], dim=-1),
        torch.stack([X0, Y1], dim=-1),
        torch.stack([X1, Y1], dim=-1),
    ], dim=2).reshape(n_cells, 4, 2)

    # Map all corners through differentiable closed-loop
    y = mc_cl_dynamics_soft(corners.reshape(-1, 2), temperature=temperature).reshape(n_cells, 4, 2)

    # Soft destination cell distribution per corner
    px = _soft_bin_probs(y[..., 0].reshape(-1), edges_x, beta=beta).reshape(n_cells, 4, nbx)
    py = _soft_bin_probs(y[..., 1].reshape(-1), edges_y, beta=beta).reshape(n_cells, 4, nby)
    pcell = (px[..., :, None] * py[..., None, :]).reshape(n_cells, 4, n_cells)

    # Soft union occupancy across 4 corners: occ = 1 - Π(1 - p)
    occ = 1.0 - torch.prod(1.0 - pcell + 1e-12, dim=1)  # (n_cells, n_cells)

    if include_self:
        # force each source cell to include itself (matches your hard meta objective option)
        self_idx = torch.arange(n_cells, device=edges_x.device)
        occ[self_idx, self_idx] = 1.0

    return occ.sum(dim=1).mean()


def soft_successor_occupancy(
    edges_x: torch.Tensor,
    edges_y: torch.Tensor,
    temperature: float = 0.1,
    beta: float = 200.0,
    include_self: bool = False,
    samples_per_cell: int = 0,
    include_corners: bool = True,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Return soft edge-occupancy matrix occ[src_cell, dst_cell] in [0,1]."""
    nbx = edges_x.numel() - 1
    nby = edges_y.numel() - 1
    n_cells = nbx * nby

    x0 = edges_x[:-1]
    x1 = edges_x[1:]
    y0 = edges_y[:-1]
    y1 = edges_y[1:]
    X0, Y0 = torch.meshgrid(x0, y0, indexing='ij')
    X1, _ = torch.meshgrid(x1, y0, indexing='ij')
    _, Y1 = torch.meshgrid(x0, y1, indexing='ij')

    points = []
    if include_corners:
        corners = torch.stack([
            torch.stack([X0, Y0], dim=-1),
            torch.stack([X1, Y0], dim=-1),
            torch.stack([X0, Y1], dim=-1),
            torch.stack([X1, Y1], dim=-1),
        ], dim=2).reshape(n_cells, 4, 2)
        points.append(corners)

    if samples_per_cell > 0:
        dx = (X1 - X0).clamp_min(1e-12)
        dy = (Y1 - Y0).clamp_min(1e-12)
        # u,v are treated as constants w.r.t. edges; points are affine in edges -> differentiable.
        ux = torch.rand((nbx, nby, samples_per_cell), device=edges_x.device, dtype=edges_x.dtype, generator=rng)
        uy = torch.rand((nbx, nby, samples_per_cell), device=edges_x.device, dtype=edges_x.dtype, generator=rng)
        sx = (X0[..., None] + ux * dx[..., None])
        sy = (Y0[..., None] + uy * dy[..., None])
        samples = torch.stack([sx, sy], dim=-1).reshape(n_cells, samples_per_cell, 2)
        points.append(samples)

    if not points:
        raise ValueError("soft_successor_occupancy: need include_corners=True or samples_per_cell>0")

    pts = torch.cat(points, dim=1)  # (n_cells, P, 2)

    y = mc_cl_dynamics_soft(pts.reshape(-1, 2), temperature=temperature).reshape(n_cells, -1, 2)

    P = y.shape[1]
    px = _soft_bin_probs(y[..., 0].reshape(-1), edges_x, beta=beta).reshape(n_cells, P, nbx)
    py = _soft_bin_probs(y[..., 1].reshape(-1), edges_y, beta=beta).reshape(n_cells, P, nby)
    pcell = (px[..., :, None] * py[..., None, :]).reshape(n_cells, P, n_cells)

    occ = 1.0 - torch.prod(1.0 - pcell + 1e-12, dim=1)  # (n_cells, n_cells)

    if include_self:
        self_idx = torch.arange(n_cells, device=edges_x.device)
        occ[self_idx, self_idx] = 1.0

    return occ


def soft_goal_indicator_from_edges(edges_x: torch.Tensor, nby: int, p_goal: float = 0.5, sharpness: float = 80.0) -> torch.Tensor:
    """Per-cell goal indicator in [0,1] based on left x-edge (underapprox).

    Cell is definitely goal if x_left >= p_goal. This uses a smooth step.
    """
    x_left = edges_x[:-1]  # (nbx,)
    g_x = torch.sigmoid(sharpness * (x_left - p_goal))  # (nbx,)
    nbx = x_left.numel()
    g = g_x[:, None].expand(nbx, nby).reshape(-1)  # (n_cells,)
    return g


def cell_areas_from_edges(edges_x: torch.Tensor, edges_y: torch.Tensor) -> torch.Tensor:
    """Compute per-cell areas for a rectilinear grid, shape (n_cells,)."""
    dx = torch.diff(edges_x)  # (nbx,)
    dy = torch.diff(edges_y)  # (nby,)
    A = (dx[:, None] * dy[None, :]).reshape(-1)
    return A


def soft_must_reach_value_iteration(
    occ: torch.Tensor,
    goal: torch.Tensor,
    iters: int = 40,
    tau: float = 40.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a differentiable approximation to "must eventually reach goal".

    Boolean analogue: V_i = goal_i OR (AND_{j in Succ(i)} V_j).
    We approximate AND/min via a weighted soft-min over successor values.

    occ: (n_cells,n_cells) soft edge existence in [0,1]
    goal: (n_cells,) in [0,1]
    """
    n = occ.shape[0]
    V = torch.zeros(n, device=occ.device, dtype=occ.dtype)

    w = occ.clamp_min(0.0) + eps

    for _ in range(iters):
        # soft-min over successors: sum_j pi_ij * V_j, where pi ∝ w_ij * exp(-tau * V_j)
        logits = torch.log(w) - tau * V[None, :]
        logits = logits - torch.max(logits, dim=1, keepdim=True).values  # stabilize
        pi = torch.exp(logits)
        pi = pi / (pi.sum(dim=1, keepdim=True) + eps)
        worst_next = (pi * V[None, :]).sum(dim=1)

        V = goal + (1.0 - goal) * worst_next

    return V


def soft_must_reach_objective(
    edges_x: torch.Tensor,
    edges_y: torch.Tensor,
    temperature: float = 0.1,
    beta: float = 200.0,
    p_goal: float = 0.5,
    goal_sharpness: float = 80.0,
    vi_iters: int = 40,
    vi_tau: float = 40.0,
    weight_mode: str = "area",
    samples_per_cell: int = 0,
    include_corners: bool = True,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Loss = 1 - avg(V), where V approximates must-reach(goal).

    IMPORTANT: default `weight_mode="area"` prevents a key failure mode where the
    optimizer inflates the objective by carving many tiny cells in easy/goal regions.
    """
    nbx = edges_x.numel() - 1
    nby = edges_y.numel() - 1
    occ = soft_successor_occupancy(
        edges_x,
        edges_y,
        temperature=temperature,
        beta=beta,
        include_self=False,
        samples_per_cell=samples_per_cell,
        include_corners=include_corners,
        rng=rng,
    )
    goal = soft_goal_indicator_from_edges(edges_x, nby=nby, p_goal=p_goal, sharpness=goal_sharpness)
    V = soft_must_reach_value_iteration(occ, goal, iters=vi_iters, tau=vi_tau)

    if weight_mode == "uniform":
        avgV = V.mean()
    elif weight_mode == "area":
        w = cell_areas_from_edges(edges_x, edges_y).clamp_min(1e-12)
        avgV = (w * V).sum() / w.sum()
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    return 1.0 - avgV


def optimize_grid_soft_successors(
    n_lines_x: int = 22,
    n_lines_y: int = 22,
    iters: int = 300,
    lr: float = 0.05,
    temperature: float = 0.1,
    beta: float = 200.0,
    print_every: int = 25,
    min_spacing_frac: float = 0.02,
    log_spacing_weight: float = 0.05,
):
    """Optimize grid line locations using a differentiable soft successor-count surrogate."""
    x_min, x_max = -1.2, 0.6
    y_min, y_max = -0.07, 0.07

    # Initialize from uniform grid
    edges_x0 = torch.linspace(x_min, x_max, n_lines_x, device=_DEVICE)
    edges_y0 = torch.linspace(y_min, y_max, n_lines_y, device=_DEVICE)

    dx0 = torch.diff(edges_x0).clamp_min(1e-6)
    dy0 = torch.diff(edges_y0).clamp_min(1e-6)
    wx = torch.nn.Parameter(_softplus_inverse(dx0))
    wy = torch.nn.Parameter(_softplus_inverse(dy0))

    opt = torch.optim.Adam([wx, wy], lr=lr)

    loss_hist = []
    hard_meta_hist = []

    # Warm policy cache
    _ = get_policy()

    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        uniform_dx = (x_max - x_min) / (n_lines_x - 1)
        uniform_dy = (y_max - y_min) / (n_lines_y - 1)
        delta_min_x = float(min_spacing_frac) * float(uniform_dx)
        delta_min_y = float(min_spacing_frac) * float(uniform_dy)

        edges_x = _grid_edges_from_unconstrained(wx, x_min, x_max, delta_min=delta_min_x)
        edges_y = _grid_edges_from_unconstrained(wy, y_min, y_max, delta_min=delta_min_y)

        loss = soft_successor_objective(edges_x, edges_y, temperature=temperature, beta=beta, include_self=True)

        # Even-spacing regularizer to discourage line collapse / sandwiching
        reg = _log_spacing_var(edges_x) + _log_spacing_var(edges_y)
        total = loss + float(log_spacing_weight) * reg

        total.backward()
        opt.step()

        loss_hist.append(float(loss.detach().cpu().item()))

        if (it % print_every) == 0 or it == iters - 1:
            ex = edges_x.detach().cpu().numpy().tolist()
            ey = edges_y.detach().cpu().numpy().tolist()
            hard_meta = meta_objective_function(ex, ey)
            hard_meta_hist.append((it, float(hard_meta)))
            print(f"Iter {it:5d} | soft successors = {loss_hist[-1]:.6f} | hard meta = {hard_meta:.6f}")

    uniform_dx = (x_max - x_min) / (n_lines_x - 1)
    uniform_dy = (y_max - y_min) / (n_lines_y - 1)
    delta_min_x = float(min_spacing_frac) * float(uniform_dx)
    delta_min_y = float(min_spacing_frac) * float(uniform_dy)
    final_edges_x = _grid_edges_from_unconstrained(wx, x_min, x_max, delta_min=delta_min_x).detach().cpu().numpy().tolist()
    final_edges_y = _grid_edges_from_unconstrained(wy, y_min, y_max, delta_min=delta_min_y).detach().cpu().numpy().tolist()
    return final_edges_x, final_edges_y, loss_hist, hard_meta_hist


def optimize_grid_soft_reachability(
    n_lines_x: int = 22,
    n_lines_y: int = 22,
    iters: int = 1000,
    lr: float = 0.05,
    temperature: float = 0.1,
    beta: float = 200.0,
    print_every: int = 200,
    min_spacing_frac: float = 0.03,
    log_spacing_weight: float = 0.08,
    p_goal: float = 0.5,
    goal_sharpness: float = 80.0,
    vi_iters: int = 40,
    vi_tau: float = 40.0,
    weight_mode: str = "area",
    samples_per_cell: int = 16,
    include_corners: bool = True,
    sample_seed: int = 0,
):
    """Optimize grid lines to maximize a soft must-reach(goal) value function."""
    x_min, x_max = -1.2, 0.6
    y_min, y_max = -0.07, 0.07

    edges_x0 = torch.linspace(x_min, x_max, n_lines_x, device=_DEVICE)
    edges_y0 = torch.linspace(y_min, y_max, n_lines_y, device=_DEVICE)

    dx0 = torch.diff(edges_x0).clamp_min(1e-6)
    dy0 = torch.diff(edges_y0).clamp_min(1e-6)
    wx = torch.nn.Parameter(_softplus_inverse(dx0))
    wy = torch.nn.Parameter(_softplus_inverse(dy0))

    opt = torch.optim.Adam([wx, wy], lr=lr)

    loss_hist = []
    hard_meta_hist = []

    _ = get_policy()

    uniform_dx = (x_max - x_min) / (n_lines_x - 1)
    uniform_dy = (y_max - y_min) / (n_lines_y - 1)
    delta_min_x = float(min_spacing_frac) * float(uniform_dx)
    delta_min_y = float(min_spacing_frac) * float(uniform_dy)

    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        edges_x = _grid_edges_from_unconstrained(wx, x_min, x_max, delta_min=delta_min_x)
        edges_y = _grid_edges_from_unconstrained(wy, y_min, y_max, delta_min=delta_min_y)

        rng = torch.Generator(device=_DEVICE)
        rng.manual_seed(int(sample_seed) + int(it))

        loss = soft_must_reach_objective(
            edges_x,
            edges_y,
            temperature=temperature,
            beta=beta,
            p_goal=p_goal,
            goal_sharpness=goal_sharpness,
            vi_iters=vi_iters,
            vi_tau=vi_tau,
            weight_mode=weight_mode,
            samples_per_cell=samples_per_cell,
            include_corners=include_corners,
            rng=rng,
        )

        reg = _log_spacing_var(edges_x) + _log_spacing_var(edges_y)
        total = loss + float(log_spacing_weight) * reg

        total.backward()
        opt.step()

        loss_hist.append(float(loss.detach().cpu().item()))

        if (it % print_every) == 0 or it == iters - 1:
            ex = edges_x.detach().cpu().numpy().tolist()
            ey = edges_y.detach().cpu().numpy().tolist()
            # Hard meta is slow; sample it sparsely.
            hard_meta = meta_objective_function(ex, ey)
            hard_meta_hist.append((it, float(hard_meta)))
            print(f"Iter {it:5d} | reachability loss = {loss_hist[-1]:.6f} | hard meta = {hard_meta:.6f}")

    final_edges_x = _grid_edges_from_unconstrained(wx, x_min, x_max, delta_min=delta_min_x).detach().cpu().numpy().tolist()
    final_edges_y = _grid_edges_from_unconstrained(wy, y_min, y_max, delta_min=delta_min_y).detach().cpu().numpy().tolist()
    return final_edges_x, final_edges_y, loss_hist, hard_meta_hist


def plot_training_history(
    soft_hist,
    hard_hist,
    meta0=None,
    meta_final=None,
    save_path: str = "training_history.png",
    show: bool = True,
    soft_label: str = "soft objective",
    hard_label: str = "hard meta (sampled)",
):
    """Plot optimization history.

    soft_hist: list[float] for every iteration
    hard_hist: list[tuple[int,float]] sampled (iter, hard_meta)
    """
    soft_hist = np.asarray(soft_hist, dtype=float)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    ax1.plot(np.arange(len(soft_hist)), soft_hist, color='tab:blue', linewidth=2.0, label=soft_label)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(soft_label, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    if hard_hist is not None and len(hard_hist) > 0:
        iters = np.asarray([t[0] for t in hard_hist], dtype=int)
        metas = np.asarray([t[1] for t in hard_hist], dtype=float)
        ax2.plot(iters, metas, color='tab:orange', linewidth=2.0, marker='o', markersize=4, label=hard_label)

    if meta0 is not None:
        ax2.axhline(float(meta0), color='tab:orange', linestyle='--', linewidth=1.5, alpha=0.7, label='hard meta (initial)')
    if meta_final is not None:
        ax2.axhline(float(meta_final), color='tab:orange', linestyle=':', linewidth=1.8, alpha=0.9, label='hard meta (final)')

    ax2.set_ylabel('Hard meta successors', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # One combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles1 or handles2:
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='best', framealpha=0.9)

    title = 'Grid Optimization History'
    if meta0 is not None and meta_final is not None:
        title += f" | hard meta: {float(meta0):.3f} → {float(meta_final):.3f}"
    ax1.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved training plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_grid(
    edges_x,
    edges_y,
    title: str = "Final Grid",
    save_path: str = "final_grid.png",
    show: bool = True,
):
    """Plot a rectilinear grid given x/y edges."""
    edges_x = np.asarray(edges_x, dtype=float)
    edges_y = np.asarray(edges_y, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, x in enumerate(edges_x):
        is_boundary = (i == 0) or (i == len(edges_x) - 1)
        ax.axvline(x, color=('tab:red' if is_boundary else 'tab:blue'), linewidth=(2.5 if is_boundary else 1.0), alpha=0.85)

    for j, y in enumerate(edges_y):
        is_boundary = (j == 0) or (j == len(edges_y) - 1)
        ax.axhline(y, color=('tab:red' if is_boundary else 'tab:green'), linewidth=(2.5 if is_boundary else 1.0), alpha=0.85)

    ax.set_xlim(edges_x[0], edges_x[-1])
    ax.set_ylim(edges_y[0], edges_y[-1])
    ax.set_xlabel('position (p)')
    ax.set_ylabel('velocity (v)')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved grid plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

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
        return int(q.argmax(dim=1).item())  # 0,1,2

# Closed-loop dynamics using greedy policy (no need to pass network each call)
def mc_cl_dynamics(x):
    x_np = np.asarray(x, dtype=np.float64)
    a_idx = policy_action(x_np)
    return mc_ol_dynamics(x_np, a_idx)

# Approximate Jacobian of closed-loop dynamics (holds action fixed)
def approx_jacobian_mc_cl(x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    a = policy_action(x)  # action chosen at base state
    base = mc_ol_dynamics(x, a)
    J = np.zeros((2, 2))
    for k in range(2):
        x_pert = x.copy()
        x_pert[k] += eps
        pert = mc_ol_dynamics(x_pert, a)  # keep same action to avoid discontinuities
        J[:, k] = (pert - base) / eps
    return J

# A "meta" objective function counting average number of successor cells
def _point_to_cell(x, y, params1, params2):
    if x < params1[0] or x > params1[-1] or y < params2[0] or y > params2[-1]:
        return None
    i = np.searchsorted(params1, x, side='right') - 1
    j = np.searchsorted(params2, y, side='right') - 1
    # Ensure indices refer to a valid cell (exclude last line index)
    if i < 0 or i >= len(params1) - 1 or j < 0 or j >= len(params2) - 1:
        return None
    return (i, j)
def meta_objective_function(params1, params2, include_self=True, ignore_out_of_bounds=True):
    params1 = np.asarray(params1, dtype=float)
    params2 = np.asarray(params2, dtype=float)
    n_cells = (len(params1) - 1) * (len(params2) - 1)
    if n_cells == 0:
        return 0.0

    total_successors = 0

    for i in range(len(params1) - 1):
        for j in range(len(params2) - 1):
            corners = np.array([
                [params1[i],   params2[j]],
                [params1[i+1], params2[j]],
                [params1[i],   params2[j+1]],
                [params1[i+1], params2[j+1]],
            ])
            transformed = np.array([mc_cl_dynamics(c) for c in corners])  # (4,2)

            successor_cells = set()
            if include_self:
                successor_cells.add((i, j))

            for (tx, ty) in transformed:
                cell = _point_to_cell(tx, ty, params1, params2)
                if cell is None:
                    if ignore_out_of_bounds:
                        continue
                    else:
                        continue  # placeholder for potential future handling
                successor_cells.add(cell)

            total_successors += len(successor_cells)

    return total_successors / n_cells

# Differentiable replacement of the desired objective
def objective_function(params1, params2):
    cost = 0.0
    for i in range(len(params1) - 1):
        for j in range(len(params2) - 1):
            corners = np.array([
                [params1[i],   params2[j]],
                [params1[i+1], params2[j]],
                [params1[i],   params2[j+1]],
                [params1[i+1], params2[j+1]],
            ])
            transformed = np.array([mc_cl_dynamics(c) for c in corners])   # shape (4, 2)
            avg = transformed.mean(axis=0)                           # shape (2,)
            sq_dists = np.sum((transformed - avg)**2, axis=1)        # shape (4,)
            cost += 0.5 * sq_dists.sum()
    return cost





if __name__ == "__main__":
    # Initialize the abstraction grid
    x1min, x1max = -1.2, 0.6
    x2min, x2max = -0.07, 0.07
    params1 = sorted([x1min] + [np.random.uniform(x1min, x1max) for _ in range(20)] + [x1max])
    params2 = sorted([x2min] + [np.random.uniform(x2min, x2max) for _ in range(20)] + [x2max])
    # params1 = np.linspace(x1min, x1max, 22).tolist()
    # params2 = np.linspace(x2min, x2max, 22).tolist()

    # Baselines (hard, true evaluation)
    print("Device:", _DEVICE)
    cost0 = objective_function(params1, params2)
    meta0 = meta_objective_function(params1, params2)
    print("Initial surrogate cost (variance):", cost0)
    print("Initial hard meta cost (successors):", meta0)

    # Recommended optimization: differentiable must-reach proxy (soft value iteration)
    final_x, final_y, soft_hist, hard_hist = optimize_grid_soft_reachability(
        n_lines_x=len(params1),
        n_lines_y=len(params2),
        iters=2000,
        lr=0.08,
        temperature=0.1,
        beta=200.0,
        print_every=200,
        min_spacing_frac=0.03,
        log_spacing_weight=0.08,
        p_goal=0.5,
        goal_sharpness=80.0,
        vi_iters=40,
        vi_tau=40.0,
        weight_mode="area",
        samples_per_cell=16,
        include_corners=True,
        sample_seed=0,
    )
    meta_final = meta_objective_function(final_x, final_y)
    print("\nFinal hard meta cost (successors):", meta_final)
    print("Final params1:", final_x)
    print("Final params2:", final_y)

    plot_training_history(
        soft_hist,
        hard_hist,
        meta0=meta0,
        meta_final=meta_final,
        save_path="training_history.png",
        show=True,
        soft_label="reachability loss (1-mean V)",
        hard_label="hard meta successors (sampled)",
    )
    plot_grid(final_x, final_y, title="Final Grid Partition", save_path="final_grid.png", show=True)

