# model_checking_tools.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# -----------------------------
# Numerics helpers (NumPy)
# -----------------------------

def softplus_np(x: np.ndarray) -> np.ndarray:
    """Stable softplus for NumPy."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def make_lines_from_gaps_np(u: np.ndarray, lo: float, hi: float, min_gap: float = 0.0) -> np.ndarray:
    """
    Mirrors your JAX make_lines_from_gaps:
      gaps_raw = softplus(u) + min_gap
      gaps = gaps_raw * (hi-lo)/sum(gaps_raw)
      lines = [lo, lo+cumsum(gaps)[:-1], hi]
    If len(u)=n, returns n+1 lines => n cells.
    """
    u = np.asarray(u, dtype=float)
    gaps_raw = softplus_np(u) + float(min_gap)
    total = float(np.sum(gaps_raw)) + 1e-12
    gaps = gaps_raw * ((hi - lo) / total)
    internal = lo + np.cumsum(gaps)[:-1]  # length n-1
    return np.concatenate([[lo], internal, [hi]]).astype(float)

def trans_matrix_np(theta: float, a1: float, a2: float, h: float) -> np.ndarray:
    """
    NumPy version of your trans_matrix: M = H @ S @ R, with s1=exp(a1), s2=exp(a2).
    """
    s1 = float(np.exp(a1))
    s2 = float(np.exp(a2))
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    S = np.array([[s1, 0.0],
                  [0.0, s2]], dtype=float)
    H = np.array([[1.0, float(h)],
                  [0.0, 1.0]], dtype=float)
    return H @ S @ R

def y_bounds_from_x_box_np(M: np.ndarray,
                           x1_min: float, x1_max: float,
                           x2_min: float, x2_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hard y-bounds from corners of X-box: y = Minv * x.
    Returns y_lo (2,), y_hi (2,)
    """
    Minv = np.linalg.inv(M)
    Xverts = np.array([
        [x1_min, x2_min],
        [x1_min, x2_max],
        [x1_max, x2_max],
        [x1_max, x2_min],
    ], dtype=float)
    Yverts = (Xverts @ Minv.T)
    y_lo = Yverts.min(axis=0)
    y_hi = Yverts.max(axis=0)
    return y_lo, y_hi

def dynamics_affine_np(x: np.ndarray, A: np.ndarray, center: np.ndarray) -> np.ndarray:
    """x_next = center + A @ (x-center). x can be (2,) or (N,2)."""
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    c = np.asarray(center, dtype=float)
    if x.ndim == 1:
        return c + A @ (x - c)
    return c + (x - c) @ A.T

def bbox_successor_ranges(y_lo: np.ndarray, y_hi: np.ndarray,
                          y1_lines: np.ndarray, y2_lines: np.ndarray) -> Tuple[range, range, bool]:
    """
    Given bbox [y_lo, y_hi] in y-space, compute index ranges of cells that intersect it.
    Returns (irange, jrange, outside_flag).
      outside_flag True iff bbox extends outside the y-grid span.
    """
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    y1_lines = np.asarray(y1_lines, dtype=float)
    y2_lines = np.asarray(y2_lines, dtype=float)

    n1 = len(y1_lines) - 1
    n2 = len(y2_lines) - 1

    y1_min, y1_max = y1_lines[0], y1_lines[-1]
    y2_min, y2_max = y2_lines[0], y2_lines[-1]

    outside = bool((y_lo[0] < y1_min) or (y_hi[0] > y1_max) or (y_lo[1] < y2_min) or (y_hi[1] > y2_max))

    # Clamp bbox to grid span for intersection range computation
    yy_lo = np.array([max(y_lo[0], y1_min), max(y_lo[1], y2_min)], dtype=float)
    yy_hi = np.array([min(y_hi[0], y1_max), min(y_hi[1], y2_max)], dtype=float)

    # If clamped bbox is empty in any dimension => no in-grid successors
    if (yy_lo[0] > yy_hi[0]) or (yy_lo[1] > yy_hi[1]):
        return range(0, 0), range(0, 0), outside

    i0 = max(0, np.searchsorted(y1_lines, yy_lo[0], side="right") - 1)
    i1 = min(n1, np.searchsorted(y1_lines, yy_hi[0], side="left"))
    j0 = max(0, np.searchsorted(y2_lines, yy_lo[1], side="right") - 1)
    j1 = min(n2, np.searchsorted(y2_lines, yy_hi[1], side="left"))

    return range(i0, i1), range(j0, j1), outside


# -----------------------------
# Abstraction + labeling
# -----------------------------

@dataclass
class Abstraction:
    adj: List[List[int]]             # adjacency list
    labels: List[set]                # AP labels per state
    n1: int
    n2: int
    y1_lines: np.ndarray
    y2_lines: np.ndarray
    M: np.ndarray
    has_out_state: bool
    out_state: Optional[int] = None  # index of OUT, if present

def idx(i: int, j: int, n2: int) -> int:
    return i * n2 + j

def cell_corners_y(y1_lines: np.ndarray, y2_lines: np.ndarray, i: int, j: int) -> np.ndarray:
    y1a, y1b = y1_lines[i], y1_lines[i+1]
    y2a, y2b = y2_lines[j], y2_lines[j+1]
    return np.array([[y1a, y2a],
                     [y1a, y2b],
                     [y1b, y2b],
                     [y1b, y2a]], dtype=float)

def cell_centers_y(y1_lines: np.ndarray, y2_lines: np.ndarray) -> np.ndarray:
    y1c = 0.5 * (y1_lines[:-1] + y1_lines[1:])
    y2c = 0.5 * (y2_lines[:-1] + y2_lines[1:])
    Yc = np.stack(np.meshgrid(y1c, y2c, indexing="ij"), axis=-1)  # (n1,n2,2)
    return Yc

def build_abstraction_from_params(
    params: Union[np.ndarray, Sequence[float]],
    *,
    A: np.ndarray,
    center: np.ndarray,
    x1_min: float, x1_max: float, x2_min: float, x2_max: float,
    n1_internal: int, n2_internal: int,
    # labeling
    ap_labelers: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    # domain/out handling
    add_out_state: bool = True,
    # gap controls
    min_gap_frac: float = 0.0,
) -> Abstraction:
    """
    Builds a *hard* Kripke-style abstraction from learned parameters.

    ap_labelers: dict of AP -> function(Xc, cell_poly_X) -> bool array
      - Xc: (n1,n2,2) cell centers in x-space
      - cell_poly_X: (n1,n2,4,2) corners in x-space
      - returns: (n1,n2) boolean mask for that AP
    """

    params = np.asarray(params, dtype=float)
    u1 = params[:n1_internal]
    u2 = params[n1_internal:n1_internal+n2_internal]
    theta, a1, a2, h = params[n1_internal+n2_internal:]

    # Transform + y-bounds induced from x-box
    M = trans_matrix_np(theta, a1, a2, h)
    Minv = np.linalg.inv(M)

    y_lo, y_hi = y_bounds_from_x_box_np(M, x1_min, x1_max, x2_min, x2_max)
    y1_lo, y2_lo = float(y_lo[0]), float(y_lo[1])
    y1_hi, y2_hi = float(y_hi[0]), float(y_hi[1])

    span1 = y1_hi - y1_lo
    span2 = y2_hi - y2_lo
    min_gap1 = float(min_gap_frac * span1 / n1_internal)
    min_gap2 = float(min_gap_frac * span2 / n2_internal)

    y1_lines = make_lines_from_gaps_np(u1, y1_lo, y1_hi, min_gap=min_gap1)
    y2_lines = make_lines_from_gaps_np(u2, y2_lo, y2_hi, min_gap=min_gap2)

    n1 = len(y1_lines) - 1  # should equal n1_internal
    n2 = len(y2_lines) - 1  # should equal n2_internal
    N = n1 * n2

    # Precompute centers/corners in x for labeling
    Yc = cell_centers_y(y1_lines, y2_lines)              # (n1,n2,2)
    Xc = (Yc @ M.T)                                      # (n1,n2,2)

    # corners in y then to x
    Xcorn = np.zeros((n1, n2, 4, 2), dtype=float)
    for i in range(n1):
        for j in range(n2):
            Yk = cell_corners_y(y1_lines, y2_lines, i, j)     # (4,2)
            Xcorn[i, j] = (Yk @ M.T)

    # Label APs
    labels = [set() for _ in range(N + (1 if add_out_state else 0))]
    ap_masks = {}
    for ap, fn in ap_labelers.items():
        mask = fn(Xc, Xcorn)  # (n1,n2) boolean
        ap_masks[ap] = mask

    for i in range(n1):
        for j in range(n2):
            s = idx(i, j, n2)
            for ap, mask in ap_masks.items():
                if bool(mask[i, j]):
                    labels[s].add(ap)

    # OUT state (optional)
    out_state = None
    if add_out_state:
        out_state = N
        labels[out_state] = set()  # usually no APs true
        # self-loop will be added later (keeps traces infinite)

    # Build transitions conservatively using hard bbox intersection in y-space
    adj: List[List[int]] = [[] for _ in range(N + (1 if add_out_state else 0))]

    A = np.asarray(A, dtype=float)
    center = np.asarray(center, dtype=float)

    for i in range(n1):
        for j in range(n2):
            s = idx(i, j, n2)

            # Map y-corners -> x, propagate -> y
            Yk = cell_corners_y(y1_lines, y2_lines, i, j)     # (4,2)
            Xk = (Yk @ M.T)                                   # (4,2)
            Xn = dynamics_affine_np(Xk, A=A, center=center)   # (4,2)
            Yn = (Xn @ Minv.T)                                # (4,2)

            # bbox in y
            y_lo_s = Yn.min(axis=0)
            y_hi_s = Yn.max(axis=0)

            # determine in-grid intersecting successors
            ir, jr, outside_y = bbox_successor_ranges(y_lo_s, y_hi_s, y1_lines, y2_lines)

            succ = []
            for ii in ir:
                for jj in jr:
                    succ.append(idx(ii, jj, n2))

            # Also mark outside-x transitions as OUT if desired (more meaningful than outside_y)
            outside_x = bool(
                (Xn[:, 0].min() < x1_min) or (Xn[:, 0].max() > x1_max) or
                (Xn[:, 1].min() < x2_min) or (Xn[:, 1].max() > x2_max)
            )

            if add_out_state and (outside_x or outside_y or len(succ) == 0):
                succ.append(out_state)

            # Guarantee at least one successor (LTL wants infinite traces)
            if len(succ) == 0:
                succ = [s]

            # Remove duplicates
            adj[s] = sorted(set(succ))

    # OUT self-loop
    if add_out_state and out_state is not None:
        adj[out_state] = [out_state]

    return Abstraction(
        adj=adj,
        labels=labels,
        n1=n1,
        n2=n2,
        y1_lines=y1_lines,
        y2_lines=y2_lines,
        M=M,
        has_out_state=add_out_state,
        out_state=out_state,
    )


# -----------------------------
# LTL checks (common patterns)
# -----------------------------

def tarjan_scc(adj: List[List[int]], nodes: Sequence[int]) -> List[List[int]]:
    """
    Tarjan SCC on induced subgraph of `nodes`.
    Returns list of SCCs (each SCC is list of nodes).
    """
    node_set = set(nodes)
    index = 0
    stack: List[int] = []
    onstack = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    sccs: List[List[int]] = []

    def strongconnect(v: int):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in adj[v]:
            if w not in node_set:
                continue
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in nodes:
        if v not in indices:
            strongconnect(v)

    return sccs

def reverse_reachable(adj: List[List[int]], targets: Sequence[int], allowed: Optional[set] = None) -> set:
    """
    Compute set of nodes that can reach any node in targets, optionally restricted to `allowed`.
    Uses reverse graph BFS/DFS.
    """
    n = len(adj)
    rev = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            rev[v].append(u)

    allowed_set = allowed if allowed is not None else set(range(n))

    seen = set()
    stack = [t for t in targets if t in allowed_set]
    for t in stack:
        seen.add(t)

    while stack:
        v = stack.pop()
        for u in rev[v]:
            if u in allowed_set and u not in seen:
                seen.add(u)
                stack.append(u)

    return seen

def check_AF_goal_all_states(adj: List[List[int]], is_goal: np.ndarray) -> np.ndarray:
    """
    Returns sat[s]=True iff s satisfies A(F goal) under adversarial nondeterminism,
    i.e., there is no infinite path that avoids goal forever.

    Algorithm:
      - Consider induced subgraph on non-goal states.
      - Find cyclic SCCs there (size>1 or self-loop): these support infinite goal-avoiding runs.
      - Any non-goal state that can reach a cyclic SCC within non-goal subgraph is violating.
      - Goal states satisfy trivially.
    """
    n = len(adj)
    is_goal = np.asarray(is_goal, dtype=bool)
    non_goal_nodes = [s for s in range(n) if not is_goal[s]]
    non_goal_set = set(non_goal_nodes)

    # SCCs in non-goal subgraph
    sccs = tarjan_scc(adj, non_goal_nodes)

    bad_cycle_nodes = []
    for comp in sccs:
        if len(comp) > 1:
            bad_cycle_nodes.extend(comp)
        else:
            v = comp[0]
            if v in adj[v]:
                bad_cycle_nodes.append(v)

    # states in non-goal subgraph that can reach a bad SCC within non-goal subgraph are violating
    bad_ng = reverse_reachable(adj, bad_cycle_nodes, allowed=non_goal_set) if bad_cycle_nodes else set()

    sat = np.ones(n, dtype=bool)
    for s in bad_ng:
        sat[s] = False
    # goal states remain True
    return sat

def check_AG_not_unsafe_all_states(adj: List[List[int]], is_unsafe: np.ndarray) -> np.ndarray:
    """
    sat[s]=True iff s satisfies A(G !unsafe) under adversarial nondeterminism,
    i.e., there is no path that ever reaches unsafe.
    With adversarial nondeterminism, this is equivalent to: unsafe is NOT reachable from s.
    """
    n = len(adj)
    is_unsafe = np.asarray(is_unsafe, dtype=bool)
    unsafe_nodes = [s for s in range(n) if is_unsafe[s]]

    if not unsafe_nodes:
        return np.ones(n, dtype=bool)

    bad = reverse_reachable(adj, unsafe_nodes)  # any predecessor can reach unsafe
    sat = np.ones(n, dtype=bool)
    for s in bad:
        sat[s] = False
    return sat

def check_A_until_reach_avoid_all_states(adj: List[List[int]],
                                        is_goal: np.ndarray,
                                        is_unsafe: np.ndarray) -> np.ndarray:
    """
    sat[s]=True iff s satisfies A((!unsafe) U goal) under adversarial nondeterminism.

    Violation happens if there exists a path that:
      - hits unsafe before goal, OR
      - avoids goal forever without hitting goal (i.e., stays in non-goal region forever).
    Conservative graph check:
      - Work in the subgraph of states where goal is false.
      - Within that subgraph, any cycle is an "avoid-goal forever" witness.
      - Also, reaching unsafe within that subgraph is an "unsafe before goal" witness.
      - A state is good iff it cannot reach either witness within the non-goal subgraph.
    """
    n = len(adj)
    is_goal = np.asarray(is_goal, dtype=bool)
    is_unsafe = np.asarray(is_unsafe, dtype=bool)

    non_goal_nodes = [s for s in range(n) if not is_goal[s]]
    non_goal_set = set(non_goal_nodes)

    # Unsafe witnesses that occur before reaching goal: unsafe nodes that are non-goal
    unsafe_ng = [s for s in non_goal_nodes if is_unsafe[s]]

    # Cyclic SCCs within non-goal subgraph (avoid-goal forever witnesses)
    sccs = tarjan_scc(adj, non_goal_nodes)
    cyclic = []
    for comp in sccs:
        if len(comp) > 1:
            cyclic.extend(comp)
        else:
            v = comp[0]
            if v in adj[v]:
                cyclic.append(v)

    witness = set(unsafe_ng) | set(cyclic)
    bad_ng = reverse_reachable(adj, list(witness), allowed=non_goal_set) if witness else set()

    sat = np.ones(n, dtype=bool)
    for s in bad_ng:
        sat[s] = False
    # goal states satisfy immediately
    return sat


# -----------------------------
# Public API
# -----------------------------

@dataclass
class ModelCheckResult:
    sat: np.ndarray               # (N,) boolean, satisfaction from each state
    ap_truth: Dict[str, np.ndarray]  # AP -> (N,) boolean
    abstraction: Abstraction

def model_check_tessellation(
    params: Union[np.ndarray, Sequence[float]],
    *,
    A: np.ndarray,
    center: np.ndarray,
    x1_min: float, x1_max: float, x2_min: float, x2_max: float,
    n1_internal: int, n2_internal: int,
    # AP definitions (simple defaults below)
    goal_center: Tuple[float, float] = (5.0, 5.0),
    goal_radius: float = 1.0,
    unsafe_center: Optional[Tuple[float, float]] = None,
    unsafe_radius: Optional[float] = None,
    # property choice
    spec_kind: str = "AF_goal",
    # discretization behavior
    add_out_state: bool = True,
    min_gap_frac: float = 0.0,
) -> ModelCheckResult:
    """
    Build hard abstraction from params and check an LTL property from each abstract cell.

    spec_kind options:
      - "AF_goal"        : A(F goal)
      - "AG_not_unsafe"  : A(G !unsafe)
      - "AU_avoid_unsafe_reach_goal": A((!unsafe) U goal)

    AP labeling is center-based by default (matches your training).
    You can replace labelers below with more conservative variants if needed.
    """

    gc = np.array(goal_center, dtype=float)
    gr2 = float(goal_radius) ** 2

    def goal_labeler(Xc: np.ndarray, Xcorn: np.ndarray) -> np.ndarray:
        # center-based: goal iff center is inside goal ball
        d2 = np.sum((Xc - gc) ** 2, axis=-1)
        return d2 <= gr2

    labelers: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {"goal": goal_labeler}

    if unsafe_center is not None and unsafe_radius is not None:
        uc = np.array(unsafe_center, dtype=float)
        ur2 = float(unsafe_radius) ** 2

        def unsafe_labeler(Xc: np.ndarray, Xcorn: np.ndarray) -> np.ndarray:
            d2 = np.sum((Xc - uc) ** 2, axis=-1)
            return d2 <= ur2

        labelers["unsafe"] = unsafe_labeler
    else:
        # if not provided, mark no unsafe states
        def unsafe_none(Xc: np.ndarray, Xcorn: np.ndarray) -> np.ndarray:
            return np.zeros(Xc.shape[:-1], dtype=bool)
        labelers["unsafe"] = unsafe_none

    absys = build_abstraction_from_params(
        params,
        A=A,
        center=center,
        x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max,
        n1_internal=n1_internal, n2_internal=n2_internal,
        ap_labelers=labelers,
        add_out_state=add_out_state,
        min_gap_frac=min_gap_frac,
    )

    N = len(absys.adj)
    is_goal = np.zeros(N, dtype=bool)
    is_unsafe = np.zeros(N, dtype=bool)
    for s in range(N):
        if "goal" in absys.labels[s]:
            is_goal[s] = True
        if "unsafe" in absys.labels[s]:
            is_unsafe[s] = True

    # Evaluate property from each state (treat all states initial by returning sat per state)
    if spec_kind == "AF_goal":
        sat = check_AF_goal_all_states(absys.adj, is_goal)
    elif spec_kind == "AG_not_unsafe":
        sat = check_AG_not_unsafe_all_states(absys.adj, is_unsafe)
    elif spec_kind == "AU_avoid_unsafe_reach_goal":
        sat = check_A_until_reach_avoid_all_states(absys.adj, is_goal, is_unsafe)
    else:
        raise ValueError(f"Unknown spec_kind: {spec_kind}")

    ap_truth = {
        "goal": is_goal,
        "unsafe": is_unsafe,
    }

    return ModelCheckResult(sat=sat, ap_truth=ap_truth, abstraction=absys)
