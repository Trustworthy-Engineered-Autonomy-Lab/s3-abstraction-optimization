# cegar_loop.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from abstraction import Abstraction, Rect, AffineDynamics

# rectangles
def rect_intersection(a: Rect, b: Rect) -> Optional[Rect]:
    xmin = max(a.xmin, b.xmin)
    xmax = min(a.xmax, b.xmax)
    ymin = max(a.ymin, b.ymin)
    ymax = min(a.ymax, b.ymax)
    if xmin > xmax or ymin > ymax:
        return None
    return Rect(xmin, xmax, ymin, ymax)

def rect_width(r: Rect) -> float:
    return r.xmax - r.xmin

def rect_height(r: Rect) -> float:
    return r.ymax - r.ymin

def rect_depth(r: Rect) -> float:
    """Theta (z) extent of a rect."""
    return r.zmax - r.zmin

# Spot

def _compute_reachable_from(absys: Abstraction, init_uids: Set[int]) -> Set[int]:
    """
    Forward BFS
    """
    reachable: Set[int] = set()
    seed = [u for u in init_uids if u == absys.OUT_UID or u in absys.part.leaves]
    queue: deque = deque(seed)
    reachable.update(seed)

    while queue:
        u = queue.popleft()
        if u == absys.OUT_UID:
            continue  # OUT_UID has no outgoing transitions to expand
        by_a = absys.tr.succ.get(u, {})
        succs: Set[int] = set()
        for a, s in by_a.items():
            succs |= s
        for v in succs:
            if v not in reachable:
                reachable.add(v)
                queue.append(v)

    reachable.add(absys.OUT_UID)
    return reachable


def build_spot_kripke(
    absys: Abstraction,
    init_uids: Set[int],
    merge_actions: bool = True,
):
    """
    Build a Spot Kripke structure from absys
    """
    import spot  # type: ignore
    from buddy import bdd_ithvar, bddtrue  # type: ignore

    all_aps, labels_by_uid = absys.aps_and_labels()

    d = spot.make_bdd_dict()
    k = spot.make_kripke_graph(d)

    # Register APs and create BDD vars
    ap_to_bdd = {ap: bdd_ithvar(k.register_ap(ap)) for ap in sorted(all_aps)}

    # Restrict to the subgraph reachable from init_uids.
    reachable_uids = _compute_reachable_from(absys, init_uids)
    uids = sorted(reachable_uids)

    uid_to_sid: Dict[int, int] = {}
    sid_to_uid: Dict[int, int] = {}
    state_names: List[str] = []

    aps_sorted = sorted(all_aps)

    for u in uids:
        b = bddtrue
        labs = labels_by_uid.get(u, set())

        # force all missing APs to false so that unsafe=true doesnt get inverted by spot
        for ap in aps_sorted:
            v = ap_to_bdd[ap]
            if ap in labs:
                b &= v
            else:
                b &= -v

        sid = k.new_state(b)
        uid_to_sid[u] = sid
        sid_to_uid[sid] = u

        state_names.append(str(u))

    # Give Spot readable names for these states.
    k.set_state_names(state_names)

    # Initial state(s)
    if len(init_uids) == 1:

        valid_inits = [u for u in init_uids if u in uid_to_sid]
        if not valid_inits:
            # all init ids are stale -> caller should skip this run
            return None, None
        k.set_init_state(uid_to_sid[valid_inits[0]])

    else:
        init_sid = k.new_state(bddtrue)
        k.set_init_state(init_sid)
        for u in init_uids:
            if u in uid_to_sid:
                k.new_edge(init_sid, uid_to_sid[u])

    for u in uids:
        if u == absys.OUT_UID:
            continue
        by_a = absys.tr.succ.get(u, {})
        su = uid_to_sid[u]
        if merge_actions:
            dsts: Set[int] = set().union(*by_a.values()) if by_a else set()
            for v in dsts:
                if v in uid_to_sid:
                    k.new_edge(su, uid_to_sid[v])
        else:
            dsts = set().union(*by_a.values()) if by_a else set()
            for v in dsts:
                if v in uid_to_sid:
                    k.new_edge(su, uid_to_sid[v])

    # return k, d
    return k, d


def _parse_spot_run_to_lasso(run_str: str) -> Tuple[List[int], List[int]]:
    """
    Parse Spot's textual run format into (prefix_uids, cycle_uids)
    """
    prefix: List[int] = []
    cycle: List[int] = []

    section: Optional[str] = None
    for raw in run_str.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Prefix:"):
            section = "prefix"
            continue
        if line.startswith("Cycle:"):
            section = "cycle"
            continue
        if line.startswith("|"):
            continue  # label line

        if section in ("prefix", "cycle"):
            # First token is the state name
            tok = line.split()[0]
            try:
                uid = int(tok)
            except ValueError:
                # e.g., init hub state with numeric spot id or unnamed
                continue
            if section == "prefix":
                prefix.append(uid)
            else:
                cycle.append(uid)

    return prefix, cycle


def spot_get_counterexample_lasso(
    absys: Abstraction,
    init_uids: Set[int],
    phi: str,
    merge_actions: bool = True,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Return (prefix, cycle) as lists of abstract uids if phi is violated,
    otherwise return None.
    """
    import spot  # type: ignore

    k, d = build_spot_kripke(absys, init_uids, merge_actions=merge_actions)
    if k is None:
        return None


    # Translate the negation of the formula using the same dict
    f = spot.formula(phi)
    not_phi = spot.formula.Not(f)

    aut = spot.translate(not_phi, dict=d)

    run = k.intersecting_run(aut)
    if not run:
        return None

    prefix, cycle = _parse_spot_run_to_lasso(str(run))
    return prefix, cycle


def reach_avoid_get_counterexample_lasso(
    absys: Abstraction,
    init_uids: Set[int],
    merge_actions: bool = True,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Find a counterexample directly in the graph.
    """
    label_cache: Dict[int, Set[str]] = {
        absys.OUT_UID: set(absys.ap_labeler(None))
    }

    def labels(uid: int) -> Set[str]:
        if uid not in label_cache:
            node = absys.part.leaves.get(uid)
            label_cache[uid] = (
                set(absys.ap_labeler(node.rect))
                if node is not None
                else {"unsafe"}
            )
        return label_cache[uid]

    def successors(uid: int) -> List[int]:
        by_action = absys.tr.succ.get(uid, {})
        if not by_action:
            return [uid]  # deadlocks stutter in the corresponding Kripke graph
        if merge_actions:
            merged: Set[int] = set()
            for values in by_action.values():
                merged |= values
            return sorted(merged) if merged else [uid]
        merged = set()
        for values in by_action.values():
            merged |= values
        return sorted(merged) if merged else [uid]

    def is_goal(uid: int) -> bool:
        return "goal" in labels(uid)

    def is_bad(uid: int) -> bool:
        return uid == absys.OUT_UID or "unsafe" in labels(uid)

    def complete_bad_prefix_to_lasso(path: List[int]):
        """After safety is violated, follow arbitrary edges to a real cycle."""
        walk = list(path)
        first_position = {uid: i for i, uid in enumerate(walk)}
        while True:
            nxt = successors(walk[-1])[0]
            if nxt in first_position:
                cycle_start = first_position[nxt]
                return walk[:cycle_start], walk[cycle_start:]
            first_position[nxt] = len(walk)
            walk.append(nxt)

    finished: Set[int] = set()
    for initial in sorted(init_uids):
        if initial != absys.OUT_UID and initial not in absys.part.leaves:
            continue
        # Reaching goal ends the reach-avoid obligation for this run.
        if is_goal(initial):
            continue
        if is_bad(initial):
            return complete_bad_prefix_to_lasso([initial])
        if initial in finished:
            continue

        stack_nodes: List[int] = [initial]
        stack_iters = [iter(successors(initial))]
        stack_position = {initial: 0}

        while stack_nodes:
            uid = stack_nodes[-1]
            try:
                nxt = next(stack_iters[-1])
            except StopIteration:
                finished.add(uid)
                stack_position.pop(uid, None)
                stack_nodes.pop()
                stack_iters.pop()
                continue

            if is_goal(nxt):
                continue
            if is_bad(nxt):
                return complete_bad_prefix_to_lasso(stack_nodes + [nxt])
            if nxt in stack_position:
                cycle_start = stack_position[nxt]
                return (
                    list(stack_nodes[:cycle_start]),
                    list(stack_nodes[cycle_start:]),
                )
            if nxt in finished:
                continue

            stack_position[nxt] = len(stack_nodes)
            stack_nodes.append(nxt)
            stack_iters.append(iter(successors(nxt)))

    return None

# Validation 
def expand_lasso(prefix: List[int], cycle: List[int], repeat_cycle_once: bool = True) -> List[int]:
    path = list(prefix) + list(cycle)
    if repeat_cycle_once and cycle:
        path += list(cycle)
    return path

@dataclass
class ValidationResult:
    feasible: bool
    fail_index: Optional[int]
    refine_uid: Optional[int]
    trace_sets: List[Rect]

import numpy as np

def _rect_corners(r: Rect) -> np.ndarray:
    """Return corners of r. 4 corners for 2D, 8 corners for 3D."""
    if r.zmin != r.zmax:  # 3D
        return np.array([
            [r.xmin, r.ymin, r.zmin],
            [r.xmin, r.ymin, r.zmax],
            [r.xmin, r.ymax, r.zmin],
            [r.xmin, r.ymax, r.zmax],
            [r.xmax, r.ymin, r.zmin],
            [r.xmax, r.ymin, r.zmax],
            [r.xmax, r.ymax, r.zmin],
            [r.xmax, r.ymax, r.zmax],
        ], dtype=float)
    else:  # 2D
        return np.array([
            [r.xmin, r.ymin],
            [r.xmin, r.ymax],
            [r.xmax, r.ymax],
            [r.xmax, r.ymin],
        ], dtype=float)

def _bbox_of_points(pts: np.ndarray) -> Rect:
    xs = pts[:, 0]; ys = pts[:, 1]
    return Rect(float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))

def _point_to_uid(absys: Abstraction, p: np.ndarray) -> int:
    x, y = float(p[0]), float(p[1])
    z = float(p[2]) if len(p) > 2 else 0.0
    uid = absys.part.query_point(x, y, z)
    return uid if uid >= 0 else absys.OUT_UID


def _infer_goal_ball_from_ap_labeler(absys: Abstraction, action: str = "step"):
    import numpy as np

    dyn = absys.dyn_by_action.get(action)
    if dyn is not None and hasattr(dyn, "xstar"):
        c = np.asarray(dyn.xstar, dtype=float).reshape(2,)
        r = float(getattr(dyn, "goal_radius", 1.0))
        return c, r

    return np.array([5.0, 5.0], dtype=float), 1.0
def validate_lasso_by_set_propagation(
    absys: Abstraction,
    prefix: List[int],
    cycle: List[int],
    action: str = "step",
) -> ValidationResult:
    import numpy as np

    MAX_STEPS = 5000

    path = expand_lasso(prefix, cycle, repeat_cycle_once=True)
    if len(path) < 1:
        return ValidationResult(True, None, None, [])

    u0 = path[0]

    if u0 == absys.OUT_UID:
        return ValidationResult(True, None, None, [])

    if u0 not in absys.part.leaves:
        return ValidationResult(True, None, None, [])

    dyn = absys.dyn_by_action[action]

    goal_ball = _infer_goal_ball_from_ap_labeler(absys, action=action)
    if goal_ball is None:
        return ValidationResult(True, None, None, [])

    center, radius = goal_ball
    center = np.asarray(center, dtype=float)
    r2 = float(radius) * float(radius)

    # For 3D cells use center point only — 8-corner propagation is too
    # conservative and drives boundary corners out of domain immediately
    # For 2D cells keep 4-corner propagation (exact for affine dynamics)
    r0 = absys.part.leaves[u0].rect
    if r0.zmin != r0.zmax:  # 3D
        cx = 0.5 * (r0.xmin + r0.xmax)
        cy = 0.5 * (r0.ymin + r0.ymax)
        cz = 0.5 * (r0.zmin + r0.zmax)
        pts = np.array([[cx, cy, cz]], dtype=float)
    else:  # 2D
        pts = _rect_corners(r0)

    trace: List[Rect] = [_bbox_of_points(pts[:, :2].reshape(-1, 2)
                                          if pts.shape[1] > 2
                                          else pts)]

    # Step 0 goal check
    d0 = pts[:, :2] - center[None, :]
    if bool(np.any(np.sum(d0 * d0, axis=1) <= r2)):
        return ValidationResult(False, 0, u0, trace)

    for t in range(1, MAX_STEPS + 1):
        pts = np.array([dyn.dynamics(p) for p in pts], dtype=float)

        # OUT check — feasible only if all points leave domain
        uids = [_point_to_uid(absys, p) for p in pts]
        if all(uid == absys.OUT_UID for uid in uids):
            return ValidationResult(True, None, None, trace)

        # Filter out-of-bounds points for next iteration
        pts = np.array([p for p, u in zip(pts, uids)
                        if u != absys.OUT_UID], dtype=float)
        if len(pts) == 0:
            return ValidationResult(True, None, None, trace)

        trace.append(_bbox_of_points(pts[:, :2].reshape(-1, 2)
                                      if pts.shape[1] > 2
                                      else pts))

        # Goal check — spurious if any point reaches goal
        d = pts[:, :2] - center[None, :]
        if bool(np.any(np.sum(d * d, axis=1) <= r2)):
            return ValidationResult(False, t, u0, trace)

    return ValidationResult(True, None, None, trace)

# Refinement

def split_midpoint(absys: Abstraction, leaf_uid: int) -> Tuple[int, int, int, int]:
    """Legacy xy-only midpoint split. Kept for backward compatibility;
    run_cegar now calls split_cell (dimension-selected) instead."""
    node = absys.part.leaves[leaf_uid]
    r = node.rect
    xm = 0.5 * (r.xmin + r.xmax)
    ym = 0.5 * (r.ymin + r.ymax)
    return absys.split_and_update(xm, ym, leaf_uid)


def _taylor_error_terms_per_dim(absys: Abstraction, r: Rect, action: str = "step") -> Optional[np.ndarray]:

    try:
        import main as _main  # local import: only needed for this heuristic
    except Exception:
        return None

    dyn = absys.dyn_by_action.get(action)
    if dyn is None or not hasattr(dyn, "image_bbox"):
        return None

    try:
        lower = np.array([r.xmin, r.ymin, r.zmin], dtype=float)
        upper = np.array([r.xmax, r.ymax, r.zmax], dtype=float)
        R_lo, R_hi = _main.taylor_remainder(lower, upper)
        h = 0.5 * (upper - lower)
        mag = np.abs(R_hi) + np.abs(R_lo)
        weighted = mag * (h > 1e-9)
        return weighted
    except Exception:
        return None


def choose_split_dims(
    absys: Abstraction,
    leaf_uid: int,
    mode: str = "auto",
    action: str = "step",
    min_extent: float = 1e-9,
) -> Tuple[bool, bool, bool]:
    """
    Decide which of (x, y, theta) to split for this cell.
    """
    node = absys.part.leaves[leaf_uid]
    r = node.rect
    extents = np.array([rect_width(r), rect_height(r), rect_depth(r)], dtype=float)
    splittable = extents > min_extent

    if mode == "xy":
        return (bool(splittable[0]), bool(splittable[1]), False)
    if mode == "xyz":
        return (bool(splittable[0]), bool(splittable[1]), bool(splittable[2]))

    # mode == "auto"
    weighted = _taylor_error_terms_per_dim(absys, r, action=action)
    if weighted is not None and np.any(weighted > 0):
        weighted = np.where(splittable, weighted, -np.inf)
        best = int(np.argmax(weighted))
    else:
        sizes = np.where(splittable, extents, -np.inf)
        best = int(np.argmax(sizes))

    choice = [False, False, False]
    if splittable[best]:
        choice[best] = True
    else:
        for i in range(3):
            if splittable[i]:
                choice[i] = True
                break
    return tuple(choice)  # type: ignore[return-value]


def split_cell(
    absys: Abstraction,
    leaf_uid: int,
    mode: str = "auto",
    action: str = "step",
    split_dims: Optional[Tuple[bool, bool, bool]] = None,
) -> Tuple[int, ...]:

    node = absys.part.leaves[leaf_uid]
    r = node.rect

    if split_dims is None:
        split_dims = choose_split_dims(
            absys, leaf_uid, mode=mode, action=action
        )
    split_x, split_y, split_z = split_dims

    xm = 0.5 * (r.xmin + r.xmax) if split_x else None
    ym = 0.5 * (r.ymin + r.ymax) if split_y else None
    zm = 0.5 * (r.zmin + r.zmax) if split_z else None

    if xm is None and ym is None and zm is None:
        return absys.split_and_update(0.5 * (r.xmin + r.xmax), 0.5 * (r.ymin + r.ymax), leaf_uid)

    return absys.split_and_update_general(leaf_uid, xm=xm, ym=ym, zm=zm)


def can_refine(
    absys: Abstraction,
    uid: int,
    min_width: float,
    min_height: float,
    max_depth: Optional[int],
    min_depth_theta: Optional[float] = None,
) -> bool:
    if uid == absys.OUT_UID:
        return False
    node = absys.part.leaves.get(uid)
    if node is None:
        return False
    r = node.rect
    if rect_width(r) <= min_width or rect_height(r) <= min_height:
        return False
    if min_depth_theta is not None and rect_depth(r) <= min_depth_theta:
        return False
    if max_depth is not None and node.depth >= max_depth:
        return False
    return True

 
# Full CEGAR loop
@dataclass
class CEGARResult:
    verified: bool
    iterations: int
    last_cex: Optional[Tuple[List[int], List[int]]]
    ignored_counterexamples: int
    refinements: int
    stopped: bool = False
    stop_reason: Optional[str] = None


def run_cegar(
    absys: Abstraction,
    init_uids: Set[int],
    phi: Optional[str] = None,
    action: str = "step",
    max_iters: int = 50,
    merge_actions: bool = True,
    min_cell_width: float = 1e-6,
    min_cell_height: float = 1e-6,
    max_refine_depth: Optional[int] = None,
    min_cell_theta: Optional[float] = None,
    split_mode: str = "auto",
    verbose: bool = True,
    counterexample_backend: str = "auto",
    stop_requested: Optional[Callable[[], bool]] = None,
    max_total_states: Optional[int] = None,
) -> CEGARResult:
    """
    Full CEGAR loop
    """
    if phi is None:
        # original phi value
        phi = absys.reach_avoid_ltl(goal_ap="goal", unsafe_ap="unsafe")


    if counterexample_backend not in {"auto", "graph", "spot"}:
        raise ValueError(
            "counterexample_backend must be 'auto', 'graph', or 'spot'"
        )
    if max_total_states is not None and max_total_states <= 0:
        raise ValueError("max_total_states must be positive or None")

    refinements = 0
    ignored = 0
    last_cex: Optional[Tuple[List[int], List[int]]] = None
    iterations_completed = 0

    # Local, mutable copy -- we update this after every split so that stale
    init_uids = set(init_uids)

    # Ensure transitions exist
    # (caller can rebuild or rely on incremental updates from split_and_update)
    if not absys.tr.succ:
        absys.rebuild_all_transitions()

    for it in range(max_iters):
        if (
            max_total_states is not None
            and len(absys.part.leaves) >= max_total_states
        ):
            if verbose:
                print(
                    f"[CEGAR] State limit reached: "
                    f"{len(absys.part.leaves)}/{max_total_states} leaves."
                )
            return CEGARResult(
                False,
                iterations_completed,
                last_cex,
                ignored,
                refinements,
                stopped=True,
                stop_reason="state_limit",
            )
        if stop_requested is not None and stop_requested():
            if verbose:
                print("[CEGAR] Stop requested; returning at an iteration boundary.")
            return CEGARResult(
                False, iterations_completed, last_cex, ignored, refinements,
                stopped=True,
                stop_reason="external_stop",
            )

        if verbose:
            print(f"\n[CEGAR] Iter {it}: checking phi = {phi}")

        normalized_phi = "".join(phi.split())
        use_graph = (
            counterexample_backend == "graph" or
            (
                counterexample_backend == "auto" and
                normalized_phi in {
                    "(!unsafe)Ugoal",
                    "G!unsafe&Fgoal",
                }
            )
        )
        if use_graph:
            cex = reach_avoid_get_counterexample_lasso(
                absys, init_uids, merge_actions=merge_actions
            )
        else:
            cex = spot_get_counterexample_lasso(
                absys, init_uids, phi, merge_actions=merge_actions
            )
        if cex is None:
            if verbose:
                print("[CEGAR] VERIFIED (no abstract counterexample).")
            return CEGARResult(
                True, iterations_completed, None, ignored, refinements
            )

        prefix, cycle = cex
        last_cex = cex
        iterations_completed += 1
        if verbose:
            print("[CEGAR] Abstract counterexample found.")
            print("  prefix:", prefix)
            print("  cycle :", cycle)

        # Validate
        vr = validate_lasso_by_set_propagation(absys, prefix, cycle, action=action)
        if vr.feasible:
            if verbose:
                print("[CEGAR] Found CONCRETE-feasible counterexample at current precision.")
                print("[CEGAR] NOT VERIFIED.")
            return CEGARResult(
                False, iterations_completed, last_cex, ignored, refinements
            )

        # Spurious -> refine
        uid = vr.refine_uid
        if uid is None or uid == absys.OUT_UID:
            ignored += 1
            if verbose:
                print("[CEGAR] Spurious CE but no valid refine uid. Ignoring.")
            break

        if not can_refine(absys, uid, min_cell_width, min_cell_height, max_refine_depth,
                           min_depth_theta=min_cell_theta):
            ignored += 1
            if verbose:
                node = absys.part.leaves.get(uid)
                d = node.depth if node else None
                print(f"[CEGAR] Spurious but uid={uid} hit refinement limits (depth={d}). Ignoring CE.")

            break

        if verbose:
            print(f"[CEGAR] Spurious. Refining uid={uid} (iter {it}, mode={split_mode}).")

        split_dims = choose_split_dims(
            absys, uid, mode=split_mode, action=action
        )
        child_count = 2 ** sum(bool(value) for value in split_dims)
        prospective_leaf_count = (
            len(absys.part.leaves) - 1 + child_count
        )
        if (
            max_total_states is not None
            and prospective_leaf_count > max_total_states
        ):
            if verbose:
                print(
                    "[CEGAR] Next split would exceed state limit: "
                    f"{prospective_leaf_count}>{max_total_states}."
                )
            return CEGARResult(
                False,
                iterations_completed,
                last_cex,
                ignored,
                refinements,
                stopped=True,
                stop_reason="state_limit",
            )

        new_children = split_cell(
            absys,
            uid,
            mode=split_mode,
            action=action,
            split_dims=split_dims,
        )
        refinements += 1

        # --- BUG FIX: keep init_uids in sync with the (now-changed) partition ---
        if uid in init_uids:
            init_uids = (init_uids - {uid}) | set(new_children)
        # --------------------------------------------------------------------

        # absys.rebuild_all_transitions()

    if verbose:
        print(
            f"\n[CEGAR] Gave up after {iterations_completed} iteration(s) "
            f"(max_iters={max_iters})."
        )
    return CEGARResult(
        False, iterations_completed, last_cex, ignored, refinements
    )
