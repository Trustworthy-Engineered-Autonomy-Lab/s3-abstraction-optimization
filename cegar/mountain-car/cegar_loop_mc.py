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
    Forward BFS over absys.tr.succ .
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
    Build a Spot Kripke structure from `absys` using a single shared BDD dict,
    so that we can also translate formulas with that same dict.
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

        state_names.append("INIT_HUB")

    # Give Spot readable names for these states (now includes the hub's
    # name too, if one was created above -- must run AFTER hub creation).
    k.set_state_names(state_names)

    # Edges -- only iterate over uids in the reachable set; their successors
    # are guaranteed to already be in uid_to_sid since reachable_uids is
    # closed under the successor relation by construction.
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
    Parse Spot's textual run format into (prefix_uids, cycle_uids).

    We extract the integer state-names (uids) from the lines in Prefix/Cycle sections.
    Any non-integer state names (e.g., init hub) are ignored.
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
    # (this is required so APs match the Kripke's BDD vars)
    # not_phi = spot.formula.Not(phi)
    f = spot.formula(phi)
    not_phi = spot.formula.Not(f)

    aut = spot.translate(not_phi, dict=d)

    run = k.intersecting_run(aut)
    if not run:
        return None

    prefix, cycle = _parse_spot_run_to_lasso(str(run))
    return prefix, cycle

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
    z = float(p[2]) if len(p) > 2 else None
    for uid, node in absys.part.leaves.items():
        if uid == absys.OUT_UID:
            continue
        r = node.rect
        if not (r.xmin <= x <= r.xmax and r.ymin <= y <= r.ymax):
            continue
        if z is not None and r.zmin != r.zmax:
            if not (r.zmin <= z <= r.zmax):
                continue
        return uid
    return absys.OUT_UID


def _infer_goal_ball_from_ap_labeler(absys: Abstraction, action: str = "step"):
    import numpy as np

    dyn = absys.dyn_by_action.get(action)
    if dyn is not None and hasattr(dyn, "xstar"):
        c = np.asarray(dyn.xstar, dtype=float).reshape(2,)
        r = float(getattr(dyn, "goal_radius", 1.0))
        return c, r

    return np.array([5.0, 5.0], dtype=float), 1.0


def _goal_check_batch(absys: Abstraction, pts_xy, action: str = "step"):

    import numpy as np

    dyn = absys.dyn_by_action.get(action)
    if dyn is not None and hasattr(dyn, "is_goal_batch"):
        return np.asarray(dyn.is_goal_batch(pts_xy), dtype=bool)

    center, radius = _infer_goal_ball_from_ap_labeler(absys, action=action)
    d = pts_xy - center[None, :]
    return np.sum(d * d, axis=1) <= (radius * radius)


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
    if bool(np.any(_goal_check_batch(absys, pts[:, :2], action=action))):
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
        if bool(np.any(_goal_check_batch(absys, pts[:, :2], action=action))):
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

    dyn = absys.dyn_by_action.get(action)
    if dyn is None or not hasattr(dyn, "taylor_remainder"):
        return None

    try:
        is_3d = (r.zmin != r.zmax)
        if is_3d:
            lower = np.array([r.xmin, r.ymin, r.zmin], dtype=float)
            upper = np.array([r.xmax, r.ymax, r.zmax], dtype=float)
        else:
            lower = np.array([r.xmin, r.ymin], dtype=float)
            upper = np.array([r.xmax, r.ymax], dtype=float)

        R_lo, R_hi = dyn.taylor_remainder(lower, upper)
        # magnitude of remainder contributed "at" each output dim; use as a
        # proxy for which input half-width is driving imprecision the most.
        # Combine with each dim's own half-width so we don't keep splitting
        # an axis that's already essentially degenerate.
        h = 0.5 * (upper - lower)
        mag = np.abs(R_hi) + np.abs(R_lo)
        # Weight by remaining half-width so a dimension that's already tiny
        # isn't picked just because its remainder term happens to be large.
        weighted_partial = mag * (h > 1e-9)

        # choose_split_dims always expects a length-3 (x, y, theta) array,
        # even for genuinely-2D systems (where the z/theta axis is a
        # degenerate 0-width placeholder). Pad rather than assume 3D.
        weighted = np.zeros(3, dtype=float)
        weighted[:len(weighted_partial)] = weighted_partial
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

    mode:
      "xy"        -- legacy behavior, always split x and y only.
      "xyz"       -- always split all three dims (8-way split).
      "auto"      -- (default) split the single dimension responsible for
                     the most imprecision, falling back to "largest extent"
                     when a Taylor-remainder-based signal isn't available.
                     This keeps state-count growth closer to CEGAR's normal
                     per-iteration cost (one bisection, 2 children) while
                     still allowing theta to shrink when it's the bottleneck.

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
        # Fallback: split whichever splittable dimension is geometrically
        # largest (matches the intuition of "cut the biggest side").
        sizes = np.where(splittable, extents, -np.inf)
        best = int(np.argmax(sizes))

    choice = [False, False, False]
    if splittable[best]:
        choice[best] = True
    else:
        # Nothing splittable at all (shouldn't normally happen since
        # can_refine already checked width/height before calling us) --
        # fall back to x then y then z, whichever is available.
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
) -> Tuple[int, ...]:

    node = absys.part.leaves[leaf_uid]
    r = node.rect

    split_x, split_y, split_z = choose_split_dims(absys, leaf_uid, mode=mode, action=action)

    xm = 0.5 * (r.xmin + r.xmax) if split_x else None
    ym = 0.5 * (r.ymin + r.ymax) if split_y else None
    zm = 0.5 * (r.zmin + r.zmax) if split_z else None

    if xm is None and ym is None and zm is None:
        # Degenerate: nothing to split on. Caller's can_refine should have
        # already prevented this; fall back to legacy xy split as a safety net.
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
    """
    min_depth_theta: minimum theta extent allowed (analogous to min_width /
    min_height but for the z/theta axis)
    """
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
    prefer_cycle_refine: bool = False,
    also_refine_prefix_head: bool = False,
    verbose: bool = True,
    stop_requested: Optional[Callable[[], bool]] = None,
    checkpoint_callback: Optional[Callable[[Abstraction, Set[int], int, Optional[Tuple[List[int], List[int]]], int, int], None]] = None,
    checkpoint_every: Optional[int] = None,
) -> CEGARResult:
    """
    Full CEGAR loop:

      repeat:
        - Spot finds abstract counterexample lasso for phi
        - validate lasso via set-propagation (path-consistency)
        - if spurious: split selected cell and continue
        - if feasible: return NOT VERIFIED (real counterexample witness at this precision)
      until max_iters

    split_mode: passed straight through to split_cell / choose_split_dims.
      "xy" reproduces the old x/y-only bisection; "xyz" always splits all
      three axes; "auto" (default) picks the single most-useful axis per
      split (falls back to "largest extent" if no Taylor-remainder signal
      is available for the current dynamics object).

    min_cell_theta: analogous to min_cell_width/min_cell_height but for the
      theta axis. Pass this if you want refinement to also stop once theta
      extent gets small enough -- otherwise theta can in principle keep
      splitting down toward min_extent inside choose_split_dims.

    prefer_cycle_refine: when True and a spurious counterexample has a
      non-empty cycle, refine the FIRST CELL OF THE CYCLE instead of
      validate_lasso_by_set_propagation's default choice (always
      path[0], i.e. the closest-to-init cell). Default False -- this is
      an OPT-IN behavioral change, since past results (e.g. unicycle) were
      produced without it and this preserves exact reproducibility unless
        the user explicitly enables it.
    also_refine_prefix_head: when True (and typically combined with
      prefer_cycle_refine=True), ALSO refine prefix[0] every iteration a
      spurious CE is found, in addition to whatever prefer_cycle_refine
      selects. Default False -- opt-in, same reproducibility rationale as
      prefer_cycle_refine.

    """
    if phi is None:
        # original phi value
        phi = absys.reach_avoid_ltl(goal_ap="goal", unsafe_ap="unsafe")


    refinements = 0
    ignored = 0
    last_cex: Optional[Tuple[List[int], List[int]]] = None

    # Local, mutable copy -- we update this after every split so that stale
    init_uids = set(init_uids)

    def _emit_checkpoint(iteration: int) -> None:
        if checkpoint_callback is not None:
            checkpoint_callback(absys, init_uids, iteration, last_cex, ignored, refinements)

    # Ensure transitions exist
    # (caller can rebuild or rely on incremental updates from split_and_update)
    if not absys.tr.succ:
        absys.rebuild_all_transitions()

    for it in range(max_iters):
        if stop_requested is not None and stop_requested():
            if verbose:
                print(f"\n[CEGAR] Stop requested at iter {it}; saving checkpoint and exiting.")
            _emit_checkpoint(it)
            return CEGARResult(False, it, last_cex, ignored, refinements)

        if verbose:
            print(f"\n[CEGAR] Iter {it}: checking phi = {phi}")

        cex = spot_get_counterexample_lasso(absys, init_uids, phi, merge_actions=merge_actions)
        if cex is None:
            if verbose:
                print("[CEGAR] VERIFIED (no abstract counterexample).")
            _emit_checkpoint(it)
            return CEGARResult(True, it, None, ignored, refinements)

        prefix, cycle = cex
        last_cex = cex
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
            _emit_checkpoint(it + 1)
            return CEGARResult(False, it + 1, last_cex, ignored, refinements)

        # Spurious -> refine
        default_uid = vr.refine_uid  # validate_lasso_by_set_propagation's own choice (path[0])

        # --- Build the list of candidate refine targets for this iteration ---
        # Default (both flags False): identical to the original behavior --
        # exactly one target, default_uid, preserving full backward
        # compatibility / reproducibility of past (e.g. unicycle) results.
        targets: List[int] = []

        cycle_uid = None
        if prefer_cycle_refine and cycle:
            candidate = cycle[0]
            if candidate != absys.OUT_UID and candidate in absys.part.leaves:
                cycle_uid = candidate

        prefix_uid = None
        if also_refine_prefix_head and prefix:
            candidate = prefix[0]
            if candidate != absys.OUT_UID and candidate in absys.part.leaves:
                prefix_uid = candidate

        if also_refine_prefix_head:

            if prefix_uid is not None:
                targets.append(prefix_uid)
            if cycle_uid is not None and cycle_uid != prefix_uid:
                targets.append(cycle_uid)
            if not targets and default_uid is not None:
                targets.append(default_uid)
        elif prefer_cycle_refine and cycle_uid is not None:
            if verbose and cycle_uid != default_uid:
                print(f"[CEGAR] prefer_cycle_refine: overriding refine target "
                      f"{default_uid} -> {cycle_uid} (first cell of cycle).")
            targets.append(cycle_uid)
        elif default_uid is not None:
            targets.append(default_uid)
        # ----------------------------------------------------------------

        if not targets:
            ignored += 1
            if verbose:
                print("[CEGAR] Spurious CE but no valid refine uid. Ignoring.")
            break

        any_refined = False
        for t in targets:
            if t is None or t == absys.OUT_UID or t not in absys.part.leaves:
                continue
            if not can_refine(absys, t, min_cell_width, min_cell_height, max_refine_depth,
                               min_depth_theta=min_cell_theta):
                if verbose:
                    node = absys.part.leaves.get(t)
                    d = node.depth if node else None
                    print(f"[CEGAR] Spurious but uid={t} hit refinement limits (depth={d}). Skipping this target.")
                continue

            if verbose:
                print(f"[CEGAR] Spurious. Refining uid={t} (iter {it}, mode={split_mode}).")

            new_children = split_cell(absys, t, mode=split_mode, action=action)
            refinements += 1
            any_refined = True

            # --- BUG FIX: keep init_uids in sync with the (now-changed) partition ---
            if t in init_uids:
                init_uids = (init_uids - {t}) | set(new_children)
            # --------------------------------------------------------------------

        if not any_refined:
            ignored += 1
            if verbose:
                print("[CEGAR] Spurious CE but every candidate target hit refinement limits. Ignoring.")
            break

        if checkpoint_every is not None and checkpoint_every > 0 and ((it + 1) % checkpoint_every == 0):
            _emit_checkpoint(it + 1)

        # absys.rebuild_all_transitions()

    if verbose:
        print(f"\n[CEGAR] Gave up after {it + 1} iteration(s) (max_iters={max_iters}).")
    return CEGARResult(False, it + 1, last_cex, ignored, refinements)