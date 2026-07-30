from __future__ import annotations
from bisect import bisect_left, bisect_right
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Set, Tuple, Iterable, Callable

@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle object."""
    xmin: float; xmax: float
    ymin: float; ymax: float
    zmin: float = 0.0; zmax: float = 0.0   # optional 3rd dim

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.xmax < other.xmin or other.xmax < self.xmin or
            self.ymax < other.ymin or other.ymax < self.ymin or
            (self.zmax != self.zmin and
             (self.zmax < other.zmin or other.zmax < self.zmin))
        )

    def split4(self, xm, ym) -> Tuple["Rect","Rect","Rect","Rect"]:

        return (
            Rect(self.xmin, xm, self.ymin, ym, self.zmin, self.zmax),
            Rect(xm, self.xmax, self.ymin, ym, self.zmin, self.zmax),
            Rect(self.xmin, xm, ym, self.ymax, self.zmin, self.zmax),
            Rect(xm, self.xmax, ym, self.ymax, self.zmin, self.zmax),
        )

    def split_dims(self, xm=None, ym=None, zm=None) -> Tuple["Rect", ...]:

        x_ranges = ((self.xmin, xm), (xm, self.xmax)) if xm is not None else ((self.xmin, self.xmax),)
        y_ranges = ((self.ymin, ym), (ym, self.ymax)) if ym is not None else ((self.ymin, self.ymax),)
        z_ranges = ((self.zmin, zm), (zm, self.zmax)) if zm is not None else ((self.zmin, self.zmax),)

        out: List[Rect] = []
        for xlo, xhi in x_ranges:
            for ylo, yhi in y_ranges:
                for zlo, zhi in z_ranges:
                    out.append(Rect(xlo, xhi, ylo, yhi, zlo, zhi))
        return tuple(out)


class CellNode:
    """Node in the rectangle state space partition tree.
    """
    __slots__ = ("uid", "rect", "parent", "children", "depth")
    def __init__(self, uid: int, rect: Rect, parent: Optional["CellNode"]=None, depth: int=0):
        self.uid = uid
        self.rect = rect
        self.parent = parent
        self.children: Optional[Tuple["CellNode", ...]] = None
        self.depth = depth

    def is_leaf(self) -> bool:
        return self.children is None


class RectPartition:
    """Forest of roots (initial uniform grid cells), each refined independently."""
    def __init__(self, roots: List[CellNode], next_uid: int):
        self.roots = roots
        self._next_uid = next_uid
        self.leaves: Dict[int, CellNode] = {}
        # Lazily inferred so legacy raw-Abstraction pickles remain loadable.
        self._uniform_index_ready = False
        self._uniform_root_edges = None
        self._uniform_root_shape = None
        for r in roots:
            self._collect_leaves(r)

    @staticmethod
    def uniform_grid(domain: Rect, nx: int, ny: int, nz: int = 1) -> "RectPartition":
        x_edges = np.linspace(domain.xmin, domain.xmax, nx + 1)
        y_edges = np.linspace(domain.ymin, domain.ymax, ny + 1)
        z_edges = (
            np.linspace(domain.zmin, domain.zmax, nz + 1)
            if nz > 1
            else None
        )
        uid = 0
        roots = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz if nz > 1 else 1):
                    zlo = z_edges[k] if z_edges is not None else domain.zmin
                    zhi = (
                        z_edges[k + 1]
                        if z_edges is not None
                        else domain.zmax
                    )
                    r = Rect(
                        float(x_edges[i]), float(x_edges[i + 1]),
                        float(y_edges[j]), float(y_edges[j + 1]),
                        float(zlo), float(zhi),
                    )
                    roots.append(CellNode(uid, r))
                    uid += 1
        return RectPartition(roots, next_uid=uid)

    def query_point(self, x, y, z=0.0) -> int:
        """Return uid of leaf containing (x, y, z)."""
        point_box = Rect(x, x, y, y, z, z)
        for uid in self.query_intersecting_leaves(point_box):
            node = self.leaves[uid]
            r = node.rect
            if (r.xmin <= x <= r.xmax and
                r.ymin <= y <= r.ymax and
                r.zmin <= z <= r.zmax):
                return uid
        return -1

    def _collect_leaves(self, node: CellNode) -> None:
        if node.is_leaf():
            self.leaves[node.uid] = node
        else:
            for c in node.children:
                self._collect_leaves(c)

    def split(self, xm, ym, leaf_uid: int) -> Tuple[int, int, int, int]:
        """old 4-way xy-only split """
        node = self.leaves[leaf_uid]
        assert node.is_leaf()
        del self.leaves[leaf_uid]

        r1, r2, r3, r4 = node.rect.split4(xm, ym)
        kids = []
        for rr in (r1, r2, r3, r4):
            kids.append(CellNode(self._next_uid, rr, parent=node, depth=node.depth + 1))
            self._next_uid += 1
        node.children = tuple(kids)

        for k in kids:
            self.leaves[k.uid] = k
        return tuple(k.uid for k in kids)

    def split_general(self, leaf_uid: int, xm=None, ym=None, zm=None) -> Tuple[int, ...]:
        """
        Split a leaf along any subset of {x, y, z}. Returns the tuple of new
        child uids (length 2, 4, or 8 depending on how many midpoints given).
        """
        node = self.leaves[leaf_uid]
        assert node.is_leaf()
        del self.leaves[leaf_uid]

        child_rects = node.rect.split_dims(xm=xm, ym=ym, zm=zm)
        kids = []
        for rr in child_rects:
            kids.append(CellNode(self._next_uid, rr, parent=node, depth=node.depth + 1))
            self._next_uid += 1
        node.children = tuple(kids)

        for k in kids:
            self.leaves[k.uid] = k
        return tuple(k.uid for k in kids)

    def query_intersecting_leaves(self, box: Rect) -> List[int]:
        """Uses the partition tree itself as the spatial index."""
        out: List[int] = []
        self._ensure_uniform_root_index()

        edges = getattr(self, "_uniform_root_edges", None)
        shape = getattr(self, "_uniform_root_shape", None)
        if edges is None or shape is None:
            for root in self.roots:
                self._query_node(root, box, out)
            return out

        x_edges, y_edges, z_edges = edges
        nx, ny, nz = shape
        x_range = self._overlapping_root_range(
            box.xmin, box.xmax, x_edges, nx
        )
        y_range = self._overlapping_root_range(
            box.ymin, box.ymax, y_edges, ny
        )
        z_range = self._overlapping_root_range(
            box.zmin, box.zmax, z_edges, nz
        )
        if x_range is None or y_range is None or z_range is None:
            return out

        for i in range(x_range[0], x_range[1] + 1):
            for j in range(y_range[0], y_range[1] + 1):
                base = (i * ny + j) * nz
                for k in range(z_range[0], z_range[1] + 1):
                    self._query_node(self.roots[base + k], box, out)
        return out

    @staticmethod
    def _overlapping_root_range(lo, hi, edges, count):
        if lo != lo or hi != hi:
            return (0, count - 1)
        if hi < edges[0] or lo > edges[-1]:
            return None
        first = max(0, min(count - 1, bisect_left(edges, lo) - 1))
        last = max(0, min(count - 1, bisect_right(edges, hi) - 1))
        return (first, last)

    def _ensure_uniform_root_index(self) -> None:
        if getattr(self, "_uniform_index_ready", False):
            return

        self._uniform_index_ready = True
        self._uniform_root_edges = None
        self._uniform_root_shape = None
        if not self.roots:
            return

        x_edges = sorted({
            edge
            for node in self.roots
            for edge in (node.rect.xmin, node.rect.xmax)
        })
        y_edges = sorted({
            edge
            for node in self.roots
            for edge in (node.rect.ymin, node.rect.ymax)
        })
        z_edges = sorted({
            edge
            for node in self.roots
            for edge in (node.rect.zmin, node.rect.zmax)
        })

        nx = len(x_edges) - 1
        ny = len(y_edges) - 1
        nz = len(z_edges) - 1
        if nx <= 0 or ny <= 0:
            return
        if nz <= 0:
            z_value = self.roots[0].rect.zmin
            z_edges = [z_value, z_value]
            nz = 1
        if nx * ny * nz != len(self.roots):
            return

        tolerance = 1e-12
        for flat, node in enumerate(self.roots):
            i = flat // (ny * nz)
            remainder = flat % (ny * nz)
            j = remainder // nz
            k = remainder % nz
            rect = node.rect
            expected = (
                x_edges[i], x_edges[i + 1],
                y_edges[j], y_edges[j + 1],
                z_edges[k], z_edges[k + 1],
            )
            actual = (
                rect.xmin, rect.xmax,
                rect.ymin, rect.ymax,
                rect.zmin, rect.zmax,
            )
            if any(
                abs(value - wanted) > tolerance
                for value, wanted in zip(actual, expected)
            ):
                return

        self._uniform_root_edges = (x_edges, y_edges, z_edges)
        self._uniform_root_shape = (nx, ny, nz)

    def _query_node(self, node: CellNode, box: Rect, out: List[int]) -> None:
        if not node.rect.intersects(box):
            return
        if node.is_leaf():
            out.append(node.uid)
            return
        for c in node.children:
            self._query_node(c, box, out)


class TransitionRelation:
    """Transition relation with efficient updates and reverse lookups."""
    def __init__(self):
        self.succ: Dict[int, Dict[str, Set[int]]] = {}
        self.pred: Dict[int, Dict[str, Set[int]]] = {}

    def set_succ(self, u: int, a: str, vs: Set[int]) -> None:
        old_vs = self.succ.get(u, {}).get(a, set())
        for v in old_vs:
            if v in self.pred and a in self.pred[v]:
                self.pred[v][a].discard(u)
                if not self.pred[v][a]:
                    del self.pred[v][a]
                if not self.pred[v]:
                    del self.pred[v]
        self.succ.setdefault(u, {})[a] = set(vs)
        for v in vs:
            self.pred.setdefault(v, {}).setdefault(a, set()).add(u)


class AffineDynamics:
    goal_radius = 1.0
    def __init__(self, A: np.ndarray, xstar: np.ndarray):
        self.A = A
        self.xstar = xstar

    def dynamics(self, x):
        x = np.asarray(x, dtype=float)
        return self.xstar + self.A @ (x - self.xstar)

    def image_bbox(self, r: Rect) -> List[Rect]:
        corners = np.array([
            [r.xmin, r.ymin],
            [r.xmin, r.ymax],
            [r.xmax, r.ymin],
            [r.xmax, r.ymax],
        ], dtype=float)
        img = np.array([self.dynamics(corner) for corner in corners])
        xmin, ymin = img.min(axis=0)
        xmax, ymax = img.max(axis=0)
        return [Rect(float(xmin), float(xmax), float(ymin), float(ymax))]


class Abstraction:
    # Absorbing state id for "out of bounds" transitions
    OUT_UID = -1
    TRANSITION_SEMANTICS = "mountain-car-v3-aabb-v1"

    def __init__(
        self,
        part: RectPartition,
        dyn_by_action: Dict[str, AffineDynamics],
        ap_labeler: Callable[[Optional[Rect]], Set[str]],
    ):
        self.part = part
        self.dyn_by_action = dyn_by_action
        self.ap_labeler = ap_labeler
        self.tr = TransitionRelation()
        self._domain_xy_bounds_cache = None
        self._transition_source_recomputations_total = 0
        self._full_transition_rebuilds_total = 0
        self._incremental_transition_updates_total = 0

    def _domain_xy_bounds(self) -> Tuple[float, float, float, float]:
        """Return partition bounds, with lazy fallback for legacy pickles."""
        cached = getattr(self, "_domain_xy_bounds_cache", None)
        if cached is None:
            roots = self.part.roots
            cached = (
                min(root.rect.xmin for root in roots),
                max(root.rect.xmax for root in roots),
                min(root.rect.ymin for root in roots),
                max(root.rect.ymax for root in roots),
            )
            self._domain_xy_bounds_cache = cached
        return cached

    def _ensure_out_state(self) -> None:
        self.tr.succ.setdefault(self.OUT_UID, {})
        self.tr.pred.setdefault(self.OUT_UID, {})
        for a in self.dyn_by_action.keys():
            self.tr.set_succ(self.OUT_UID, a, {self.OUT_UID})

    def _compute_succs(self, u: int, a: str) -> Set[int]:
        """
        Compute successors using AABB convention.
        """
        self._transition_source_recomputations_total = (
            getattr(self, "_transition_source_recomputations_total", 0) + 1
        )
        node = self.part.leaves[u]
        dyn  = self.dyn_by_action[a]
        boxes: List[Rect] = dyn.image_bbox(node.rect)
        vs: Set[int] = set()
        (
            _domain_xmin,
            domain_xmax,
            _domain_ymin,
            domain_ymax,
        ) = self._domain_xy_bounds()
        for box in boxes:
            for candidate in self.part.query_intersecting_leaves(box):
                rect = self.part.leaves[candidate].rect
                x_intersects = (
                    rect.xmax > box.xmin
                    or (
                        rect.xmax == domain_xmax
                        and box.xmin == domain_xmax
                    )
                ) and rect.xmin <= box.xmax
                y_intersects = (
                    rect.ymax > box.ymin
                    or (
                        rect.ymax == domain_ymax
                        and box.ymin == domain_ymax
                    )
                ) and rect.ymin <= box.ymax
                if x_intersects and y_intersects:
                    vs.add(candidate)
        if not vs:
            vs = {self.OUT_UID}
        return vs

    def rebuild_all_transitions(self) -> None:
        self._full_transition_rebuilds_total = (
            getattr(self, "_full_transition_rebuilds_total", 0) + 1
        )
        self.tr = TransitionRelation()
        self._ensure_out_state()
        for u in self.part.leaves:
            for a in self.dyn_by_action:
                self.tr.set_succ(u, a, self._compute_succs(u, a))

    def _update_after_split(self, leaf_uid: int, new_uids: Tuple[int, ...]) -> None:
        self._incremental_transition_updates_total = (
            getattr(self, "_incremental_transition_updates_total", 0) + 1
        )

        # Clear outgoing edges of the split cell
        old_succ = self.tr.succ.get(leaf_uid, {})
        for a in list(old_succ.keys()):
            self.tr.set_succ(leaf_uid, a, set())
        self.tr.succ.pop(leaf_uid, None)

        self._ensure_out_state()

        # Recompute outgoing edges for the new children
        for u in new_uids:
            for a in self.dyn_by_action:
                self.tr.set_succ(u, a, self._compute_succs(u, a))

        # Recompute outgoing edges for predecessors of the old cell
        affected_preds: Set[Tuple[int, str]] = set()
        for a, preds in self.tr.pred.get(leaf_uid, {}).items():
            for p in preds:
                affected_preds.add((p, a))
        self.tr.pred.pop(leaf_uid, None)

        for p, a in affected_preds:
            if p == self.OUT_UID:
                continue
            self.tr.set_succ(p, a, self._compute_succs(p, a))

    def split_and_update(self, xm, ym, leaf_uid: int) -> Tuple[int,int,int,int]:
        """old 4-way xy-only split with incremental transition update."""
        new_uids = self.part.split(xm, ym, leaf_uid)
        self._update_after_split(leaf_uid, new_uids)
        return new_uids

    def split_and_update_general(self, leaf_uid: int, xm=None, ym=None, zm=None) -> Tuple[int, ...]:
        """
        Split a leaf along any subset of {x, y, z} (theta = z) and
        incrementally update the transition relation for the new children
        and any affected predecessors.
        """
        new_uids = self.part.split_general(leaf_uid, xm=xm, ym=ym, zm=zm)
        self._update_after_split(leaf_uid, new_uids)
        return new_uids

    def aps_and_labels(self) -> Tuple[Set[str], Dict[int, Set[str]]]:
        """Return (all_aps, labels_by_uid) including the OUT state."""
        labels_by_uid: Dict[int, Set[str]] = {}
        all_aps: Set[str] = set()

        out_labels = set(self.ap_labeler(None))
        labels_by_uid[self.OUT_UID] = out_labels
        all_aps |= out_labels

        for u, node in self.part.leaves.items():
            labs = set(self.ap_labeler(node.rect))
            labels_by_uid[u] = labs
            all_aps |= labs

        return all_aps, labels_by_uid

    def to_spot_kripke(self, init_uids: Set[int], merge_actions: bool = True):
        import spot
        from buddy import bdd_ithvar, bddtrue

        all_aps, labels_by_uid = self.aps_and_labels()
        d = spot.make_bdd_dict()
        k = spot.make_kripke_graph(d)
        ap_to_bdd = {ap: bdd_ithvar(k.register_ap(ap)) for ap in sorted(all_aps)}
        uids = [self.OUT_UID] + sorted(self.part.leaves.keys())
        uid_to_sid: Dict[int, int] = {}
        state_names: List[str] = []

        for u in uids:
            b = bddtrue
            for ap in labels_by_uid.get(u, set()):
                b &= ap_to_bdd[ap]
            sid = k.new_state(b)
            uid_to_sid[u] = sid
            state_names.append(str(u))

        k.set_state_names(state_names)

        if len(init_uids) == 1:
            k.set_init_state(uid_to_sid[next(iter(init_uids))])
        else:
            init_sid = k.new_state(bddtrue)
            k.set_init_state(init_sid)
            for u in init_uids:
                k.new_edge(init_sid, uid_to_sid[u])

        for u, by_a in self.tr.succ.items():
            su = uid_to_sid[u]
            dsts: Set[int] = set().union(*by_a.values()) if by_a else set()
            for v in dsts:
                k.new_edge(su, uid_to_sid[v])

        sid_to_uid = {sid: uid for uid, sid in uid_to_sid.items()}
        return k, sid_to_uid

    def reach_avoid_ltl(self, goal_ap: str = "goal", unsafe_ap: str = "unsafe") -> str:
        return f"G !{unsafe_ap} & F {goal_ap}"
