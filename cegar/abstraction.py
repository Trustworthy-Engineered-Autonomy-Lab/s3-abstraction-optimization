from __future__ import annotations
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
        """Bisect x and y only (theta/z untouched). Kept for backward
        compatibility with any caller that explicitly wants a 4-way xy split."""
        return (
            Rect(self.xmin, xm, self.ymin, ym, self.zmin, self.zmax),
            Rect(xm, self.xmax, self.ymin, ym, self.zmin, self.zmax),
            Rect(self.xmin, xm, ym, self.ymax, self.zmin, self.zmax),
            Rect(xm, self.xmax, ym, self.ymax, self.zmin, self.zmax),
        )

    def split_dims(self, xm=None, ym=None, zm=None) -> Tuple["Rect", ...]:
        """
        General splitter: bisect along any subset of {x, y, z} at the given
        midpoint(s). Pass only the midpoints for dimensions you want to cut;
        omitted dimensions (None) are left whole. Returns 2^k children where
        k = number of midpoints provided (1, 2, or 3 of them).

        """
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
        for r in roots:
            self._collect_leaves(r)

    @staticmethod
    def uniform_grid(domain: Rect, nx: int, ny: int, nz: int = 1) -> "RectPartition":
        dx = (domain.xmax - domain.xmin) / nx
        dy = (domain.ymax - domain.ymin) / ny
        dz = (domain.zmax - domain.zmin) / nz if nz > 1 else 0.0
        uid = 0
        roots = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz if nz > 1 else 1):
                    zlo = domain.zmin + k * dz if nz > 1 else domain.zmin
                    zhi = domain.zmin + (k+1)*dz if nz > 1 else domain.zmax
                    r = Rect(
                        domain.xmin + i*dx, domain.xmin + (i+1)*dx,
                        domain.ymin + j*dy, domain.ymin + (j+1)*dy,
                        zlo, zhi,
                    )
                    roots.append(CellNode(uid, r))
                    uid += 1
        return RectPartition(roots, next_uid=uid)

    def query_point(self, x, y, z=0.0) -> int:
        """Return uid of leaf containing (x, y, z)."""
        for uid, node in self.leaves.items():
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
        for r in self.roots:
            self._query_node(r, box, out)
        return out

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

    def _ensure_out_state(self) -> None:
        self.tr.succ.setdefault(self.OUT_UID, {})
        self.tr.pred.setdefault(self.OUT_UID, {})
        for a in self.dyn_by_action.keys():
            self.tr.set_succ(self.OUT_UID, a, {self.OUT_UID})

    def _compute_succs(self, u: int, a: str) -> Set[int]:
        """
        Compute successor set for cell u under action a.

        Calls dyn.image_bbox which returns List[Rect],get 
        the union of intersecting leaves across all returned image boxes. If
        none of the boxes overlap any partition cell, the cell maps to OUT.
        """
        node = self.part.leaves[u]
        dyn  = self.dyn_by_action[a]
        boxes: List[Rect] = dyn.image_bbox(node.rect)
        vs: Set[int] = set()
        for box in boxes:
            vs |= set(self.part.query_intersecting_leaves(box))
        if not vs:
            vs = {self.OUT_UID}
        return vs

    def rebuild_all_transitions(self) -> None:
        self.tr = TransitionRelation()
        self._ensure_out_state()
        for u in self.part.leaves:
            for a in self.dyn_by_action:
                self.tr.set_succ(u, a, self._compute_succs(u, a))

    def _update_after_split(self, leaf_uid: int, new_uids: Tuple[int, ...]) -> None:

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
        and any affected predecessors. enables theta-direction
        refinement
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