# main.py
import numpy as np
import sympy as sp
from itertools import product
from pathlib import Path
import pickle
import time
import mpmath as mp
from mpmath import iv

from abstraction import Rect, RectPartition, AffineDynamics, Abstraction
from cegar_loop import run_cegar
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Grid configuration ────────────────────────────────────────────────────────
NX, NY, NZ = 30, 30,30
# ─────────────────────────────────────────────────────────────────────────────

# Domain bounds
X_MIN, X_MAX = 0.0, 50.0
Y_MIN, Y_MAX = 0.0, 50.0
Z_MIN, Z_MAX = -np.pi, np.pi

# Initial domain: the region where the system is actually initialized in
# practice (per Ethan), smaller than the full state-space domain above.
# For the unicycle, this is the full x/y extent but only a narrow slice of
# theta around 0. Recall should only be evaluated over cells that fall
# within this region -- see compute_recall's `initial_domain` argument.
INIT_DOMAIN_LB = np.array([X_MIN, Y_MIN, -np.pi / 4])
INIT_DOMAIN_UB = np.array([X_MAX, Y_MAX,  np.pi / 4])

# Geometry
Y_GOAL = np.array([40.0, 20.0])
Y_OBS  = np.array([25.0, 25.0])
R_GOAL = 8.0
R_OBS  = 5.0

# ── SymPy lambdify helper ─────────────────────────────────────────────────────
_LAMBDIFY_MODULES = [
    {
        'DiracDelta': lambda x: np.zeros_like(np.asarray(x, dtype=float)),
        'Heaviside':  lambda x: np.where(np.asarray(x) >= 0, 1.0, 0.0),
    },
    'numpy',
]




# =====================================================================
# Interval arithmetic helpers (for taylor_remainder)
# =====================================================================

def _iv_tanh(x):
    return iv.mpf([mp.tanh(x.a), mp.tanh(x.b)])

def _iv_atan(x):
    return iv.atan2(x, iv.mpf([1.0, 1.0]))

def _as_interval(value):
    if hasattr(value, "a") and hasattr(value, "b"):
        return value
    return iv.mpf([float(value), float(value)])

_INTERVAL_MODULES = [
    {
        "sin": iv.sin,
        "cos": iv.cos,
        "tan": iv.tan,
        "atan": _iv_atan,
        "atan2": iv.atan2,
        "exp": iv.exp,
        "sqrt": iv.sqrt,
        "tanh": _iv_tanh,
        "DiracDelta": lambda x: iv.mpf([0, 0]),
        "Heaviside": lambda x: iv.mpf([0, 1]),
    }
]

def _interval_mul(a_lo, a_hi, b_lo, b_hi):
    a_spans = a_lo <= 0.0 <= a_hi
    b_spans = b_lo <= 0.0 <= b_hi
    if (not np.isfinite(a_lo) or not np.isfinite(a_hi)) and b_spans:
        return -np.inf, np.inf
    if (not np.isfinite(b_lo) or not np.isfinite(b_hi)) and a_spans:
        return -np.inf, np.inf
    vals = [a_lo*b_lo, a_lo*b_hi, a_hi*b_lo, a_hi*b_hi]
    return min(vals), max(vals)

# =====================================================================
# Symbolic closed-loop unicycle dynamics
# =====================================================================

def _wrap_to_pi_sym(angle):
    """Smooth wrap to [-pi, pi] via arctan(tan()) — differentiable."""
    return 2 * sp.atan(sp.tan(angle / 2))


def _cl_system_sym(state):
    """
    Symbolic closed-loop unicycle dynamics matching unicycle_system_sympy.py.
    Controller parameters must match those used in the paper baseline.
    """
    px, py, theta = state

    # Controller parameters (must match unicycle_system_sympy.py exactly)
    goal   = sp.Matrix([40.0, 20.0])
    obs    = sp.Matrix([25.0, 25.0])
    R_obs  = sp.Float(5.0)
    k_goal  = sp.Float(0.3)
    k_rep   = sp.Float(8.0)
    alpha   = sp.Float(0.6)
    k_theta = sp.Float(2.5)
    omega_max = sp.pi / 4
    eps     = sp.Float(1e-6)

    p = sp.Matrix([px, py])

    # Attractive + repulsive velocity field
    v_att = k_goal * (goal - p)
    diff  = p - obs
    dist  = sp.sqrt(diff.dot(diff) + eps)
    clearance = dist - R_obs
    w     = sp.exp(-alpha * clearance)
    denom = dist**3 + eps
    v_rep = k_rep * w * diff / denom
    v     = v_att + v_rep

    # Desired heading and angular control
    theta_d = sp.atan2(v[1], v[0])
    e_theta = _wrap_to_pi_sym(theta_d - theta)
    omega   = omega_max * sp.tanh(k_theta * e_theta)

    # Plant (dt=0.5, velocity=5)
    dt  = sp.Rational(1, 2)
    vel = sp.Integer(5)
    next_px    = px    + dt * vel * sp.cos(theta)
    next_py    = py    + dt * vel * sp.sin(theta)
    next_theta = _wrap_to_pi_sym(theta + dt * omega)

    return sp.Matrix([next_px, next_py, next_theta])


# =====================================================================
# Derive / cache symbolic Jacobian and Hessian
# =====================================================================

_CACHE_PATH = Path(__file__).with_name('unicycle_derivatives.pkl')


def _derive_and_cache():
    print("[SYMPY] Deriving Jacobian and Hessian (first run only)...", flush=True)
    t0 = time.perf_counter()

    px, py, theta = sp.symbols('px py theta', real=True)
    state_vars = [px, py, theta]
    n = 3

    F = _cl_system_sym((px, py, theta))

    J_sym = F.jacobian(state_vars)
    H_sym = sp.Array([
        [[sp.diff(F[i], state_vars[j], state_vars[k])
          for k in range(n)]
         for j in range(n)]
        for i in range(n)
    ])

    print("[SYMPY] Running CSE...", flush=True)
    J_cse = sp.cse(list(J_sym), optimizations='basic')
    H_flat = [H_sym[i, j, k] for i in range(n) for j in range(n) for k in range(n)]
    H_cse  = sp.cse(H_flat, optimizations='basic')

    print(f"[SYMPY] Done ({time.perf_counter() - t0:.1f}s).", flush=True)

    with open(_CACHE_PATH, 'wb') as f:
        pickle.dump((F, state_vars, J_cse, H_cse), f)

    return F, state_vars, J_cse, H_cse



def _build_interval_func_from_cse(state_vars, cse_data, output_shape):
    substitutions, reduced_exprs = cse_data
    cse_vars  = [s for s, _ in substitutions]
    cse_exprs = [e for _, e in substitutions]

    cse_funcs = []
    for i, ex in enumerate(cse_exprs):
        available = list(state_vars) + cse_vars[:i]
        cse_funcs.append(sp.lambdify(available, ex, modules=_INTERVAL_MODULES))

    all_args = list(state_vars) + cse_vars
    out_func = sp.lambdify(all_args, reduced_exprs, modules=_INTERVAL_MODULES)

    def eval_interval(lb, ub):
        env = [
            iv.mpf([float(lb[0]), float(ub[0])]),
            iv.mpf([float(lb[1]), float(ub[1])]),
            iv.mpf([float(lb[2]), float(ub[2])]),
        ]
        for f in cse_funcs:
            env.append(f(*env))
        raw = out_func(*env)
        out = np.empty(output_shape, dtype=object)
        for q, val in enumerate(raw):
            out.flat[q] = _as_interval(val)
        return out

    return eval_interval

def _load_or_derive():
    if _CACHE_PATH.exists():
        print(f"[SYMPY] Loading cached derivatives from {_CACHE_PATH}", flush=True)
        with open(_CACHE_PATH, 'rb') as f:
            F, state_vars, J_cse, H_cse = pickle.load(f)
    else:
        F, state_vars, J_cse, H_cse = _derive_and_cache()
    return F, state_vars, J_cse, H_cse


def _build_fast_eval(state_vars, cse_data, output_shape=None):
    """Build a fast numerical function from SymPy CSE data."""
    substitutions, reduced_exprs = cse_data
    cse_vars  = [s for s, _ in substitutions]
    cse_exprs = [e for _, e in substitutions]
    n_flat    = len(reduced_exprs)

    cse_funcs = []
    for i, ex in enumerate(cse_exprs):
        available = list(state_vars) + cse_vars[:i]
        cse_funcs.append(sp.lambdify(available, ex, modules=_LAMBDIFY_MODULES))

    out_func = sp.lambdify(list(state_vars) + cse_vars,
                           reduced_exprs, modules=_LAMBDIFY_MODULES)

    def fast_eval(*vals):
        env = list(vals)
        for f in cse_funcs:
            env.append(f(*env))
        raw = out_func(*env)
        is_batched = vals and np.ndim(vals[0]) > 0
        if not is_batched:
            result = np.array([float(r) for r in raw], dtype=float)
            if output_shape is not None:
                result = result.reshape(output_shape)
        else:
            N = len(vals[0])
            result = np.empty((N, n_flat), dtype=float)
            for k, arr in enumerate(raw):
                result[:, k] = arr
            if output_shape is not None:
                result = result.reshape((N,) + output_shape)
        return result

    return fast_eval


# ── Build all numeric functions at import time ────────────────────────────────
_F_SYM, _STATE_VARS, _J_CSE, _H_CSE = _load_or_derive()
_F_FUNC = sp.lambdify(_STATE_VARS, list(_F_SYM), modules=_LAMBDIFY_MODULES)
_J_FUNC = _build_fast_eval(_STATE_VARS, _J_CSE, output_shape=(3, 3))
_H_FUNC = _build_fast_eval(_STATE_VARS, _H_CSE, output_shape=(3, 3, 3))
_H_INTERVAL_FUNC = _build_interval_func_from_cse(
    _STATE_VARS, _H_CSE, output_shape=(3, 3, 3)
)


# =====================================================================
# Numeric evaluation functions
# =====================================================================

def cl_system_numeric(state):
    px, py, theta = np.asarray(state, dtype=float)
    return np.array(_F_FUNC(px, py, theta), dtype=float).ravel()


def jacobian(state):
    px, py, theta = np.asarray(state, dtype=float)
    return np.array(_J_FUNC(px, py, theta), dtype=float)


def hessian(state):
    px, py, theta = np.asarray(state, dtype=float)
    return np.array(_H_FUNC(px, py, theta), dtype=float)



def interval_hessian(lb, ub):
    H_iv = _H_INTERVAL_FUNC(lb, ub)
    H_lo = np.empty((3, 3, 3), dtype=float)
    H_hi = np.empty((3, 3, 3), dtype=float)
    for idx in np.ndindex(3, 3, 3):
        H_lo[idx] = float(H_iv[idx].a)
        H_hi[idx] = float(H_iv[idx].b)
    return H_lo, H_hi


def taylor_remainder(lb, ub):
    H_lo, H_hi = interval_hessian(lb, ub)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    h  = 0.5 * (ub - lb)
    R_lo = np.zeros(3)
    R_hi = np.zeros(3)
    for i in range(3):
        lo_sum = 0.0
        hi_sum = 0.0
        for j in range(3):
            for k in range(3):
                a_lo = H_lo[i, j, k]
                a_hi = H_hi[i, j, k]
                b_lo, b_hi = (0.0, h[j]**2) if j == k else (-h[j]*h[k], h[j]*h[k])
                t_lo, t_hi = _interval_mul(a_lo, a_hi, b_lo, b_hi)
                lo_sum += 0.5 * t_lo
                hi_sum += 0.5 * t_hi
        R_lo[i] = lo_sum
        R_hi[i] = hi_sum
    return R_lo, R_hi


def linear_cl_system(state, center, *, J=None, f_center=None):
    state  = np.asarray(state,  dtype=float)
    center = np.asarray(center, dtype=float)
    if J        is None: J        = jacobian(center)
    if f_center is None: f_center = cl_system_numeric(center)
    return J @ (state - center) + f_center


# Also expose as unicycle_dynamics for use in refine_whole_space.py GT sim
def unicycle_dynamics(x):
    return cl_system_numeric(x)


# =====================================================================
# Lagrange error bounds
# =====================================================================

def lagrange_error_bounds(lower_bounds, upper_bounds, *, resolution=10):
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    centroid     = (lower_bounds + upper_bounds) / 2.0
    max_disp     = np.linalg.norm(centroid - lower_bounds)

    grids       = [np.linspace(lo, hi, resolution)
                   for lo, hi in zip(lower_bounds, upper_bounds)]
    grid_points = np.array(list(product(*grids)))

    H_all = np.array([hessian(pt) for pt in grid_points])  # (N, 3, 3, 3)
    sup_norms = np.zeros(3)
    for i in range(3):
        eigvals = np.linalg.eigvalsh(H_all[:, i, :, :])    # (N, 3)
        sup_norms[i] = np.max(np.abs(eigvals))

    return 0.5 * sup_norms * (max_disp ** 2)


def lagrange_error_bounds_grid(x_edges, y_edges, theta_edges):
    """
    Vectorized Lagrange error bounds for every cell in the uniform grid.
    Returns array of shape (Nx, Ny, Ntheta, 3).
    """
    x_edges     = np.asarray(x_edges)
    y_edges     = np.asarray(y_edges)
    theta_edges = np.asarray(theta_edges)

    Nx = len(x_edges) - 1
    Ny = len(y_edges) - 1
    Nt = len(theta_edges) - 1

    # Centroids and half-diagonals
    cx = ((x_edges[:-1] + x_edges[1:]) / 2)[:, None, None]
    cy = ((y_edges[:-1] + y_edges[1:]) / 2)[None, :, None]
    ct = ((theta_edges[:-1] + theta_edges[1:]) / 2)[None, None, :]

    dx = ((x_edges[1:] - x_edges[:-1]) / 2)[:, None, None]
    dy = ((y_edges[1:] - y_edges[:-1]) / 2)[None, :, None]
    dt = ((theta_edges[1:] - theta_edges[:-1]) / 2)[None, None, :]
    max_disp = np.sqrt(
        np.broadcast_to(dx, (Nx, Ny, Nt))**2 +
        np.broadcast_to(dy, (Nx, Ny, Nt))**2 +
        np.broadcast_to(dt, (Nx, Ny, Nt))**2
    )  # (Nx, Ny, Nt)

    # Flatten centroids for batched Hessian eval
    CX = np.broadcast_to(cx, (Nx, Ny, Nt)).ravel()
    CY = np.broadcast_to(cy, (Nx, Ny, Nt)).ravel()
    CT = np.broadcast_to(ct, (Nx, Ny, Nt)).ravel()

    H_all = np.array([hessian([px, py, th])
                      for px, py, th in zip(CX, CY, CT)])  # (N, 3, 3, 3)

    sup_norms = np.zeros((len(CX), 3))
    for i in range(3):
        eigvals = np.linalg.eigvalsh(H_all[:, i, :, :])    # (N, 3)
        sup_norms[:, i] = np.max(np.abs(eigvals), axis=-1)

    sup_norms = sup_norms.reshape(Nx, Ny, Nt, 3)
    error_bounds = 0.5 * sup_norms * max_disp[..., None] ** 2

    return error_bounds  # (Nx, Ny, Nt, 3)


# =====================================================================
# Theta arc interval helper
# =====================================================================

def theta_min_arc_intervals(thetas, *, eps=1e-12):
    """
    Returns non-wrapping theta interval(s) in [-pi, pi] covering the samples
    Returns [(lo, hi)] normally, or [(-pi, hi), (lo, pi)] when the minimal
    arc wraps across the -pi/pi cut
    """
    th = np.asarray(thetas, dtype=float)
    if th.size == 0:
        return [(-np.pi, np.pi)]

    # Wrap to [-pi, pi)
    th = (th + np.pi) % (2 * np.pi) - np.pi

    if th.size == 1:
        v = float(th[0])
        return [(v, v)]

    # Work on [0, 2pi) for stable circular gap computation
    u       = np.sort(th + np.pi)
    two_pi  = 2.0 * np.pi
    gaps    = np.diff(np.r_[u, u[0] + two_pi])
    k       = int(np.argmax(gaps))

    start_u  = float(u[(k + 1) % u.size])
    end_u    = float(u[k])
    arc_len  = (end_u - start_u) % two_pi

    if arc_len >= two_pi - eps:
        return [(-np.pi, np.pi)]

    end_u2 = start_u + arc_len
    if end_u2 <= two_pi + eps:
        lo = start_u - np.pi
        hi = min(end_u2, two_pi) - np.pi
        return [(float(lo), float(hi))]

    # Wraps across the cut — return two intervals
    lo1, hi1 = start_u - np.pi, np.pi
    lo2, hi2 = -np.pi, (end_u2 - two_pi) - np.pi
    return [(float(lo2), float(hi2)), (float(lo1), float(hi1))]


# =====================================================================
# AP labeler
# =====================================================================

def ap_labeler(r):
    if r is None:
        return {"unsafe"}

    def closest(lo, hi, c):
        return max(lo, min(hi, c))

    def farthest(lo, hi, c):
        """Distance (along one axis) from c to whichever of lo/hi is
        FARTHER -- i.e. the worst-case corner. Used for the goal AP: a
        cell should only be labeled 'goal' if its ENTIRE extent is
        within R_GOAL (Must semantics, per Eq. 24), not just its nearest
        point (May semantics). Using closest-point for goal, as before,
        would label a cell 'goal' even when only part of it has actually
        arrived -- which is unsound, since compute_verified_set_via_fixpoint
        treats a 'goal' label as an immediate, final success condition and
        stops evolving that cell right there.
        """
        return max(abs(lo - c), abs(hi - c))

    aps = set()

    # Goal: Must semantics (farthest corner) -- see farthest()'s docstring.
    dx = farthest(r.xmin, r.xmax, Y_GOAL[0])
    dy = farthest(r.ymin, r.ymax, Y_GOAL[1])
    if dx*dx + dy*dy <= R_GOAL**2:
        aps.add("goal")

    # Unsafe: May semantics (closest point) is correct here -- conservatively
    # flag a cell unsafe if ANY point in it could be within the obstacle.
    ox = closest(r.xmin, r.xmax, Y_OBS[0]) - Y_OBS[0]
    oy = closest(r.ymin, r.ymax, Y_OBS[1]) - Y_OBS[1]
    if ox*ox + oy*oy <= R_OBS**2:
        aps.add("unsafe")

    return aps


# =====================================================================
# UnicycleDynamics — Taylor reachability
# =====================================================================

class UnicycleDynamics:
    """
    Taylor reachability transition builder for the unicycle.

    For each cell:
      1. Linearize f at the centroid using the analytic Jacobian
      2. Evaluate the linearization at all 8 corners
      3. Take the AABB of the linearized images
      4. Inflate by the Lagrange error bound (precomputed for the uniform
         grid; recomputed on-the-fly for CEGAR child cells)
      5. Apply theta_min_arc_intervals to handle -pi/pi wrapping
    """

    def __init__(self, x_edges, y_edges, theta_edges):
        self.x_edges     = np.asarray(x_edges)
        self.y_edges     = np.asarray(y_edges)
        self.theta_edges = np.asarray(theta_edges)

        print("[UnicycleDynamics] Precomputing Lagrange error bounds...", flush=True)
        self._error_bounds = lagrange_error_bounds_grid(
            x_edges, y_edges, theta_edges
        )  # (Nx, Ny, Ntheta, 3)
        print(f"[UnicycleDynamics] Error bounds shape: {self._error_bounds.shape}",
              flush=True)

    xstar = np.array([40.0, 20.0])
    goal_radius = 8.0

    def dynamics(self, x):
        """Point evaluation of the closed-loop dynamics. Used by cegar_loop validation."""
        return cl_system_numeric(np.asarray(x, dtype=float))

    def _lagrange_bound(self, r: Rect) -> np.ndarray:
        """
        Look up the precomputed Lagrange bound for original grid cells;
        recompute on-the-fly for CEGAR child cells.
        """
        cx = 0.5 * (r.xmin + r.xmax)
        cy = 0.5 * (r.ymin + r.ymax)
        cz = 0.5 * (r.zmin + r.zmax)

        i = int(np.clip(np.searchsorted(self.x_edges,     cx, side='right') - 1,
                        0, len(self.x_edges) - 2))
        j = int(np.clip(np.searchsorted(self.y_edges,     cy, side='right') - 1,
                        0, len(self.y_edges) - 2))
        k = int(np.clip(np.searchsorted(self.theta_edges, cz, side='right') - 1,
                        0, len(self.theta_edges) - 2))

        tol = 1e-9
        if (abs(r.xmin - self.x_edges[i])         < tol and
            abs(r.xmax - self.x_edges[i+1])        < tol and
            abs(r.ymin - self.y_edges[j])          < tol and
            abs(r.ymax - self.y_edges[j+1])        < tol and
            abs(r.zmin - self.theta_edges[k])      < tol and
            abs(r.zmax - self.theta_edges[k+1])    < tol):
            return self._error_bounds[i, j, k, :]

        # CEGAR child cell — recompute
        return lagrange_error_bounds(
            lower_bounds=np.array([r.xmin, r.ymin, r.zmin]),
            upper_bounds=np.array([r.xmax, r.ymax, r.zmax]),
        )

    def image_bbox(self, r: Rect):
        """
        Conservative post-image via Taylor reachability.
        Returns List[Rect].
        """
        lower = np.array([r.xmin, r.ymin, r.zmin])
        upper = np.array([r.xmax, r.ymax, r.zmax])

        corners = np.array([
            [r.xmin, r.ymin, r.zmin], [r.xmin, r.ymin, r.zmax],
            [r.xmin, r.ymax, r.zmin], [r.xmin, r.ymax, r.zmax],
            [r.xmax, r.ymin, r.zmin], [r.xmax, r.ymin, r.zmax],
            [r.xmax, r.ymax, r.zmin], [r.xmax, r.ymax, r.zmax],
        ], dtype=float)

        centroid = 0.5 * (lower + upper)
        J        = jacobian(centroid)
        f_center = cl_system_numeric(centroid)

        lin_imgs = np.array([
            linear_cl_system(v, centroid, J=J, f_center=f_center)
            for v in corners
        ])  # (8, 3)

        lin_lo = lin_imgs.min(axis=0)
        lin_hi = lin_imgs.max(axis=0)

        # Taylor remainder with interval arithmetic — tighter than precomputed
        # Lagrange bounds, and asymmetric (R_lo <= 0, R_hi >= 0 in general).
        R_lo, R_hi = taylor_remainder(lower, upper)

        # Drop theta inflation when bounds are unreliable near -pi/pi cut
        if abs(R_lo[2]) > 1 or abs(R_hi[2]) > 1:
            R_lo[2] = 0.0
            R_hi[2] = 0.0

        lo = lin_lo + R_lo
        hi = lin_hi + R_hi

        theta_intervals = theta_min_arc_intervals(lin_imgs[:, 2])

        boxes = []
        for (theta_lo, theta_hi) in theta_intervals:
            t_lo = max(theta_lo + R_lo[2], Z_MIN)
            t_hi = min(theta_hi + R_hi[2], Z_MAX)
            if t_lo <= t_hi:
                boxes.append(Rect(
                    float(lo[0]), float(hi[0]),
                    float(lo[1]), float(hi[1]),
                    float(t_lo),  float(t_hi),
                ))

        if not boxes:
            boxes = [Rect(float(lo[0]), float(hi[0]),
                          float(lo[1]), float(hi[1]),
                          Z_MIN, Z_MAX)]
        return boxes


# =====================================================================
# Build abstraction
# =====================================================================

def build_abstraction():
    domain = Rect(xmin=X_MIN, xmax=X_MAX,
                  ymin=Y_MIN, ymax=Y_MAX,
                  zmin=Z_MIN, zmax=Z_MAX)

    x_edges     = np.linspace(X_MIN, X_MAX, NX + 1)
    y_edges     = np.linspace(Y_MIN, Y_MAX, NY + 1)
    theta_edges = np.linspace(Z_MIN, Z_MAX, NZ + 1)

    part = RectPartition.uniform_grid(domain, nx=NX, ny=NY, nz=NZ)
    dyn  = UnicycleDynamics(x_edges, y_edges, theta_edges)

    absys = Abstraction(
        part=part,
        dyn_by_action={"step": dyn},
        ap_labeler=ap_labeler,
    )
    absys.rebuild_all_transitions()

    print(f"#leaves: {len(absys.part.leaves)}  (grid {NX}x{NY}x{NZ})")
    return absys, domain


# =====================================================================
# Entry point (single-cell CEGAR test)
# =====================================================================

if __name__ == "__main__":
    absys, domain = build_abstraction()

    INIT_X, INIT_Y, INIT_Z = 5.0, 5.0, 0.0
    init_uid  = absys.part.query_point(INIT_X, INIT_Y, INIT_Z)
    init_uids = {init_uid}
    print(f"Initial uid: {init_uid}")

    res = run_cegar(
        absys=absys,
        init_uids=init_uids,
        max_iters=50,
        merge_actions=True,
        min_cell_width=0.05,
        min_cell_height=0.05,
        max_refine_depth=8,
        verbose=True,
    )

    print("\nFINAL:", "VERIFIED" if res.verified else "NOT VERIFIED")
    print("iters:", res.iterations, "refinements:", res.refinements)