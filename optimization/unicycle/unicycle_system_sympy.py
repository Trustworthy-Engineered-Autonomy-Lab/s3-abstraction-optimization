# =====================================================================
# Description: contains the unicycle system open-loop dynamics, the
# differentiable controller, and the closed-loop dynamics w/controller.
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================
import sympy as sp
import numpy as np
from itertools import product
import time
from pathlib import Path
import pickle
import mpmath as mp
from mpmath import iv


_LAMBDIFY_MODULES = [
    {
        'DiracDelta': lambda x: np.zeros_like(np.asarray(x, dtype=float)),
        'Heaviside': lambda x: np.where(np.asarray(x) >= 0, 1.0, 0.0),
    },
    'numpy',
]

def iv_tanh(x):
    return iv.mpf([mp.tanh(x.a), mp.tanh(x.b)])


def iv_atan(x):
    return iv.atan2(x, iv.mpf([1.0, 1.0]))


def as_interval(value):
    if hasattr(value, 'a') and hasattr(value, 'b'):
        return value

    scalar = float(value)
    return iv.mpf([scalar, scalar])

_INTERVAL_MODULES = [
    {
        "sin": iv.sin,
        "cos": iv.cos,
        "tan": iv.tan,
        "atan": iv_atan,
        "atan2": iv.atan2,
        "exp": iv.exp,
        "sqrt": iv.sqrt,
        "tanh": iv_tanh,
        "DiracDelta": lambda x: iv.mpf([0, 0]),
        "Heaviside": lambda x: iv.mpf([0, 1]),
    }
]

# =====================================================================
# Helper functions
# =====================================================================

def wrap_to_pi_diff(angle):
    """Smooth wrap to [-pi, pi] via arctan(tan())"""
    return 2 * sp.atan(sp.tan(angle / 2))
    # return angle

def wrap_to_pi_numeric(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def interval_mul(a_lo, a_hi, b_lo, b_hi):
    a_spans_zero = a_lo <= 0.0 <= a_hi
    b_spans_zero = b_lo <= 0.0 <= b_hi

    if ((not np.isfinite(a_lo)) or (not np.isfinite(a_hi))) and b_spans_zero:
        return -np.inf, np.inf
    if ((not np.isfinite(b_lo)) or (not np.isfinite(b_hi))) and a_spans_zero:
        return -np.inf, np.inf

    vals = [
        a_lo * b_lo,
        a_lo * b_hi,
        a_hi * b_lo,
        a_hi * b_hi,
    ]
    return min(vals), max(vals)

# =====================================================================
# Symbolic unicycle plant
# =====================================================================

def unicycle_plant(state, control, *, max_control=np.pi/4):
    """
    Open-loop discrete equations of motion of Unicycle (symbolic)
    """

    # Unicycle model parameters
    delta_t  = sp.Rational(1, 2)   # 0.5
    velocity = sp.Integer(5)

    pose_x, pose_y, theta = state

    # Update the state
    next_pose_x = pose_x + delta_t * velocity * sp.cos(theta)
    next_pose_y = pose_y + delta_t * velocity * sp.sin(theta)
    next_theta  = wrap_to_pi_diff(theta + delta_t * control)

    return sp.Matrix([next_pose_x, next_pose_y, next_theta])


# =====================================================================
# Symbolic state controller
# =====================================================================

def state_controller(
    state,
    *,
    goal_center,
    obs_center,
    obs_radius,
    k_goal  = 1.0,
    k_rep   = 8.0,
    alpha   = 0.6,
    k_theta = 2.5,
    omega_max = np.pi/4,
    eps     = 1e-6,
):
    """
    Deterministic repulsion-attraction smooth controller for Dubins/unicycle (symbolic)
    """

    px, py, theta = state
    p = sp.Matrix([px, py])

    goal = sp.Matrix(list(goal_center))
    obs  = sp.Matrix(list(obs_center))

    # Attractive component
    v_att = k_goal * (goal - p)

    # Repulsive component
    diff      = p - obs
    dist      = sp.sqrt(diff.dot(diff) + eps)
    clearance = dist - obs_radius
    w         = sp.exp(-alpha * clearance)
    denom     = dist**3 + eps
    v_rep     = k_rep * w * diff / denom

    # Compute desired heading angle and smooth control input
    v       = v_att + v_rep
    theta_d = sp.atan2(v[1], v[0])
    e_theta = wrap_to_pi_diff(theta_d - theta)
    omega   = omega_max * sp.tanh(k_theta * e_theta)

    return omega


# =====================================================================
# Symbolic closed-loop system
# =====================================================================

def cl_system(
    state,
    *,
    obs_center  = (25.0, 25.0),
    obs_radius  = 5.0,
    goal_center = (40.0, 20.0),
):
    """
    Closed-loop wrapper for the plant with the state controller (symbolic)
    """

    control_input = state_controller(
        state,
        goal_center = goal_center,
        obs_center  = obs_center,
        obs_radius  = obs_radius,
        k_goal      = 0.3,
        k_rep       = 300.0,
        alpha       = 0.1,
        k_theta     = 2.0,
        omega_max   = np.pi / 4,
    )
    return unicycle_plant(state, control_input)


# =====================================================================
# Analytic Jacobian and Hessian (derived once, cached to disk)
# =====================================================================
 
_CACHE_PATH = Path(__file__).parent / 'artifacts' / 'cache' / 'unicycle_derivatives.pkl'
 
def _derive_symbolic_derivatives():
    """
    Derives the symbolic Jacobian and Hessian of the closed-loop system.
    """

    print("Deriving symbolic gradients...", flush=True)
    _t0 = time.perf_counter()

    px, py, theta = sp.symbols('px py theta', real=True)
    state_vars = [px, py, theta]
    n = len(state_vars)

    F = cl_system((px, py, theta))

    J_sym = F.jacobian(state_vars)
    print(f"Jacobian: {sp.count_ops(J_sym)} ops")

    H_sym = sp.Array([
        [[sp.diff(F[i], state_vars[j], state_vars[k])
          for k in range(n)]
         for j in range(n)]
        for i in range(n)
    ])
    print(f"Hessian: {sp.count_ops(H_sym)} ops")

    print("Running CSE on Jacobian...")
    J_cse = sp.cse(list(J_sym), optimizations='basic')   # (substitutions, reduced)
    print("Running CSE on Hessian...")
    H_flat = [H_sym[i, j, k] for i in range(n) for j in range(n) for k in range(n)]
    H_cse = sp.cse(H_flat, optimizations='basic')

    print(f"Done ({time.perf_counter() - _t0:.2f}s).", flush=True)

    # Return symbolic CSE data — all SymPy objects, fully picklable
    return F, state_vars, J_cse, H_cse


def _build_funcs_from_cse(state_vars, cse_data, output_shape=None):
    """
    Given the state variables and CSE data builds fast evaluation
    functions using lambdify.
    """
    
    substitutions, reduced_exprs = cse_data
    cse_vars  = [s for s, _ in substitutions]
    cse_exprs = [e for _, e in substitutions]
    all_args  = list(state_vars) + cse_vars
    n_flat    = len(reduced_exprs)

    cse_funcs = []
    for i, ex in enumerate(cse_exprs):
        available = list(state_vars) + cse_vars[:i]
        cse_funcs.append(sp.lambdify(available, ex, modules=_LAMBDIFY_MODULES))

    out_func = sp.lambdify(all_args, reduced_exprs, modules=_LAMBDIFY_MODULES)

    def fast_eval(*vals):
        env = list(vals)
        for f in cse_funcs:
            env.append(f(*env))

        raw = out_func(*env)

        # Detect batched
        is_batched = vals and np.ndim(vals[0]) > 0
        if not is_batched:
            # Scalar input
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
            out.flat[q] = as_interval(val)

        return out

    return eval_interval


def _load_or_derive():
    """
    Loads cached CSE data if available; otherwise derives symbolic Jacobian and
    Hessian, runs CSE, and caches the results to disk for future use.
    """
    if _CACHE_PATH.exists():
        print(f"Loading cached CSE data from {_CACHE_PATH}", flush=True)
        with open(_CACHE_PATH, 'rb') as f:
            F, state_vars, J_cse, H_cse = pickle.load(f)
    else:
        F, state_vars, J_cse, H_cse = _derive_symbolic_derivatives()
        print(f"Caching CSE data to {_CACHE_PATH}", flush=True)
        with open(_CACHE_PATH, 'wb') as f:
            pickle.dump((F, state_vars, J_cse, H_cse), f)

    # Re-lambdify from CSE data
    print("Lambdifying from CSE data...", flush=True)
    J_func = _build_funcs_from_cse(state_vars, J_cse, output_shape=(3, 3))
    H_func = _build_funcs_from_cse(state_vars, H_cse, output_shape=(3, 3, 3))

    return F, state_vars, J_func, H_func, J_cse, H_cse
 

_F, _STATE_VARS, _J_FUNC, _H_FUNC, _J_CSE, _H_CSE = _load_or_derive()
_F_FUNC = sp.lambdify(_STATE_VARS, list(_F), modules=_LAMBDIFY_MODULES)
_H_INTERVAL_FUNC = _build_interval_func_from_cse(
    _STATE_VARS,
    _H_CSE,
    output_shape=(3, 3, 3)
)

# =====================================================================
# Jacobian and Hessian evaluation functions
# =====================================================================

def jacobian(state):
    """
    Evaluates the analytic 3x3 Jacobian of the closed-loop system.
    """
    px, py, theta = state
    return np.array(_J_FUNC(px, py, theta), dtype=float)


def hessian(state):
    """
    Evaluates the analytic 3x3x3 Hessian of the closed-loop system.
    """
    px, py, theta = state
    return np.array(_H_FUNC(px, py, theta), dtype=float)

def interval_hessian(lb, ub):
    H_iv = _H_INTERVAL_FUNC(lb, ub)

    H_lo = np.empty((3, 3, 3), dtype=float)
    H_hi = np.empty((3, 3, 3), dtype=float)

    for idx in np.ndindex(3, 3, 3):
        H_lo[idx] = float(H_iv[idx].a)
        H_hi[idx] = float(H_iv[idx].b)

    return H_lo, H_hi

def cl_system_numeric(state):
    """
    Evaluates the closed-loop system numerically.
    """
    px, py, theta = np.asarray(state, dtype=float)
    return np.array(_F_FUNC(px, py, theta), dtype=float).ravel()


# =====================================================================
# Linearized system and Lagrange bound helper functions
# =====================================================================

def linear_cl_system(state, center, *, J=None, f_center=None):
    """
    Linearized closed-loop system.
    """
    state  = np.asarray(state, dtype=float)
    center = np.asarray(center, dtype=float)
    if J is None:
        J = jacobian(center)
    if f_center is None:
        f_center = cl_system_numeric(center)
    return J @ (state - center) + f_center


def sup_hessian_norms(lower_bounds, upper_bounds, *, resolution=10):
    """
    Approximates the supremum of the spectral norms of the Hessian over a range
    """

    # Build sampling grid
    grids       = [np.linspace(lo, hi, resolution) for lo, hi in zip(lower_bounds, upper_bounds)]
    grid_points = np.array(list(product(*grids)))

    # Evaluate analytic Hessian over all grid points in one batched call
    H_all = np.array([hessian(point) for point in grid_points])

    # Compute spectral norm for all i at once
    def all_spectral_norms(H):
        eigvals = np.linalg.eigvalsh(H)
        return np.max(np.abs(eigvals), axis=-1)

    # Vectorize over grid points: (N, n)
    all_norms = np.array([all_spectral_norms(H) for H in H_all])

    # Take supremum over grid points: (n,)
    return np.max(all_norms, axis=0)


def lagrange_error_bounds(lower_bounds, upper_bounds, *, resolution=10):
    """
    Computes the Lagrange error bound for the linear approximation of the
    closed-loop system.
    """

    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    # Determine max possible displacement within the cell
    centroid         = (lower_bounds + upper_bounds) / 2.0
    max_displacement = np.linalg.norm(centroid - lower_bounds)
    sup_norms = sup_hessian_norms(lower_bounds, upper_bounds, resolution=resolution)

    return 0.5 * sup_norms * (max_displacement ** 2)


def _hessian_vectorized(px, py, th):
    return _H_FUNC(px, py, th)


def lagrange_error_bounds_grid(x_edges, y_edges, theta_edges, *, resolution=10):
    """
    Computes Lagrange error bounds for all cells in a 3D grid in one pass.
    Depracated (see taylor remainder)
    """

    Nx = len(x_edges) - 1
    Ny = len(y_edges) - 1
    Nt = len(theta_edges) - 1

    x_lo, x_hi       = x_edges[:-1], x_edges[1:]
    y_lo, y_hi       = y_edges[:-1], y_edges[1:]
    t_lo, t_hi       = theta_edges[:-1], theta_edges[1:]

    cx = (x_lo[:, None, None] + x_hi[:, None, None]) / 2
    cy = (y_lo[None, :, None] + y_hi[None, :, None]) / 2
    ct = (t_lo[None, None, :] + t_hi[None, None, :]) / 2

    dx = (x_hi - x_lo)[:, None, None] / 2
    dy = (y_hi - y_lo)[None, :, None] / 2
    dt = (t_hi - t_lo)[None, None, :] / 2
    max_disp = np.sqrt(dx**2 + dy**2 + dt**2)
    
    CX = np.broadcast_to(cx, (Nx, Ny, Nt))
    CY = np.broadcast_to(cy, (Nx, Ny, Nt))
    CT = np.broadcast_to(ct, (Nx, Ny, Nt))
    centroids = np.stack([CX.ravel(), CY.ravel(), CT.ravel()], axis=-1)

    px_arr  = centroids[:, 0]
    py_arr  = centroids[:, 1]
    th_arr  = centroids[:, 2]

    H_all = _hessian_vectorized(px_arr, py_arr, th_arr)

    sup_norms = np.zeros((len(centroids), 3))
    for i in range(3):
        Hi = H_all[:, i, :, :]
        eigvals = np.linalg.eigvalsh(Hi)
        sup_norms[:, i] = np.max(np.abs(eigvals), axis=-1)

    sup_norms = sup_norms.reshape(Nx, Ny, Nt, 3)

    error_bounds = 0.5 * sup_norms * max_disp[..., None] ** 2

    return error_bounds

# =====================================================================
# Faster error bounds
# =====================================================================

def taylor_remainder(lb, ub):
    """
    Computes interval Taylor remainder R for first-order Taylor model.
    """

    H_lo, H_hi = interval_hessian(lb, ub) # evaluate interval Hessian

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    h = 0.5 * (ub - lb)

    R_lo = np.zeros(3)
    R_hi = np.zeros(3)

    for i in range(3):
        lo_sum = 0.0
        hi_sum = 0.0

        for j in range(3):
            for k in range(3):

                # Hessian interval
                a_lo = H_lo[i, j, k]
                a_hi = H_hi[i, j, k]

                # Displacement product interval:
                if j == k:
                    b_lo = 0.0
                    b_hi = h[j] ** 2
                else:
                    b_lo = -h[j] * h[k]
                    b_hi =  h[j] * h[k]

                t_lo, t_hi = interval_mul(a_lo, a_hi, b_lo, b_hi)

                lo_sum += 0.5 * t_lo
                hi_sum += 0.5 * t_hi

        R_lo[i] = lo_sum
        R_hi[i] = hi_sum

    return R_lo, R_hi
