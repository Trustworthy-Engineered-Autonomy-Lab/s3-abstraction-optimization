# =====================================================================
# Description: contains the unicycle system open-loop dynamics, the
# differentiable controller, and the closed-loop dynamics w/controller.
# Jacobian and Hessian are derived analytically via SymPy (once, at
# import time) and lambdified into fast NumPy-callable functions.
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


# =====================================================================
# Helper functions
# =====================================================================

def wrap_to_pi_diff(angle):
    """Smooth wrap to [-pi, pi] via arctan(tan()) — differentiable everywhere
    except at ±pi boundaries, which are never visited in practice."""
    return 2 * sp.atan(sp.tan(angle / 2))


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
 
# Cache file lives alongside this source file so it is shared across
# all scripts that import this module.
_CACHE_PATH = Path(__file__).with_name('unicycle_derivatives.pkl')
 

def _derive_symbolic_derivatives():
    """
    Derives the symbolic Jacobian and Hessian of the closed-loop system.
    """

    print("[USS] Deriving symbolic gradients...", flush=True)
    _t0 = time.perf_counter()

    px, py, theta = sp.symbols('px py theta', real=True)
    state_vars = [px, py, theta]
    n = len(state_vars)

    F = cl_system((px, py, theta))

    J_sym = F.jacobian(state_vars)
    print(f"[USS] Jacobian: {sp.count_ops(J_sym)} ops")

    H_sym = sp.Array([
        [[sp.diff(F[i], state_vars[j], state_vars[k])
          for k in range(n)]
         for j in range(n)]
        for i in range(n)
    ])
    print(f"[USS] Hessian: {sp.count_ops(H_sym)} ops")

    print("[USS] Running CSE on Jacobian...")
    J_cse = sp.cse(list(J_sym), optimizations='basic')   # (substitutions, reduced)
    print("[USS] Running CSE on Hessian...")
    H_flat = [H_sym[i, j, k] for i in range(n) for j in range(n) for k in range(n)]
    H_cse = sp.cse(H_flat, optimizations='basic')

    print(f"[USS] Done ({time.perf_counter() - _t0:.2f}s).", flush=True)

    # Return symbolic CSE data — all SymPy objects, fully picklable
    return F, state_vars, J_cse, H_cse


def _build_funcs_from_cse(state_vars, cse_data, output_shape=None):
    """
    Given the state variables and CSE data (substitutions and reduced expressions),
    builds fast evaluation functions using lambdify. The returned function takes
    numerical values for the state variables and evaluates the expressions efficiently.
    """
    
    modules = [
        {'DiracDelta': lambda x: np.zeros_like(np.asarray(x, dtype=float)),
         'Heaviside':  lambda x: np.where(np.asarray(x) >= 0, 1.0, 0.0)},
        'numpy',
    ]

    substitutions, reduced_exprs = cse_data
    cse_vars  = [s for s, _ in substitutions]
    cse_exprs = [e for _, e in substitutions]
    all_args  = list(state_vars) + cse_vars
    n_flat    = len(reduced_exprs)

    cse_funcs = []
    for i, ex in enumerate(cse_exprs):
        available = list(state_vars) + cse_vars[:i]
        cse_funcs.append(sp.lambdify(available, ex, modules=modules))

    out_func = sp.lambdify(all_args, reduced_exprs, modules=modules)

    def fast_eval(*vals):
        env = list(vals)
        for f in cse_funcs:
            env.append(f(*env))

        raw = out_func(*env)   # list of n_flat scalars OR list of n_flat arrays

        # Detect batched (vector) vs scalar call from the *inputs*, not outputs.
        # Some output entries may be constants (ndim==0) even in batched mode
        # (e.g. second derivatives of linear terms), so checking raw[0] is unreliable.
        is_batched = vals and np.ndim(vals[0]) > 0
        if not is_batched:
            # Scalar input: raw is [scalar, scalar, ...]
            result = np.array([float(r) for r in raw], dtype=float)  # (n_flat,)
            if output_shape is not None:
                result = result.reshape(output_shape)
        else:
            # Batched input: raw is [array(N,) or scalar, ...]
            # Scalar constants broadcast automatically via assignment.
            N = len(vals[0])
            result = np.empty((N, n_flat), dtype=float)
            for k, arr in enumerate(raw):
                result[:, k] = arr                      # (N, n_flat)
            if output_shape is not None:
                result = result.reshape((N,) + output_shape)

        return result

    return fast_eval


def _load_or_derive():
    """
    Loads cached CSE data if available; otherwise derives symbolic Jacobian and
    Hessian, runs CSE, and caches the results to disk for future imports.
    Executed on import.
    """
    if _CACHE_PATH.exists():
        print(f"[USS] Loading cached CSE data from {_CACHE_PATH}", flush=True)
        with open(_CACHE_PATH, 'rb') as f:
            F, state_vars, J_cse, H_cse = pickle.load(f)
    else:
        F, state_vars, J_cse, H_cse = _derive_symbolic_derivatives()
        print(f"[USS] Caching CSE data to {_CACHE_PATH}", flush=True)
        with open(_CACHE_PATH, 'wb') as f:
            pickle.dump((F, state_vars, J_cse, H_cse), f)

    # Re-lambdify from CSE data (fast — no large expression printing)
    print("[USS] Lambdifying from CSE data...", flush=True)
    J_func = _build_funcs_from_cse(state_vars, J_cse, output_shape=(3, 3))
    H_func = _build_funcs_from_cse(state_vars, H_cse, output_shape=(3, 3, 3))

    return F, state_vars, J_func, H_func
 

_F, _STATE_VARS, _J_FUNC, _H_FUNC = _load_or_derive()


# =====================================================================
# Jacobian and Hessian evaluation functions
# =====================================================================

def jacobian(state):
    """
    Evaluates the analytic 3x3 Jacobian of the closed-loop system at `state`.
    Derived symbolically (once at import); evaluation is O(1) arithmetic.
    """
    px, py, theta = state
    return np.array(_J_FUNC(px, py, theta), dtype=float)


def hessian(state):
    """
    Evaluates the analytic 3x3x3 Hessian of the closed-loop system at `state`.
    Derived symbolically (once at import); evaluation is O(1) arithmetic.
    """
    px, py, theta = state
    return np.array(_H_FUNC(px, py, theta), dtype=float)


# =====================================================================
# Linearized system and Lagrange bound helper functions
# =====================================================================

def linear_cl_system(state, center, *, J=None):
    """
    Linearized closed-loop system around x*
    """
    state  = np.asarray(state)
    center = np.asarray(center)
    if J is None:
        J = jacobian(center)
    f_center = np.array(_F.subs(list(zip(_STATE_VARS, center))).evalf().tolist(), dtype=float).ravel()
    return J @ (state - center) + f_center


def sup_hessian_norms(lower_bounds, upper_bounds, *, resolution=10):
    """
    Approximates the supremum of the spectral norms of the Hessian over a range
    """

    # Build sampling grid
    grids       = [np.linspace(lo, hi, resolution) for lo, hi in zip(lower_bounds, upper_bounds)]
    grid_points = np.array(list(product(*grids)))

    # Evaluate analytic Hessian over all grid points in one batched call
    H_all = np.array([hessian(point) for point in grid_points])  # (N, n, n, n)

    # Compute spectral norm for all i at once
    def all_spectral_norms(H):
        # H is (n, n, n); H[i, :, :] is the Hessian matrix of the i-th output
        eigvals = np.linalg.eigvalsh(H)            # (n, n) eigenvalues
        return np.max(np.abs(eigvals), axis=-1)    # (n,) spectral norms

    # Vectorize over grid points: (N, n)
    all_norms = np.array([all_spectral_norms(H) for H in H_all])

    # Take supremum over grid points: (n,)
    return np.max(all_norms, axis=0)


def lagrange_error_bounds(lower_bounds, upper_bounds, *, resolution=10):
    """
    Computes the Lagrange error bound for the linear approximation of the
    closed-loop system
    """

    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    # Determine max possible displacement within the cell
    centroid         = (lower_bounds + upper_bounds) / 2.0
    max_displacement = np.linalg.norm(centroid - lower_bounds)

    # Approximate supremum of Hessian spectral norms
    sup_norms = sup_hessian_norms(lower_bounds, upper_bounds, resolution=resolution)

    # Lagrange error bound: (1/2) * sup_norm * (max_displacement^2)
    return 0.5 * sup_norms * (max_displacement ** 2)


def _hessian_vectorized(px, py, th):
    """
    px, py, th: 1D arrays of length N
    returns: (N, 3, 3, 3)
    """
    return _H_FUNC(px, py, th)


def lagrange_error_bounds_grid(x_edges, y_edges, theta_edges, *, resolution=10):
    """
    Computes Lagrange error bounds for all cells in a 3D grid in one vectorized pass.
    Returns: error_bounds array of shape (Nx-1, Ny-1, Ntheta-1, 3)
    """

    Nx = len(x_edges) - 1
    Ny = len(y_edges) - 1
    Nt = len(theta_edges) - 1

    # --- Build all cell centroids and half-diagonals upfront ---
    x_lo, x_hi       = x_edges[:-1], x_edges[1:]
    y_lo, y_hi       = y_edges[:-1], y_edges[1:]
    t_lo, t_hi       = theta_edges[:-1], theta_edges[1:]

    # Cell centroids: (Nx, Ny, Nt, 3)
    cx = (x_lo[:, None, None] + x_hi[:, None, None]) / 2
    cy = (y_lo[None, :, None] + y_hi[None, :, None]) / 2
    ct = (t_lo[None, None, :] + t_hi[None, None, :]) / 2

    # Max displacement from centroid to corner (same for all cells if uniform)
    dx = (x_hi - x_lo)[:, None, None] / 2
    dy = (y_hi - y_lo)[None, :, None] / 2
    dt = (t_hi - t_lo)[None, None, :] / 2
    max_disp = np.sqrt(dx**2 + dy**2 + dt**2)  # (Nx, Ny, Nt)

    # --- Sample sup Hessian norm per cell ---
    # For small cells with resolution=10, sample around centroid ± half-width
    # We do ONE batched Hessian evaluation over all centroids (fast approximation)
    # For tighter bounds, increase resolution here.
    
    # cx/cy/ct have shapes (Nx,1,1), (1,Ny,1), (1,1,Nt) — must broadcast to full grid first
    CX = np.broadcast_to(cx, (Nx, Ny, Nt))
    CY = np.broadcast_to(cy, (Nx, Ny, Nt))
    CT = np.broadcast_to(ct, (Nx, Ny, Nt))
    centroids = np.stack([CX.ravel(), CY.ravel(), CT.ravel()], axis=-1)  # (N, 3)

    # Evaluate Hessian at all centroids in one vectorized call
    # H_all: (N, 3, 3, 3)
    px_arr  = centroids[:, 0]
    py_arr  = centroids[:, 1]
    th_arr  = centroids[:, 2]

    # Vectorized hessian over the full batch
    H_all = _hessian_vectorized(px_arr, py_arr, th_arr)  # (N, 3, 3, 3)

    # Spectral norm per output dimension: max |eigenvalue| of H[i,:,:] for each i
    # H_all[:,i,:,:] is the (3,3) Hessian of the i-th output
    sup_norms = np.zeros((len(centroids), 3))
    for i in range(3):
        Hi = H_all[:, i, :, :]                          # (N, 3, 3)
        eigvals = np.linalg.eigvalsh(Hi)                 # (N, 3)
        sup_norms[:, i] = np.max(np.abs(eigvals), axis=-1)  # (N,)

    sup_norms = sup_norms.reshape(Nx, Ny, Nt, 3)        # (Nx, Ny, Nt, 3)

    # Lagrange bound: (1/2) * M * r^2
    error_bounds = 0.5 * sup_norms * max_disp[..., None] ** 2  # (Nx, Ny, Nt, 3)

    return error_bounds


# =====================================================================
# Section for testing the above methods
# =====================================================================

if __name__ == "__main__":

    num_x_intervals = 100
    num_y_intervals = 100
    num_theta_intervals = 100

    x_edges     = np.linspace(0.0, 50.0, num_x_intervals + 1)
    y_edges     = np.linspace(0.0, 50.0, num_y_intervals + 1)
    theta_edges = np.linspace(-np.pi, np.pi, num_theta_intervals + 1)

    error_bounds = lagrange_error_bounds_grid(x_edges, y_edges, theta_edges)

    import matplotlib.pyplot as plt

    # Find the theta slice closest to 0
    theta_idx = 74

    # error_bounds shape: (Nx, Ny, Nt, 3); take the theta slice and compute the norm over dim 3
    slice_2d = error_bounds[:, :, theta_idx, :]          # (Nx, Ny, 3)
    norm_2d  = np.linalg.norm(slice_2d, axis=-1)         # (Nx, Ny)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(x_edges, y_edges, norm_2d.T, cmap='turbo', shading='flat',
                       edgecolors='k', linewidth=0.1)
    fig.colorbar(im, ax=ax, label='‖error bound‖')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Lagrange error bound norm')
    plt.tight_layout()
    plt.show()
