# =====================================================================
# Description: contains the unicycle system open-loop dynamics, the
# differentiable controler, and the closed-loop dynamics w/controller.
# The end-to-end system is JAX-compatible, enabling linearization
# around any critical point.
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================
import jax
import jax.numpy as jnp
import numpy as np
from itertools import product
import time


# =====================================================================
# End-to-end unicycle system
# =====================================================================

def unicycle_plant(
        state,
        control,
        *,
        max_control=np.pi/4):
    """
    Description: Open-loop discrete equations of motion of Unicycle
    """

    state = jnp.asarray(state)
    control = jnp.asarray(control)

    # Unicycle model parameters
    delta_t = jnp.asarray(0.5, dtype=state.dtype)
    velocity = jnp.asarray(5.0, dtype=state.dtype)
    max_control = jnp.asarray(max_control, dtype=state.dtype)
    pose_x, pose_y, theta = state

    # Apply control bounds (heading rate of change)
    control = jnp.clip(control, -max_control, max_control)

    # Update the state
    next_pose_x = pose_x + (delta_t * velocity * jnp.cos(theta))
    next_pose_y = pose_y + (delta_t * velocity * jnp.sin(theta))
    next_theta = wrap_to_pi(theta + (delta_t * control))  # normalize angle to [-pi, pi]

    return jnp.stack([next_pose_x, next_pose_y, next_theta])


def state_controller(
    state,
    *,
    goal_center,
    obs_center,
    obs_radius,
    k_goal=1.0,
    k_rep=8.0,
    alpha=0.6,
    k_theta=2.5,
    omega_max=np.pi/4,
    eps=1e-6,
):
    """
    Deterministic repulsion-attraction smooth controller for Dubins/unicycle
    """

    # Format all inputs as JAX arrays
    state = jnp.asarray(state)
    goal_center = jnp.asarray(goal_center, dtype=state.dtype)
    obs_center = jnp.asarray(obs_center, dtype=state.dtype)
    obs_radius = jnp.asarray(obs_radius, dtype=state.dtype)
    k_goal = jnp.asarray(k_goal, dtype=state.dtype)
    k_rep = jnp.asarray(k_rep, dtype=state.dtype)
    alpha = jnp.asarray(alpha, dtype=state.dtype)
    k_theta = jnp.asarray(k_theta, dtype=state.dtype)
    omega_max = jnp.asarray(omega_max, dtype=state.dtype)
    eps = jnp.asarray(eps, dtype=state.dtype)
    px, py, theta = state
    p = jnp.stack([px, py])

    # Attractive component
    v_att = k_goal * (goal_center - p)

    # Repulsive component
    diff = p - obs_center
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + eps)
    clearance = dist - obs_radius
    w = jnp.exp(-alpha * clearance)
    denom = (dist**3 + eps)
    v_rep = k_rep * w * diff / denom

    # Compute desire heading angle and smooth control input
    v = v_att + v_rep
    theta_d = jnp.arctan2(v[1], v[0])
    e_theta = wrap_to_pi(theta_d - theta)
    omega = omega_max * jnp.tanh(k_theta * e_theta)

    return omega


def cl_system(
        state,
        *,
        obs_center = jnp.array([25.0, 25.0], dtype=jnp.float32),
        obs_radius = jnp.array(5.0, dtype=jnp.float32),
        goal_center = jnp.array([40.0, 20.0], dtype=jnp.float32)):
    """
    Closed-loop wrapper for the plant with the state controller
    """

    state = jnp.asarray(state)

    control_input = state_controller(
        state,
        goal_center=goal_center,
        obs_center=obs_center,
        obs_radius=obs_radius,
        k_goal=0.3,
        k_rep=300.0,
        alpha=0.1,
        k_theta=2.0,
        omega_max=jnp.pi / 4,
    )
    return unicycle_plant(state, control_input)


# =====================================================================
# Autogradient partial derivative functions for the closed-loop system
# =====================================================================

def jacobian(state):
    """
    Computes and evaluates the nxn Jacobian of the closed-loop system
    """
    return jax.jacfwd(cl_system)(state)


def hessian(state):
    """
    Computes and evaluates the nxnxn Hessian of the closed-loop system
    """
    return jax.jacfwd(jax.jacfwd(cl_system))(state)


# =====================================================================
# Approximates the Lipschitz constant of the system
# =====================================================================

def estimate_lipschitz_array(
    domain_lb,
    domain_ub,
    points_per_dim=41,
    batch_size=8192
):
    """
    Grid estimate of the componentwise Lipschitz array
    """
    # domain bounds can be passed as numpy arrays or lists; keep them
    # as numpy for grid construction, but use JAX arrays for Jacobian
    # computations to avoid converting traced arrays to NumPy.
    domain_lb = np.asarray(domain_lb, dtype=float)
    domain_ub = np.asarray(domain_ub, dtype=float)

    axes = [
        np.linspace(domain_lb[i], domain_ub[i], points_per_dim)
        for i in range(3)
    ]

    mesh = np.meshgrid(*axes, indexing="ij")
    states = np.stack(mesh, axis=-1).reshape(-1, 3)

    batched_jacobian = jax.jit(jax.vmap(jacobian))

    L = jnp.zeros((3, 3), dtype=jnp.float32)

    for start in range(0, len(states), batch_size):
        state_batch = jnp.asarray(states[start:start + batch_size])

        # Keep J as a JAX array (no np.asarray conversion)
        J = batched_jacobian(state_batch)

        L_batch = jnp.max(jnp.abs(J), axis=0)
        L = jnp.maximum(L, L_batch)

    return L


# =====================================================================
# Linearized system and Lagrange bound helper functions
# =====================================================================

def linear_cl_system(
        state,
        center,
        *,
        J: jnp.ndarray = None
        ) -> jnp.ndarray:
    """
    Linearized closed-loop system around x*
    """
    if J is None:
        J = jacobian(center)
    return J @ (state - center) + cl_system(center)

def sup_hessian_norms(
        lower_bounds: jnp.ndarray,
        upper_bounds: jnp.ndarray,
        *,
        resolution=50
        ) -> jnp.ndarray:
    """
    Approximates the supremum of the spectral norms of the Hessian over a range
    """

    # Build sampling grid
    grids = [jnp.linspace(lo, hi, resolution) for lo, hi in zip(lower_bounds, upper_bounds)]
    grid_points = jnp.array(list(product(*grids)))

    # Vectorize hessian over all grid points
    H_all = jax.vmap(hessian)(grid_points)

    # Compute spectral norm for all i at once
    def all_spectral_norms(H):
        # H is (n, n, n), H[:, :, i] is the ith Hessian matrix
        # moveaxis to get shape (n, n, n) -> stack Hi's along first axis
        Hi_stack = jnp.moveaxis(H, -1, 0)  # (n, n, n) where Hi_stack[i] = H[:, :, i]
        eigvals = jax.vmap(jnp.linalg.eigvalsh)(Hi_stack)  # (n, n) eigenvalues
        return jnp.max(jnp.abs(eigvals), axis=-1)  # (n,) spectral norms

    # Vectorize over grid points: (N, n)
    all_norms = jax.vmap(all_spectral_norms)(H_all)

    # Take supremum over grid points: (n,)
    return jnp.max(all_norms, axis=0)
    

def lagrange_error_bounds(
        lower_bounds: jnp.ndarray,
        upper_bounds: jnp.ndarray,
        *,
        resolution=50
        ) -> jnp.ndarray:
    """
    Computes the Lagrange error bound for the linear approximation of the
    closed-loop system
    """

    # Determine max possible displacement within the cell
    centroid = (lower_bounds + upper_bounds) / 2.0
    max_displacement = jnp.linalg.norm(centroid - lower_bounds)

    # Approximate supremum of Hessian spectral norms
    sup_norms = sup_hessian_norms(lower_bounds,
                                  upper_bounds,
                                  resolution=resolution)

    # Lagrange error bound: (1/2) * sup_norm * (max_displacement^2)
    return 0.5 * sup_norms * (max_displacement ** 2)


# =====================================================================
# Helper functions
# =====================================================================

def wrap_to_pi(angle):
    angle = jnp.asarray(angle)
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


# =====================================================================
# Section for testing the above methods
# =====================================================================

if __name__ == "__main__":

    start_cpu = time.process_time()

    lower_bounds = np.array([-1.0,-1.0, -0.1])
    upper_bounds = np.array([1.0, 1.0, 0.1])

    error_bounds = lagrange_error_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        resolution=10)
    print(error_bounds)


    end_cpu = time.process_time()

    print(f"elapsed time: {end_cpu - start_cpu:.4f} seconds")
