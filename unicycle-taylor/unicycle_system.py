# =====================================================================
# Description: closed-loop unicycle pipeline
# =====================================================================

# =====================================================================
# Libraries 
# =====================================================================

import numpy as np
import jax
import jax.numpy as jnp
from itertools import product


# =====================================================================
# Shared unicycle parameters
# =====================================================================

DELTA_T = 0.5
VELOCITY = 5.0
MAX_CONTROL = np.pi / 4

GOAL_CENTER = (40.0, 20.0)
OBS_CENTER = (25.0, 25.0)
OBS_RADIUS = 5.0

K_GOAL = 0.3
K_REP = 300.0
ALPHA = 0.1
K_THETA = 2.0
OMEGA_MAX = np.pi / 4
CONTROLLER_EPS = 1e-6


# =====================================================================
# End-to-end unicycle system (numpy)
# =====================================================================

def unicycle_plant(
        state,
        control,
        *,
    max_control=MAX_CONTROL):
    """
    Description: Open-loop discrete equations of motion of Unicycle
    """

    state = np.asarray(state)
    control = np.asarray(control)

    # Unicycle model parameters
    delta_t = np.asarray(DELTA_T, dtype=state.dtype)
    velocity = np.asarray(VELOCITY, dtype=state.dtype)
    max_control = np.asarray(max_control, dtype=state.dtype)
    pose_x = state[..., 0]
    pose_y = state[..., 1]
    theta = state[..., 2]

    # Apply control bounds (heading rate of change)
    control = np.clip(control, -max_control, max_control)

    # Update the state
    next_pose_x = pose_x + (delta_t * velocity * np.cos(theta))
    next_pose_y = pose_y + (delta_t * velocity * np.sin(theta))
    next_theta = wrap_to_pi(theta + (delta_t * control))  # normalize angle to [-pi, pi]

    return np.stack([next_pose_x, next_pose_y, next_theta], axis=-1)


def state_controller(
    state,
    *,
    goal_center,
    obs_center,
    obs_radius,
    k_goal=K_GOAL,
    k_rep=K_REP,
    alpha=ALPHA,
    k_theta=K_THETA,
    omega_max=OMEGA_MAX,
    eps=CONTROLLER_EPS,
):
    """
    Deterministic repulsion-attraction smooth controller for Dubins/unicycle
    """

    # Format all inputs as JAX arrays
    state = np.asarray(state)
    goal_center = np.asarray(goal_center, dtype=state.dtype)
    obs_center = np.asarray(obs_center, dtype=state.dtype)
    obs_radius = np.asarray(obs_radius, dtype=state.dtype)
    k_goal = np.asarray(k_goal, dtype=state.dtype)
    k_rep = np.asarray(k_rep, dtype=state.dtype)
    alpha = np.asarray(alpha, dtype=state.dtype)
    k_theta = np.asarray(k_theta, dtype=state.dtype)
    omega_max = np.asarray(omega_max, dtype=state.dtype)
    eps = np.asarray(eps, dtype=state.dtype)
    px = state[..., 0]
    py = state[..., 1]
    theta = state[..., 2]
    p = np.stack([px, py], axis=-1)

    # Attractive component
    v_att = k_goal * (goal_center - p)

    # Repulsive component
    diff = p - obs_center
    dist = np.sqrt(np.sum(diff**2, axis=-1) + eps)
    clearance = dist - obs_radius
    w = np.exp(-alpha * clearance)
    denom = (dist**3 + eps)
    v_rep = k_rep * w[..., None] * diff / denom[..., None]

    # Compute desire heading angle and smooth control input
    v = v_att + v_rep
    theta_d = np.arctan2(v[..., 1], v[..., 0])
    e_theta = wrap_to_pi(theta_d - theta)
    omega = omega_max * np.tanh(k_theta * e_theta)

    return omega

def cl_system(
        state,
        *,
    obs_center=np.array(OBS_CENTER, dtype=np.float32),
    obs_radius=np.array(OBS_RADIUS, dtype=np.float32),
    goal_center=np.array(GOAL_CENTER, dtype=np.float32)):
    """
    Closed-loop wrapper for the plant with the state controller
    """

    state = np.asarray(state)

    control_input = state_controller(
        state,
        goal_center=goal_center,
        obs_center=obs_center,
        obs_radius=obs_radius,
        k_goal=K_GOAL,
        k_rep=K_REP,
        alpha=ALPHA,
        k_theta=K_THETA,
        omega_max=OMEGA_MAX,
        eps=CONTROLLER_EPS,
    )
    return unicycle_plant(state, control_input)


# =====================================================================
# Helper methods (numpy)
# =====================================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# =====================================================================
# End-to-end unicycle system (jax.numpy)
# =====================================================================

def unicycle_plant_jax(state, control, control_bound=MAX_CONTROL):
    """
    JAX-configured open-loop unicycle dynamics.
    """

    state = jnp.asarray(state)
    control = jnp.asarray(control)

    # Unicycle model parameters
    delta_t = jnp.asarray(DELTA_T, dtype=state.dtype)
    velocity = jnp.asarray(VELOCITY, dtype=state.dtype)
    control_bound = jnp.asarray(control_bound, dtype=state.dtype)
    pose_x, pose_y, theta = state

    # Apply control bounds (heading rate of change)
    control = jnp.clip(control, -control_bound, control_bound)

    # Update the state
    next_pose_x = pose_x + (delta_t * velocity * jnp.cos(theta))
    next_pose_y = pose_y + (delta_t * velocity * jnp.sin(theta))
    next_theta = wrap_to_pi_jax(theta + (delta_t * control))  # normalize angle to [-pi, pi]

    return jnp.stack([next_pose_x, next_pose_y, next_theta])

def state_controller_jax(
    state,
    *,
    goal_center,
    obstacle_centers,
    obstacle_radii,
    k_goal=K_GOAL,
    k_rep=K_REP,
    alpha=ALPHA,
    k_theta=K_THETA,
    omega_max=OMEGA_MAX,
    eps=CONTROLLER_EPS,
    ):
    """
    JAX-configured deterministic & smooth controller for unicycle.
    """

    state = jnp.asarray(state)
    goal_center = jnp.asarray(goal_center, dtype=state.dtype)
    obstacle_centers = jnp.asarray(obstacle_centers, dtype=state.dtype)
    obstacle_radii = jnp.asarray(obstacle_radii, dtype=state.dtype)

    k_goal = jnp.asarray(k_goal, dtype=state.dtype)
    k_rep = jnp.asarray(k_rep, dtype=state.dtype)
    alpha = jnp.asarray(alpha, dtype=state.dtype)
    k_theta = jnp.asarray(k_theta, dtype=state.dtype)
    omega_max = jnp.asarray(omega_max, dtype=state.dtype)
    eps = jnp.asarray(eps, dtype=state.dtype)

    px, py, theta = state
    p = jnp.stack([px, py])

    # Attractive component (toward goal)
    v_att = k_goal * (goal_center - p)

    # Repulsive component (sum over discs)
    # obstacle_centers: [N, 2], obstacle_radii: [N]
    diff = p[None, :] - obstacle_centers
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + eps)  # smoothed distance to center
    clearance = dist - obstacle_radii

    # Smooth activation: stronger when near obstacle, decays with clearance
    w = jnp.exp(-alpha * clearance)

    # Direction away from obstacle with smoothing in denom
    denom = (dist**3 + eps)[:, None]
    v_rep = jnp.sum((k_rep * w)[:, None] * diff / denom, axis=0)

    v = v_att + v_rep
    v_norm = jnp.linalg.norm(v)

    def _omega_when_ok(_):
        theta_d = jnp.arctan2(v[1], v[0])
        e_theta = wrap_to_pi_jax(theta_d - theta)
        return omega_max * jnp.tanh(k_theta * e_theta)

    omega = jax.lax.cond(v_norm < jnp.asarray(1e-9, dtype=state.dtype), lambda _: jnp.asarray(0.0, dtype=state.dtype), _omega_when_ok, operand=None)
    return omega

def cl_system_jax(state):
    """
    JAX-configured closed-loop unicycle dynamics.
    """

    state = jnp.asarray(state)
    obs_center = jnp.asarray(OBS_CENTER, dtype=state.dtype)
    obs_radius = jnp.asarray(OBS_RADIUS, dtype=state.dtype)
    goal_center = jnp.asarray(GOAL_CENTER, dtype=state.dtype)

    control_input = state_controller_jax(
        state,
        goal_center=goal_center,
        obstacle_centers=jnp.stack([obs_center], axis=0),
        obstacle_radii=jnp.stack([obs_radius], axis=0),
        k_goal=K_GOAL,
        k_rep=K_REP,
        alpha=ALPHA,
        k_theta=K_THETA,
        omega_max=OMEGA_MAX,
        eps=CONTROLLER_EPS,
    )
    return unicycle_plant_jax(state, control_input)


# =====================================================================
# Autogradient partial derivative functions for the closed-loop system
# =====================================================================

def jacobian_jax(state):
    """
    Computes and evaluates the nxn Jacobian of the closed-loop system
    """
    return jax.jacfwd(cl_system_jax)(state)


def hessian_jax(state):
    """
    Computes and evaluates the nxnxn Hessian of the closed-loop system
    """
    return jax.jacfwd(jax.jacfwd(cl_system_jax))(state)


# =====================================================================
# Linearized system and Lagrange bound helper functions
# =====================================================================

def linear_cl_system_jax(
        state,
        center,
        *,
        J: jnp.ndarray = None
        ) -> jnp.ndarray:
    """
    Linearized closed-loop system around x*
    """
    if J is None:
        J = jacobian_jax(center)
    return J @ (state - center) + cl_system_jax(center)

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
    H_all = jax.vmap(hessian_jax)(grid_points)

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
# Helper methods (jax.numpy)
# =====================================================================

def wrap_to_pi_jax(angle):
    angle = jnp.asarray(angle)
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi