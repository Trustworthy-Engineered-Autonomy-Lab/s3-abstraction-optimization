# =====================================================================
# Description: system functions
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp


# =====================================================================
# Dynamics (numpy)
# =====================================================================

def cl_system(
    state,
    *,
    dt=0.05,
    goal=np.array([3.2, 2.7]),
    obs=np.array([-1.2, -0.3]),
    obs_radius=0.75,
    k_goal=0.45,
    k_rep=0.18,
    alpha=3.0,
    k_vort=1.25,
    sigma_vort=1.1,
    gate_amp=2.0,
    gate_beta=8.0,
    gate_x=-2.2,
    gate_y=1.0,
    gate_sigma=0.8,
    eps=1e-6,
):
    """
    Discrete-time closed-loop 2D navigation-like system.

    Ambient domain: Y = [-6, 6]^2
    Verification/safe domain: X = [-4, 4]^2
    """

    z = np.asarray(state)
    x = z[..., 0]
    y = z[..., 1]

    # Attractive flow
    u_att = k_goal * (goal - z)

    # Obstacle geometry
    diff = z - obs
    dist = np.sqrt(np.sum(diff**2, axis=-1) + eps)

    # Smooth repulsive field
    clearance = dist - obs_radius
    rep_weight = np.exp(-alpha * clearance)
    u_rep = k_rep * rep_weight[..., None] * diff / (dist[..., None]**3 + eps)

    # Local vortex around obstacle
    Rdiff = np.stack([-diff[..., 1], diff[..., 0]], axis=-1)
    vort_weight = np.exp(-np.sum(diff**2, axis=-1) / (2.0 * sigma_vort**2))
    u_vort = k_vort * vort_weight[..., None] * Rdiff / (dist[..., None] + eps)

    # Sharp nonlinear gate / shear layer
    gate_profile = np.exp(-((y - gate_y) ** 2) / (2.0 * gate_sigma**2))
    u_gate_x = gate_amp * np.tanh(gate_beta * (x - gate_x)) * gate_profile
    u_gate = np.stack([u_gate_x, np.zeros_like(u_gate_x)], axis=-1)

    u = u_att + u_rep + u_vort + u_gate

    return z + dt * u


# =====================================================================
# Dynamics (jax.numpy)
# =====================================================================

def cl_system(
    state,
    *,
    dt=0.05,
    goal=np.array([3.2, 2.7]),
    obs=np.array([-1.2, -0.3]),
    obs_radius=0.75,
    k_goal=0.45,
    k_rep=0.18,
    alpha=3.0,
    k_vort=1.25,
    sigma_vort=1.1,
    gate_amp=2.0,
    gate_beta=8.0,
    gate_x=-2.2,
    gate_y=1.0,
    gate_sigma=0.8,
    eps=1e-6,
):
    """
    Discrete-time closed-loop 2D navigation-like system.

    Ambient domain: Y = [-6, 6]^2
    Verification/safe domain: X = [-4, 4]^2
    """

    z = np.asarray(state)
    x = z[..., 0]
    y = z[..., 1]

    # Attractive flow
    u_att = k_goal * (goal - z)

    # Obstacle geometry
    diff = z - obs
    dist = np.sqrt(np.sum(diff**2, axis=-1) + eps)

    # Smooth repulsive field
    clearance = dist - obs_radius
    rep_weight = np.exp(-alpha * clearance)
    u_rep = k_rep * rep_weight[..., None] * diff / (dist[..., None]**3 + eps)

    # Local vortex around obstacle
    Rdiff = np.stack([-diff[..., 1], diff[..., 0]], axis=-1)
    vort_weight = np.exp(-np.sum(diff**2, axis=-1) / (2.0 * sigma_vort**2))
    u_vort = k_vort * vort_weight[..., None] * Rdiff / (dist[..., None] + eps)

    # Sharp nonlinear gate / shear layer
    gate_profile = np.exp(-((y - gate_y) ** 2) / (2.0 * gate_sigma**2))
    u_gate_x = gate_amp * np.tanh(gate_beta * (x - gate_x)) * gate_profile
    u_gate = np.stack([u_gate_x, np.zeros_like(u_gate_x)], axis=-1)

    u = u_att + u_rep + u_vort + u_gate

    return z + dt * u


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    domain_lb = np.array([-4, -4])
    domain_ub = np.array([4, 4])

    grid_x = np.linspace(domain_lb[0], domain_ub[0], 25)
    grid_y = np.linspace(domain_lb[1], domain_ub[1], 25)
    x1, x2 = np.meshgrid(grid_x, grid_y)
    u = np.zeros_like(x1)
    v = np.zeros_like(x2)

    for row in range(x1.shape[0]):
        for col in range(x1.shape[1]):
            state = np.array([x1[row, col], x2[row, col]])
            next_state = cl_system(state)
            u[row, col] = next_state[0] - state[0]
            v[row, col] = next_state[1] - state[1]

    plt.figure(figsize=(6, 6))
    plt.quiver(x1, x2, u, v)
    plt.show()
