# =====================================================================
# Description: contains the necessary tools for modeling the unicycle
# system as a finite transition system with robust Taylor reachability.
# Utilizes PyModelChecking for abstraction as a Kripke structure.
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

A_GLOBAL = np.array([[0.8, -0.3],
                     [0.3,  0.8]])
XSTAR = np.array([5.0, 5.0])

# =====================================================================
# Closed-loop dynamical system (numpy and JAX)
# =====================================================================

def dynamics(state, x_star):
    A = A_GLOBAL
    state = np.asarray(state, dtype=float)
    x_star = np.asarray(x_star)
    return (state - x_star) @ A.T + x_star

def dynamics_jax(states, x_star):
    A = jnp.asarray(A_GLOBAL)
    states = jnp.asarray(states)
    x_star = jnp.asarray(x_star)
    return (states - x_star) @ A.T + x_star

def dynamics_sim(state):
    A = A_GLOBAL
    x_star = XSTAR
    state = np.asarray(state, dtype=float)
    return (state - x_star) @ A.T + x_star

# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    domain_lb = np.array([-10.0, -10.0])
    domain_ub = np.array([10.0, 10.0])
    goal_radius = 2.0
    x_values = np.linspace(domain_lb[0], domain_ub[0], 17)
    y_values = np.linspace(domain_lb[1], domain_ub[1], 17)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    positions = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    displacement = dynamics_sim(positions) - positions

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.quiver(
        grid_x,
        grid_y,
        displacement[:, 0].reshape(grid_x.shape),
        displacement[:, 1].reshape(grid_y.shape),
        color="#264653",
        angles="xy",
        scale_units="xy",
        scale=3.0,
        width=0.003,
        headwidth=4,
        headlength=5,
        zorder=2,
    )
    goal_handle = plt.Circle(
        XSTAR,
        goal_radius,
        facecolor="#2a9d8f",
        edgecolor="#1d6f66",
        linewidth=1.5,
        alpha=0.35,
        label="Goal",
        zorder=3,
    )
    axis.add_patch(goal_handle)
    axis.set_xlabel("$x_1$")
    axis.set_ylabel("$x_2$")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(domain_lb[0], domain_ub[0])
    axis.set_ylim(domain_lb[1], domain_ub[1])
    # axis.legend(handles=[goal_handle], loc="upper left", frameon=True)
    plt.show()
