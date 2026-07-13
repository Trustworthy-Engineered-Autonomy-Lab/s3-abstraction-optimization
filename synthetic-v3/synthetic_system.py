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
