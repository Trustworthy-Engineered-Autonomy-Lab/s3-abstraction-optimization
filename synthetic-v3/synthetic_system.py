# =====================================================================
# Description: contains the necessary tools for modeling the unicycle
# system as a finite transition system with robust Taylor reachability.
# Utilizes PyModelChecking for abstraction as a Kripke structure.
# =====================================================================

# =====================================================================
# Libraries
# =====================================================================

import numpy as np


# =====================================================================
# Closed-loop dynamical system
# =====================================================================

def dynamics(state, x_star):
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    state = np.asarray(state, dtype=float)
    return (state - x_star) @ A.T + x_star
