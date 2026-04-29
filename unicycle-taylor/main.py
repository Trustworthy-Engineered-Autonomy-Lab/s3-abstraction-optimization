# =====================================================================
# Description: main script for training abstraction parameters
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import unicycle_abstraction as ua
import numpy as np
import pyModelChecking as pmc
import pyModelChecking.CTL as CTL
import verification_tools as vt


# =====================================================================
# Main training or evaluation program
# =====================================================================

if __name__ == "__main__":

    # Fixed abstraction and environment settings
    abstraction_shape = [100, 100, 100]
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])

    # Define the initial domain
    init_domain_lb = np.array([20.0, 0.0, 0.0])
    init_domain_ub = np.array([50.0, 40.0, 0.0])

    # Initialize the grid parameters
    x_edges = np.linspace(domain_lb[0], domain_ub[0], abstraction_shape[0]+1)
    y_edges = np.linspace(domain_lb[1], domain_ub[1], abstraction_shape[1]+1)
    theta_edges = np.linspace(domain_lb[2], domain_ub[2], abstraction_shape[2]+1)

    # Conservatively build the initial abstract state set
    _, init_states = ua.init_cells_to_ids(init_domain_lb, init_domain_ub, x_edges, y_edges, theta_edges)

    # Build the initial Kripke components
    kripke_components = ua.build_abstraction(x_edges, y_edges, theta_edges, verbose=True)

    # Build the full kripke structure
    kripke_structure = pmc.Kripke(S=kripke_components['kripke_states'],
                                  S0=init_states,
                                  R=list(kripke_components['kripke_transitions']),
                                  L=kripke_components['kripke_labels'])
    
    # Run verification
    sat_init_states, sat_prop = vt.model_check_kripke(kripke_structure)
    print(sat_prop)
