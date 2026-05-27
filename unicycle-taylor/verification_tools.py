# =====================================================================
# Description: tools for verifying a pyModelChecking Kripke structure
# and analyzing the quality of verification
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import unicycle_abstraction as ua
import unicycle_objectives as uo
import numpy as np
import pyModelChecking as pmc
import pyModelChecking.CTL as CTL


# =====================================================================
# Kripke structure verification
# =====================================================================

def model_check_kripke(kripke_structure):
    """
    Returns the subset of initial states from which abstract
    trajectories satisfy the verification property. Also
    returns several verification metrics, including the sat-
    isfaction proportion (proportion of the initial states)
    that are verified.
    """

    phi = 'A (safe U goal)'
    sat = CTL.modelcheck(kripke_structure, phi)
    sat_set = set(sat)
    init_states = list(kripke_structure.S0)

    sat_init_states = [state for state in init_states if state in sat_set]
    sat_prop = len(sat_init_states)/len(init_states)

    return sat_init_states, sat_prop


# =====================================================================
# Model building helpers
# =====================================================================

def build_and_verify_from_params(
    params,
    shape,
    domain_lb,
    domain_ub,
    init_domain_lb,
    init_domain_ub,
    verbose=False):
    """
    Simple helper to compute verification metrics on a set of parameters.
    """

    x_edges, y_edges, theta_edges = uo.extract_grid_params(params, shape, domain_lb, domain_ub)
    edges = [x_edges, y_edges, theta_edges]

    _, init_states = ua.init_cells_to_ids(init_domain_lb, init_domain_ub, x_edges, y_edges, theta_edges)

    kripke_components = ua.build_abstraction(x_edges, y_edges, theta_edges, verbose=verbose)
    kripke_structure = pmc.Kripke(S=kripke_components['kripke_states'],
                                  S0=init_states,
                                  R=list(kripke_components['kripke_transitions']),
                                  L=kripke_components['kripke_labels'])
    sat_init_states, sat_prop = model_check_kripke(kripke_structure)
    
    return sat_prop
    
