# =====================================================================
# Description: tools for verifying a pyModelChecking Kripke structure
# and analyzing the quality of verification
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import unicycle_abstraction as ua
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
# Useful verification metrics
# =====================================================================

