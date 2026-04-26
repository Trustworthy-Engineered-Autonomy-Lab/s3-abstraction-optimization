# =====================================================================
# Description: script to perform various analyses on the output of
# the abstraction codebase for the unicycle system.
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================


import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Section for testing the above methods
# =====================================================================


if __name__ == "__main__":

    # Load Kripke components
    with open("kripke_components.pkl", "rb") as f:
        kripke_components = pkl.load(f)

