# Libraries
import numpy as np
import grid_plot_tools as gpt
from model_checking_tools import make_kripke_from_params, model_check_kripke

if __name__ == "__main__":

    # Affine dynamical system
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    x_star = np.array([5.0, 5.0])
    x1_min, x1_max = -10.0, 10.0
    x2_min, x2_max = -10.0, 10.0
    x_domain = (x1_min, x1_max, x2_min, x2_max)
    radius = 5.0

    # Define abstraction mapping parameters
    theta = -np.pi / 2
    s1, s2 = 1.0, 1.0
    h = 0.0
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    S = np.array([[s1, 0],
                  [0, s2]])
    H = np.array([[1, h],
                  [0, 1]])
    M = H @ S @ R 

    # Compute y-domain bounds from x-domain and transformation
    bounds_y, verts_y = gpt.get_yspace_bounds(M, x1_min, x1_max, x2_min, x2_max)
    y1_lo, y1_hi = bounds_y["y1"][0], bounds_y["y1"][1]
    y2_lo, y2_hi = bounds_y["y2"][0], bounds_y["y2"][1]

    # Uniform grid in y-space
    n1_internal = 49
    n2_internal = 49
    y1_params_ends = np.linspace(y1_lo, y1_hi, n1_internal + 2) # include boundaries
    y2_params_ends = np.linspace(y2_lo, y2_hi, n2_internal + 2) #

    kripke_structure = make_kripke_from_params(x_domain, x_star, radius,
                                               y1_params_ends, y2_params_ends, M)
    sat = model_check_kripke(kripke_structure)
    print(sat)

    gpt.plot_x_grid_satisfaction(M, y1_params_ends, y2_params_ends,
                                 kripke_structure=kripke_structure, sat=sat,
                                 x_domain=x_domain, goal_ap="g",
                                 x_star=x_star, radius=radius)

    # # Plot the grids
    # gpt.grid_plotter(M, y1_params_ends, y2_params_ends, x1_min, x1_max, x2_min, x2_max)