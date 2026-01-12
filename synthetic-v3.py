# Libraries
import numpy as np
import grid_plot_tools as gpt
from model_checking_tools import make_kripke_from_params, model_check_kripke, check_ground_truth


def false_negative_rate(true_safe_states, checked_safe_states):
    false_negative_states = {s for s in true_safe_states if s not in checked_safe_states}
    denom = len(true_safe_states)
    fnr = (len(false_negative_states) / denom) if denom > 0 else float('nan')
    return fnr, false_negative_states

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
    theta = -np.pi / 3
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
    n1_internal = 100
    n2_internal = 100
    y1_params_ends = np.linspace(y1_lo, y1_hi, n1_internal + 2) # include boundaries
    y2_params_ends = np.linspace(y2_lo, y2_hi, n2_internal + 2) #

    ground_truth_check = check_ground_truth(x_domain, x_star, radius,
                                            y1_params_ends, y2_params_ends, M)
    # print(ground_truth_check)

    kripke_structure = make_kripke_from_params(x_domain, x_star, radius,
                                               y1_params_ends, y2_params_ends, M)
    sat_init_states = model_check_kripke(kripke_structure)
    # print(sat_init_states)

    # Compute false negative rate
    checked_safe_states = set(sat_init_states)
    true_safe_states = {s for s, v in ground_truth_check.items() if v == 'goal'}
    fnr, false_negative_states = false_negative_rate(true_safe_states, checked_safe_states)
    print(f"False Negative Rate (FNR): {fnr:.4f}")

    gpt.plot_x_grid_false_negative_map(
        M,
        y1_params_ends,
        y2_params_ends,
        sat=sat_init_states,
        false_negative_states=false_negative_states,
        x_domain=x_domain,
        x_star=x_star,
        radius=radius,
    )

    # Old plot for reference:
    # gpt.plot_x_grid_satisfaction(M, y1_params_ends, y2_params_ends,
    #                              kripke_structure=kripke_structure, sat=sat_init_states,
    #                              x_domain=x_domain, goal_ap="g",
    #                              x_star=x_star, radius=radius)

    # # Plot the grids
    # gpt.grid_plotter(M, y1_params_ends, y2_params_ends, x1_min, x1_max, x2_min, x2_max)

