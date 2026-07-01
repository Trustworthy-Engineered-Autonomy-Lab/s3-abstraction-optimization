# =====================================================================
# Description: main script for training abstraction parameters
# =====================================================================

# =====================================================================
# Libraries for the unicycle system
# =====================================================================

import unicycle_abstraction as ua
import simulation_analysis as sa
import unicycle_objectives as uo
import verification_tools as vt
import unicycle_optimizers as u_opt
import unicycle_system_jax as usj
import jax
import jax.numpy as jnp
import numpy as np
import pyModelChecking as pmc
import pickle as pkl
import matplotlib.pyplot as plt
from unicycle_visualization import visualize_slice


# =====================================================================
# Main training and evaluation program
# =====================================================================

if __name__ == "__main__":

    # Fixed abstraction and environment settings
    abstraction_shape = [50, 50, 50]
    domain_lb = np.array([0.0, 0.0, -np.pi])
    domain_ub = np.array([50.0, 50.0, np.pi])

    # Define the initial state subset domain
    init_domain_lb = np.array([0.0, 0.0, -np.pi/4])
    init_domain_ub = np.array([50.0, 50.0, np.pi/4])

    gt_reach_fname = "unicycle-taylor/unicycle_gt_reach_regions_100.pkl"

    # Initialize abstraction parameters
    key = jax.random.PRNGKey(0)
    sigma_u = 1.0
    # u1 = jnp.zeros((abstraction_shape[0],))  # initial uniform spacing
    # u2 = jnp.zeros((abstraction_shape[1],))
    # u3 = jnp.zeros((abstraction_shape[2],))
    key, k_u1, k_u2 = jax.random.split(key, 3)
    u1 = sigma_u * jax.random.normal(k_u1, (abstraction_shape[0],))
    u2 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[1],))
    u3 = sigma_u * jax.random.normal(k_u2, (abstraction_shape[2],))
    params = jnp.concatenate([u1, u2, u3])

    # ##### New code for successor-bound optimization
    # Estimate the Lipschitz constant for the unicycle dynamics over the domain.

    L = usj.estimate_lipschitz_array(domain_lb, domain_ub)
    J = uo.succ_bound(params,
                  shape=abstraction_shape,
                  domain_lb=domain_lb,
                  domain_ub=domain_ub,
                  L=L,
                  p=20.0)
    print(f"Initial successor bound J = {J}")

    # Use the successor-bound estimate as the optimization objective.
    def succ_cost(p, *, shape, domain_lb, domain_ub):
        return uo.succ_bound(
            p,
            shape=shape,
            domain_lb=domain_lb,
            domain_ub=domain_ub,
            L=L,
            p=20.0,
        )

    params_opt, cost_history, grad_norm_history = u_opt.gradient_descent(
        params,
        succ_cost,
        shape=abstraction_shape,
        domain_lb=domain_lb,
        domain_ub=domain_ub,
        steps=50,
        lr=1e-1,
        grad_clip=1e3,
        print_every=10,
        record_every=10,
    )
    print(f"Optimized successor-bound cost = {cost_history[-1]:.4f}")
    params = params_opt
    J_opt = uo.succ_bound(
        params,
        shape=abstraction_shape,
        domain_lb=domain_lb,
        domain_ub=domain_ub,
        L=L,
        p=20.0,
    )
    print(f"Post-optimization successor bound J = {J_opt}")

    # End-to-end verification and visualization workflow
    x_edges, y_edges, theta_edges = uo.extract_grid_params(
        params,
        abstraction_shape,
        domain_lb,
        domain_ub,
    )

    init_ids = ua.init_ids_from_aabb(
        init_domain_lb,
        init_domain_ub,
        x_edges,
        y_edges,
        theta_edges,
    )

    kripke_components = ua.build_abstraction(
        x_edges,
        y_edges,
        theta_edges,
        verbose=False,
        L=L,
    )
    kripke_structure = pmc.Kripke(
        S=kripke_components["kripke_states"],
        S0=init_ids,
        R=list(kripke_components["kripke_transitions"]),
        L=kripke_components["kripke_labels"],
    )

    sat_init_states = vt.model_check_kripke(kripke_structure, log_time=True)
    print(f"Verified safe initial states: {len(sat_init_states)} / {len(init_ids)}")

    with open(gt_reach_fname, "rb") as f:
        gt_reach_regions = pkl.load(f)
    ground_truth_check = vt.check_ground_truth_fast(
        params,
        abstraction_shape,
        domain_lb,
        domain_ub,
        gt_reach_regions,
    )

    # Compute recall over initial states from the ground truth labels
    init_goal_states = [s for s in init_ids if ground_truth_check.get(s, [None])[0] == "goal"]
    verified_goal_states = [s for s in sat_init_states if ground_truth_check.get(s, [None])[0] == "goal"]
    recall = (
        len(verified_goal_states) / len(init_goal_states)
        if len(init_goal_states) > 0
        else float('nan')
    )
    print(f"Initial-state recall against ground truth: {recall:.4f}")

    fig, _, counts = visualize_slice(
        x_edges,
        y_edges,
        theta_edges,
        [abstraction_shape[2] // 2],
        sat_init_states,
        ground_truth_check=ground_truth_check,
        title="Verification visualization",
    )
    fig.savefig("unicycle-taylor/verification_plot.png", dpi=200, bbox_inches="tight")
    print(f"Visualization counts: {counts}")
    plt.show()

    # # ### Shivani New  Code to test the workflow




