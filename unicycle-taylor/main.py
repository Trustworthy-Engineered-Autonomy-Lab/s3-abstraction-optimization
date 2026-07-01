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

    # =====================================================================
    # Step 1: Initialize abstraction parameters with uniform spacing
    # =====================================================================
    # For 3 dimensions (x, y, theta), initialize gap parameters with zeros
    # (zero gap parameters correspond to uniform spacing)
    u1 = jnp.zeros((abstraction_shape[0],))
    u2 = jnp.zeros((abstraction_shape[1],))
    u3 = jnp.zeros((abstraction_shape[2],))
    params = jnp.concatenate([u1, u2, u3])

    print("=" * 70)
    print("Step 1: Initialized abstraction with uniform spacing")
    print("=" * 70)

    # =====================================================================
    # Step 2: Evaluate initial cost using succ_estimate
    # =====================================================================
    initial_cost = uo.succ_estimate(
        params,
        shape=abstraction_shape,
        domain_lb=domain_lb,
        domain_ub=domain_ub
    )
    print(f"Initial succ_estimate cost: {initial_cost:.6f}")

    # =====================================================================
    # Step 3: Optimize abstraction parameters using succ_estimate as cost
    # =====================================================================
    def succ_cost(p, *, shape, domain_lb, domain_ub):
        """Cost function: use succ_estimate as the optimization objective"""
        return uo.succ_estimate(
            p,
            shape=shape,
            domain_lb=domain_lb,
            domain_ub=domain_ub,
        )

    print("\nOptimizing abstraction parameters...")
    params_opt, cost_history, grad_norm_history = u_opt.gradient_descent(
        params,
        succ_cost,
        shape=abstraction_shape,
        domain_lb=domain_lb,
        domain_ub=domain_ub,
        steps=100,
        lr=1e-2,
        grad_clip=1e3,
        print_every=20,
        record_every=1,
    )

    final_cost = cost_history[-1]
    print(f"\nOptimization complete!")
    print(f"Initial cost: {initial_cost:.6f}")
    print(f"Final cost:   {final_cost:.6f}")
    print(f"Cost reduction: {(initial_cost - final_cost) / initial_cost * 100:.2f}%")
    params = params_opt

    # =====================================================================
    # Step 4: Extract grid parameters and build abstraction
    # =====================================================================
    print("\n" + "=" * 70)
    print("Step 4: Building abstraction from optimized parameters")
    print("=" * 70)

    # Extract grid edge locations from optimized parameters
    x_edges, y_edges, theta_edges = uo.extract_grid_params(
        params,
        abstraction_shape,
        domain_lb,
        domain_ub,
    )
    print(f"Grid dimensions: {len(x_edges)-1} x {len(y_edges)-1} x {len(theta_edges)-1}")

    # Estimate Lipschitz constant for robust reachability analysis
    L = usj.estimate_lipschitz_array(domain_lb, domain_ub)
    print(f"Lipschitz constant estimate: {L}")

    # Identify initial states (abstract cells that overlap with init domain)
    init_ids = ua.init_ids_from_aabb(
        init_domain_lb,
        init_domain_ub,
        x_edges,
        y_edges,
        theta_edges,
    )
    print(f"Initial abstract states: {len(init_ids)}")

    # Build the Kripke structure (finite-state transition system)
    kripke_components = ua.build_abstraction(
        x_edges,
        y_edges,
        theta_edges,
        verbose=False,
    )

    kripke_structure = pmc.Kripke(
        S=kripke_components["kripke_states"],
        S0=init_ids,
        R=list(kripke_components["kripke_transitions"]),
        L=kripke_components["kripke_labels"],
    )
    print(f"Kripke structure built with {len(kripke_components['kripke_states'])} states")

    # =====================================================================
    # Step 5: Model checking verification
    # =====================================================================
    print("\n" + "=" * 70)
    print("Step 5: Running model checking verification")
    print("=" * 70)

    sat_init_states = vt.model_check_kripke(kripke_structure, log_time=True)
    print(f"Verified safe initial states: {len(sat_init_states)} / {len(init_ids)}")

    # =====================================================================
    # Step 6: Ground truth validation
    # =====================================================================
    print("\n" + "=" * 70)
    print("Step 6: Validating against ground truth")
    print("=" * 70)

    with open(gt_reach_fname, "rb") as f:
        gt_reach_regions = pkl.load(f)

    ground_truth_check = vt.check_ground_truth_fast(
        params,
        abstraction_shape,
        domain_lb,
        domain_ub,
        gt_reach_regions,
    )

    # Compute recall over initial states
    init_goal_states = [s for s in init_ids if ground_truth_check.get(s, [None])[0] == "goal"]
    verified_goal_states = [s for s in sat_init_states if ground_truth_check.get(s, [None])[0] == "goal"]
    recall = (
        len(verified_goal_states) / len(init_goal_states)
        if len(init_goal_states) > 0
        else float('nan')
    )
    print(f"Initial-state recall against ground truth: {recall:.4f}")
    print(f"Goal states verified: {len(verified_goal_states)} / {len(init_goal_states)}")

    # =====================================================================
    # Step 7: Visualization
    # =====================================================================
    print("\n" + "=" * 70)
    print("Step 7: Generating visualization")
    print("=" * 70)

    fig, _, counts = visualize_slice(
        x_edges,
        y_edges,
        theta_edges,
        [abstraction_shape[2] // 2],
        sat_init_states,
        ground_truth_check=ground_truth_check,
        title="Verification visualization (theta=0 slice)",
    )
    fig.savefig("unicycle-taylor/verification_plot.png", dpi=200, bbox_inches="tight")
    print("Visualization saved to: unicycle-taylor/verification_plot.png")
    plt.show()

    print("\n" + "=" * 70)
    print("Workflow complete!")
    print("=" * 70)




