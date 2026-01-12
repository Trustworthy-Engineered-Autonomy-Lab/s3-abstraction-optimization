# Libraries
import numpy as np
import grid_plot_tools as gpt
from smc_objective_tools import objective, objective_jax
# from model_checking_tools import model_check_tessellation
import jax
import jax.numpy as jnp

CENTER = np.array([5.0, 5.0])

def adam_optimize(params, obj_fn, obj_kwargs, steps=2000, lr=1e-3,
                  b1=0.9, b2=0.999, eps=1e-8, clip=10.0, print_every=100):
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)

    value_and_grad = jax.jit(jax.value_and_grad(lambda p: obj_fn(p, **obj_kwargs)))

    for t in range(1, steps + 1):
        val, g = value_and_grad(params)

        # optional grad clip (helps a lot early on)
        gnorm = jnp.linalg.norm(g)
        g = jnp.where(gnorm > clip, g * (clip / (gnorm + 1e-12)), g)

        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)

        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)

        if (t % print_every) == 0 or t == 1:
            print(f"step {t:5d} | obj {float(val):.6f} | ||g|| {float(gnorm):.6f}")

    return params


if __name__ == "__main__":


    # Affine dynamical system
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    x_star = np.array([5.0, 5.0])
    x1_min, x1_max = -10.0, 10.0
    x2_min, x2_max = -10.0, 10.0
    x_domain = (x1_min, x1_max, x2_min, x2_max)
    radius = 2.0

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
    n1_internal = 10
    n2_internal = 10
    y1_params_ends = jnp.linspace(y1_lo, y1_hi, n1_internal + 2) # include boundaries
    y2_params_ends = jnp.linspace(y2_lo, y2_hi, n2_internal + 2) #

    # params = jnp.concatenate([u1, u2, jnp.array([theta, a1, a2, h])])

    cost = objective(x_domain, x_star, radius, y1_params_ends, y2_params_ends, M)
    print(f"Initial cost: {cost:.6f}")

    A_jax = jnp.array([[0.8, -0.3],
                       [0.3,  0.8]])

    def obj_for_grads(y1p, y2p, th, s1p, s2p, hp):
        return objective_jax(
            x_domain=x_domain,
            x_star=x_star,
            y1_params=y1p,
            y2_params=y2p,
            theta=th,
            s1=s1p,
            s2=s2p,
            h=hp,
            A=A_jax,
        )

    (gy1, gy2, gtheta, gs1, gs2, gh) = jax.grad(obj_for_grads, argnums=(0, 1, 2, 3, 4, 5))(
        y1_params_ends,
        y2_params_ends,
        theta,
        s1,
        s2,
        h,
    )
    print("||∂J/∂y1_params||:", float(jnp.linalg.norm(gy1)))
    print("||∂J/∂y2_params||:", float(jnp.linalg.norm(gy2)))
    print("∂J/∂theta:", float(gtheta))
    print("∂J/∂s1:", float(gs1))
    print("∂J/∂s2:", float(gs2))
    print("∂J/∂h:", float(gh))

    # # Define the system transition matrix
    # A = np.array([[0.8, -0.3],
    #               [0.3,  0.8]])
    
    # # Define the X-domain
    # x1_min, x1_max = -10.0, 10.0
    # x2_min, x2_max = -10.0, 10.0

    # # Initial tessellation parameters
    # n1_internal = 8
    # n2_internal = 8
    # key = jax.random.PRNGKey(0)
    # u1 = jnp.zeros((n1_internal,))
    # u2 = jnp.zeros((n2_internal,))
    # theta = -jnp.pi/4
    # a1, a2 = 0.0, 0.0
    # h = 0.0

    # # Compute y-domain bounds from x-domain and transformation
    # M = np.array(trans_matrix(theta, a1, a2, h))
    # bounds_y, verts_y = gpt.get_yspace_bounds(
    #     M,
    #     x1_min, x1_max,
    #     x2_min, x2_max
    # )
    # y1_lo, y1_hi = bounds_y["y1"][0], bounds_y["y1"][1]
    # y2_lo, y2_hi = bounds_y["y2"][0], bounds_y["y2"][1]
    
    # # Pack initial params; check initial objective + grad
    # params = jnp.concatenate([u1, u2, jnp.array([theta, a1, a2, h])])
    # # val, grad = jax.value_and_grad(diff_objective_reg)(
    # #     params,
    # #     A=A, center=CENTER,
    # #     y1_lo=y1_lo, y1_hi=y1_hi, y2_lo=y2_lo, y2_hi=y2_hi,
    # #     n1_internal=n1_internal, n2_internal=n2_internal,
    # #     tau=0.05, min_gap=0.0
    # # )
    
    # # Optimization
    # obj_kwargs = dict(
    # A=jnp.asarray(A),
    # center=jnp.asarray(CENTER),
    # x1_min=x1_min, x1_max=x1_max,
    # x2_min=x2_min, x2_max=x2_max,
    # n1_internal=n1_internal, n2_internal=n2_internal,

    # tau_bounds=0.02,
    # min_gap_frac=0.05,   # example

    # goal_center=jnp.asarray([5.0, 5.0]),
    # goal_radius=1.0,
    # init_center=jnp.asarray([5.0, 5.0]),
    # init_radius=8.0,

    # tau_bb=0.03,
    # tau_ov=0.02,
    # tau_adv=0.15,
    # gamma=0.95,
    # K=40,

    # lam_det=5.0,
    # lam_cond=1e-3,
    # lam_gap_bounds=5.0,
    # gap_ratio_max=5.0)

    # value_and_grad = jax.jit(jax.value_and_grad(lambda p: soft_adversarial_reach_objective(p, **obj_kwargs)))
    # val, g = value_and_grad(params)
    # print(g)
    # params_opt = adam_optimize(params, soft_adversarial_reach_objective, obj_kwargs,
    #                        steps=5000, lr=1e-5, clip=50.0, print_every=100)
    

    
    # mc = model_check_tessellation(
    #     params_opt,
    #     A=np.asarray(A, dtype=float),
    #     center=np.asarray(CENTER, dtype=float),
    #     x1_min=x1_min, x1_max=x1_max,
    #     x2_min=x2_min, x2_max=x2_max,
    #     n1_internal=n1_internal, n2_internal=n2_internal,
    #     goal_center=(5.0, 5.0),
    #     goal_radius=1.0,
    #     # optional unsafe:
    #     # unsafe_center=(...), unsafe_radius=...,
    #     spec_kind="AF_goal",            # or "AG_not_unsafe" or "AU_avoid_unsafe_reach_goal"
    #     add_out_state=True,
    #     min_gap_frac=0.05,              # should match what you used in training, if any
    # )

    # sat = mc.sat
    # print(f"Satisfaction fraction: {sat.mean():.4f}  ({sat.sum()}/{len(sat)})")

    # # If you want the failing state indices:
    # bad = np.where(~sat)[0]
    # print("Num failing states:", bad.size)
    # print("First 20 failing states:", bad[:20].tolist())

    # # Plot the grids
    # M_np, y1_vals, y2_vals = extract_grid_params(
    # params_opt,
    # n1_internal=n1_internal, n2_internal=n2_internal,
    # x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max,
    # min_gap=0.0,
    # y1_lo=y1_lo, y1_hi=y1_hi, y2_lo=y2_lo, y2_hi=y2_hi)
    # gpt.grid_plotter(M_np, y1_vals, y2_vals, x1_min, x1_max, x2_min, x2_max)
