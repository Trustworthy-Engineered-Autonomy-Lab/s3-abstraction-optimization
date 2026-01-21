# Libraries
import jax
import jax.numpy as jnp
import numpy as np
from grid_plot_tools import get_yspace_bounds


def dynamics(state):
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    x_star = np.array([5.0, 5.0])
    state = np.asarray(state, dtype=float)
    return (state - x_star) @ A.T + x_star

def trans_matrix(theta, a1, a2, h):
    """M = H @ S @ R where s1=exp(a1), s2=exp(a2)."""
    theta = jnp.asarray(theta)
    a1 = jnp.asarray(a1)
    a2 = jnp.asarray(a2)
    h = jnp.asarray(h)

    s1 = jnp.exp(a1)
    s2 = jnp.exp(a2)

    c, s = jnp.cos(theta), jnp.sin(theta)
    dtype = jnp.result_type(theta, a1, a2, h)

    R = jnp.array([[c, -s],
                   [s,  c]], dtype=dtype)
    S = jnp.array([[s1, 0.0],
                   [0.0, s2]], dtype=dtype)
    H = jnp.array([[1.0, h],
                   [0.0, 1.0]], dtype=dtype)
    return H @ S @ R


def goal_score(x_verts, x_star, alpha=1.0):
    """Compute goal score for a set of vertices."""
    dists = jnp.linalg.norm(x_verts - x_star, axis=1)
    avg_dist = jnp.mean(dists)
    # print("Avg dist to goal:", avg_dist)
    return jnp.exp(-avg_dist * alpha)


def fail_score(x_verts, c_star, alpha=1.0):
    """Compute goal score for a set of vertices."""
    dists = jnp.linalg.norm(x_verts - c_star, axis=1)
    avg_dist = jnp.mean(dists)
    # print("Avg dist to center:", avg_dist)
    return 1-jnp.exp(-avg_dist * alpha)



def objective(x_domain, x_star, radius, y1_params, y2_params, M):

    x1_min, x1_max, x2_min, x2_max = x_domain
    nstates_1 = len(y1_params) - 1
    nstates_2 = len(y2_params) - 1
    M = np.asarray(M, dtype=float)

    x_star=jnp.asarray(x_star)
    c_star=jnp.asarray([(x1_min + x1_max)/2, (x2_min + x2_max)/2])

    # Loop through each abstract state
    total_score = 0.0
    for i in range(nstates_1):
        y1_lo, y1_hi = y1_params[i], y1_params[i+1]
        for j in range(nstates_2):
            y2_lo, y2_hi = y2_params[j], y2_params[j+1]
            y_corners = np.array([
                    [y1_lo, y2_lo],
                    [y1_lo, y2_hi],
                    [y1_hi, y2_hi],
                    [y1_hi, y2_lo]])

            # Compute cell score of successor
            x_verts = (M @ y_corners.T).T
            x_verts = dynamics(x_verts)
            score = cell_score(
                x_verts,
                x_star=x_star,
                c_star=c_star,
                alpha_goal=0.1,
                alpha_fail=0.1,
                goal_weight=0.5,
                fail_weight=0.5
            )
            total_score += score
    return total_score/(nstates_1 * nstates_2)
            
def softmin_weighted(v, w, tau, eps=1e-12):
    z = (w + eps) * jnp.exp(-v / tau)
    return -tau * jnp.log(jnp.sum(z) / (jnp.sum(w) + eps))

def cell_score(x_verts, x_star, c_star, alpha_goal=1.0, alpha_fail=1.0, goal_weight=0.5, fail_weight=0.5):
    gs = goal_score(x_verts, x_star, alpha=alpha_goal)
    fs = fail_score(x_verts, c_star, alpha=alpha_fail)
    total_score = goal_weight*gs + fail_weight*fs
    return total_score

# def soft_value_iter_score(x_verts, x_star, c_star, alpha_goal=1.0, alpha_fail=1.0, gamma=1.0):
#     gs = goal_score(x_verts, x_star, alpha=alpha_goal)
#     fs = fail_score(x_verts, c_star, alpha=alpha_fail)
#     v_next = gs + (1-gs)*(1-fs)*gamma*softmin_weighted


if __name__ == "__main__":


    x_verts = jnp.array([[0.0, 1.0],
                         [0.0, 0.0],
                         [1.0, 0.0],
                         [1.0, 1.0]])
    x_star = jnp.array([5.0, 5.0])
    c_star = jnp.array([0.0, 0.0])
    alpha_goal = 0.1
    alpha_fail = 0.1


    for step in range(10):

        total_score = cell_score(x_verts, x_star, c_star, alpha_goal=alpha_goal, alpha_fail=alpha_fail, goal_weight=0.5, fail_weight=0.5)

        print("Cell center:", jnp.mean(x_verts, axis=0))
        # print("Goal score:", gs)
        # print("Fail score:", fs)
        print("Total score:", total_score)

        x_verts = dynamics(x_verts)


    