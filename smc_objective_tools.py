# Libraries
import jax
import jax.numpy as jnp
import numpy as np
from grid_plot_tools import get_yspace_bounds


def _dynamics_jax(state, A, x_star):
    state = jnp.asarray(state)
    A = jnp.asarray(A)
    x_star = jnp.asarray(x_star)
    return jnp.matmul(state - x_star, A.T) + x_star


def _goal_score_jax(x_verts, x_star, alpha=1.0):
    x_verts = jnp.asarray(x_verts)
    x_star = jnp.asarray(x_star)
    dists = jnp.linalg.norm(x_verts - x_star, axis=-1)
    avg_dist = jnp.mean(dists, axis=-1)
    return jnp.exp(-avg_dist * alpha)


def _fail_score_jax(x_verts, c_star, alpha=1.0):
    x_verts = jnp.asarray(x_verts)
    c_star = jnp.asarray(c_star)
    dists = jnp.linalg.norm(x_verts - c_star, axis=-1)
    avg_dist = jnp.mean(dists, axis=-1)
    return 1.0 - jnp.exp(-avg_dist * alpha)


def objective_jax(
    x_domain,
    x_star,
    y1_params,
    y2_params,
    theta,
    s1,
    s2,
    h,
    *,
    A=jnp.array([[0.8, -0.3], [0.3, 0.8]]),
    alpha_goal=0.1,
    alpha_fail=0.1,
    goal_weight=0.5,
    fail_weight=0.5,
):

    x1_min, x1_max, x2_min, x2_max = [jnp.asarray(v) for v in x_domain]
    x_star = jnp.asarray(x_star)
    c_star = jnp.stack([(x1_min + x1_max) / 2.0, (x2_min + x2_max) / 2.0])

    y1_params = jnp.asarray(y1_params)
    y2_params = jnp.asarray(y2_params)

    # Build M = H @ S @ R using the provided (theta, s1, s2, h)
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.array([[c, -s], [s, c]], dtype=y1_params.dtype)
    S = jnp.array([[s1, 0.0], [0.0, s2]], dtype=y1_params.dtype)
    H = jnp.array([[1.0, h], [0.0, 1.0]], dtype=y1_params.dtype)
    M = H @ S @ R

    # Cell edge arrays
    y1_lo = y1_params[:-1]
    y1_hi = y1_params[1:]
    y2_lo = y2_params[:-1]
    y2_hi = y2_params[1:]

    n1 = y1_lo.shape[0]
    n2 = y2_lo.shape[0]

    # Broadcast to a (n1, n2) grid
    y1_lo_g = jnp.broadcast_to(y1_lo[:, None], (n1, n2))
    y1_hi_g = jnp.broadcast_to(y1_hi[:, None], (n1, n2))
    y2_lo_g = jnp.broadcast_to(y2_lo[None, :], (n1, n2))
    y2_hi_g = jnp.broadcast_to(y2_hi[None, :], (n1, n2))

    # Corners per cell: (n1, n2, 4, 2)
    p00 = jnp.stack([y1_lo_g, y2_lo_g], axis=-1)
    p01 = jnp.stack([y1_lo_g, y2_hi_g], axis=-1)
    p11 = jnp.stack([y1_hi_g, y2_hi_g], axis=-1)
    p10 = jnp.stack([y1_hi_g, y2_lo_g], axis=-1)
    y_corners = jnp.stack([p00, p01, p11, p10], axis=-2)

    # Map Y->X and step dynamics
    x_verts = jnp.matmul(y_corners, M.T)          # (n1, n2, 4, 2)
    x_verts = _dynamics_jax(x_verts, A=A, x_star=x_star)

    gs = _goal_score_jax(x_verts, x_star=x_star, alpha=alpha_goal)
    fs = _fail_score_jax(x_verts, c_star=c_star, alpha=alpha_fail)
    score = goal_weight * gs + fail_weight * fs  # (n1, n2)

    return jnp.mean(score)






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


    