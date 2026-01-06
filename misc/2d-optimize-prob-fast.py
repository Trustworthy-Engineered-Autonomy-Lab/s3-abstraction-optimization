# Libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm

# Define the dynamics
def dynamics(x, x_star=np.array([5.0, 5.0])):
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])  # contraction + mild rotation
    x = np.asarray(x, dtype=float)
    return x_star + A @ (x - x_star)

# Compute Markovian transition probabilities
def compute_probabilities(params1, params2, sigma1=1.0, sigma2=1.0):
    params1 = np.asarray(params1, dtype=float)
    params2 = np.asarray(params2, dtype=float)
    M = len(params1) - 1
    N = len(params2) - 1
    mids1 = 0.5 * (params1[:-1] + params1[1:])
    mids2 = 0.5 * (params2[:-1] + params2[1:])
    mid_grid1, mid_grid2 = np.meshgrid(mids1, mids2, indexing='ij')
    mid_stack = np.stack([mid_grid1, mid_grid2], axis=-1)
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    x_star = np.array([-5.0, -5.0])
    mu = x_star + (mid_stack - x_star) @ A.T
    mu1 = mu[..., 0]
    mu2 = mu[..., 1]
    a_lo = params1[:-1][None, None, :, None]
    a_hi = params1[1:][None, None, :, None]
    b_lo = params2[:-1][None, None, None, :]
    b_hi = params2[1:][None, None, None, :]
    mu1_exp = mu1[:, :, None, None]
    mu2_exp = mu2[:, :, None, None]
    Am = norm.cdf(a_hi, loc=mu1_exp, scale=sigma1) - norm.cdf(a_lo, loc=mu1_exp, scale=sigma1)
    Bn = norm.cdf(b_hi, loc=mu2_exp, scale=sigma2) - norm.cdf(b_lo, loc=mu2_exp, scale=sigma2)
    probabilities = Am * Bn
    return probabilities

def build_Q_r(params1, params2, sigma1=0.5, sigma2=0.5):
    probabilities = compute_probabilities(params1, params2, sigma1=sigma1, sigma2=sigma2)
    M = len(params1) - 1
    N = len(params2) - 1
    num_states = M * N
    T_part = probabilities.reshape(num_states, num_states)
    row_sums = T_part.sum(axis=1)
    Q = T_part
    r = 1.0 - row_sums
    return Q, r

def fail_probs_within_N(Q, r, N):
    f = np.zeros_like(r)
    for _ in range(N):
        f = r + Q @ f
    return f

def fail_within_N_steps(T, s0, failure_index=None, N=50):
    n = T.shape[0]
    if failure_index is None:
        failure_index = n - 1
    non_fail = [i for i in range(n) if i != failure_index]
    Q = T[np.ix_(non_fail, non_fail)]
    r = T[non_fail, failure_index]
    f = np.zeros(len(non_fail))
    for _ in range(N):
        f = r + Q @ f
    x = np.zeros(n)
    x[non_fail] = f
    x[failure_index] = 1.0
    return float(x[s0])


def failure_reach_probability(P, s0, failure_index=None):

    n = P.shape[0]
    if failure_index is None:
        failure_index = n - 1  # default: last state
    assert P.shape == (n, n)
    if not np.allclose(P[failure_index], np.eye(n)[failure_index]):
        print("Warning: failure state does not look absorbing.")
    non_fail = [i for i in range(n) if i != failure_index]
    m = len(non_fail)
    Q = P[np.ix_(non_fail, non_fail)]
    r = P[non_fail, failure_index]
    I = np.eye(m)
    A = I - Q
    x_nonfail = np.linalg.solve(A, r)
    x = np.zeros(n)
    x[non_fail] = x_nonfail
    x[failure_index] = 1.0
    return float(x[s0])

def objective_function(params1, params2):
    Q, r = build_Q_r(params1, params2, sigma1=0.5, sigma2=0.5)
    fN = fail_probs_within_N(Q, r, 100)
    return float(fN.mean())

def gradient_objective_function(params1, params2):
    import numpy as np
    from scipy.stats import norm
    sigma1 = 0.5
    sigma2 = 0.5
    horizon = 100
    A_dyn = np.array([[0.8, -0.3],
                      [0.3,  0.8]])
    params1 = np.asarray(params1, dtype=float)
    params2 = np.asarray(params2, dtype=float)
    M = len(params1) - 1
    N = len(params2) - 1
    num_states = M * N
    m = num_states
    probabilities = compute_probabilities(params1, params2,
                                          sigma1=sigma1, sigma2=sigma2)
    T_part = probabilities.reshape(num_states, num_states)
    row_sums = T_part.sum(axis=1)
    Q = T_part.copy()
    r = 1.0 - row_sums
    f_hist = [np.zeros(m)]
    for _ in range(horizon):
        f_new = r + Q @ f_hist[-1]
        f_hist.append(f_new)
    u_hist = []
    u0 = np.full(m, 1.0 / m)
    u_hist.append(u0)
    for t in range(1, horizon):
        u_hist.append(u_hist[-1] @ Q)
    dJ_dQ = np.zeros_like(Q)
    dJ_dr = np.zeros_like(r)
    for t in range(horizon):
        dJ_dQ += np.outer(u_hist[t], f_hist[horizon - 1 - t])
        dJ_dr += u_hist[t]
    W_2D = dJ_dQ - dJ_dr[:, None]
    weights = W_2D.reshape(M, N, M, N)
    grad_cost_1 = np.zeros_like(params1)
    grad_cost_2 = np.zeros_like(params2)
    for i in range(M):
        a_lo = params1[i]
        a_hi = params1[i + 1]
        mid1 = 0.5 * (a_lo + a_hi)
        for j in range(N):
            b_lo = params2[j]
            b_hi = params2[j + 1]
            mid2 = 0.5 * (b_lo + b_hi)
            mid_x = np.array([mid1, mid2])
            mu = dynamics(mid_x)
            mu1, mu2 = mu
            for m_idx in range(M):
                a_m = params1[m_idx]
                a_m1 = params1[m_idx + 1]
                z_lo = (a_m - mu1) / sigma1
                z_hi = (a_m1 - mu1) / sigma1
                Phi_hi1 = norm.cdf(z_hi)
                Phi_lo1 = norm.cdf(z_lo)
                phi_hi1 = norm.pdf(z_hi)
                phi_lo1 = norm.pdf(z_lo)
                A_m = Phi_hi1 - Phi_lo1
                dA_dmu1 = (phi_lo1 - phi_hi1) / sigma1
                for n_idx in range(N):
                    w = weights[i, j, m_idx, n_idx]
                    if w == 0.0:
                        continue
                    b_n = params2[n_idx]
                    b_n1 = params2[n_idx + 1]
                    v_lo = (b_n - mu2) / sigma2
                    v_hi = (b_n1 - mu2) / sigma2
                    Phi_hi2 = norm.cdf(v_hi)
                    Phi_lo2 = norm.cdf(v_lo)
                    phi_hi2 = norm.pdf(v_hi)
                    phi_lo2 = norm.pdf(v_lo)
                    B_n = Phi_hi2 - Phi_lo2
                    dB_dmu2 = (phi_lo2 - phi_hi2) / sigma2
                    for k in {i, i + 1, m_idx, m_idx + 1}:
                        if k < 0 or k >= M + 1:
                            continue
                        dmid1_da = 0.5 if (k == i or k == i + 1) else 0.0
                        dmu1_da = A_dyn[0, 0] * dmid1_da
                        dmu2_da = A_dyn[1, 0] * dmid1_da
                        dA_da = dA_dmu1 * dmu1_da
                        if k == m_idx:
                            dA_da += (-phi_lo1) / sigma1
                        if k == m_idx + 1:
                            dA_da += (phi_hi1) / sigma1
                        dB_da = dB_dmu2 * dmu2_da
                        dp_da = dA_da * B_n + A_m * dB_da
                        grad_cost_1[k] += w * dp_da
                    for l in {j, j + 1, n_idx, n_idx + 1}:
                        if l < 0 or l >= N + 1:
                            continue
                        dmid2_db = 0.5 if (l == j or l == j + 1) else 0.0
                        dmu1_db = A_dyn[0, 1] * dmid2_db
                        dmu2_db = A_dyn[1, 1] * dmid2_db
                        dA_db = dA_dmu1 * dmu1_db
                        dB_db = dB_dmu2 * dmu2_db
                        if l == n_idx:
                            dB_db += (-phi_lo2) / sigma2
                        if l == n_idx + 1:
                            dB_db += (phi_hi2) / sigma2
                        dp_db = dA_db * B_n + A_m * dB_db
                        grad_cost_2[l] += w * dp_db
    return grad_cost_1, grad_cost_2



# Gradient descent optimization
def gradient_descent(params1, params2, learning_rate=0.1, max_iters=5000, tol=1e-3, cushion = [1e-2, 1e-3], verbose=True):
    params1 = params1[:]
    params2 = params2[:]
    history = []
    params1_history = [params1[:]]
    params2_history = [params2[:]]
    for it in range(max_iters):
        cost = objective_function(params1, params2)
        if np.isnan(cost):
            print(f"WARNING: NaN encountered at iteration {it}. Stopping.")
            print("Try reducing the learning rate.")
            break
        history.append(cost)
        if verbose and it % 1 == 0:
            print(f"Iter {it:5d} | cost = {cost:.4e}")
        grad1, grad2 = gradient_objective_function(params1, params2)
        grad_norm = np.sqrt(np.sum(np.array(grad1)**2) + np.sum(np.array(grad2)**2))
        if grad_norm < tol:
            if verbose:
                print(f"Converged at iter {it} with cost {cost:.6e}")
            break
        for k in range(1, len(params1) - 1):
            params1[k] -= learning_rate * grad1[k]
            params1[k] = np.clip(params1[k], params1[k-1] + cushion[0], params1[k+1] - cushion[0])
        for k in range(1, len(params2) - 1):
            params2[k] -= learning_rate * grad2[k]
            params2[k] = np.clip(params2[k], params2[k-1] + cushion[1], params2[k+1] - cushion[1])
        params1_history.append(params1[:])
        params2_history.append(params2[:])
    return params1, params2, history, params1_history, params2_history

x1min, x1max = -10, 10
x2min, x2max = -10, 10
# params1 = np.linspace(x1min, x1max, 11).tolist()
# params2 = np.linspace(x2min, x2max, 11).tolist()
num_interior_points = 8
interior1 = np.random.uniform(x1min, x1max, num_interior_points)
interior2 = np.random.uniform(x2min, x2max, num_interior_points)
params1 = sorted([x1min] + interior1.tolist() + [x1max])
params2 = sorted([x2min] + interior2.tolist() + [x2max])

cost = objective_function(params1, params2)
print("Initial cost:", cost)
grad_cost1, grad_cost2 = gradient_objective_function(params1, params2)
print("Initial gradient (params1):", grad_cost1)
print("Initial gradient (params2):", grad_cost2)

lr = 0.5
final_params1, final_params2, history, params1_history, params2_history = gradient_descent(params1, params2, learning_rate=lr, max_iters=100, cushion=[0.5, 0.5])
print("\nFinal params1:", final_params1)
print("Final params2:", final_params2)
print("Final cost:", objective_function(final_params1, final_params2))

sample_rate = max(1, len(params1_history) // 100)
sampled_indices = list(range(0, len(params1_history), sample_rate))
if len(sampled_indices) == 0 or sampled_indices[-1] != len(params1_history) - 1:
    sampled_indices.append(len(params1_history) - 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

cost_max = max(history) if len(history) > 0 else 1.0

def animate(frame_idx):
    idx = sampled_indices[frame_idx]
    p1 = params1_history[idx]
    p2 = params2_history[idx]
    ax1.clear()
    ax2.clear()
    Q, r = build_Q_r(p1, p2, sigma1=0.5, sigma2=0.5)
    fN = fail_probs_within_N(Q, r, 100)
    M_curr = len(p1) - 1
    N_curr = len(p2) - 1
    failure_grid = fN.reshape(M_curr, N_curr)
    for i in range(M_curr):
        for j in range(N_curr):
            x_left, x_right = p1[i], p1[i + 1]
            y_bottom, y_top = p2[j], p2[j + 1]
            fail_prob = failure_grid[i, j]
            rect = plt.Rectangle((x_left, y_bottom), x_right - x_left, y_top - y_bottom,
                                facecolor=plt.cm.hot_r(fail_prob),
                                edgecolor='black', linewidth=0.5, alpha=0.7)
            ax1.add_patch(rect)
    for i, x in enumerate(p1):
        color = 'black' if i == 0 or i == len(p1) - 1 else 'gray'
        linewidth = 2 if i == 0 or i == len(p1) - 1 else 0.8
        ax1.axvline(x, color=color, linewidth=linewidth, alpha=0.8)
    for j, y in enumerate(p2):
        color = 'black' if j == 0 or j == len(p2) - 1 else 'gray'
        linewidth = 2 if j == 0 or j == len(p2) - 1 else 0.8
        ax1.axhline(y, color=color, linewidth=linewidth, alpha=0.8)
    ax1.set_xlim(x1min - 0.5, x1max + 0.5)
    ax1.set_ylim(x2min - 0.5, x2max + 0.5)
    ax1.set_xlabel('x1', fontsize=12)
    ax1.set_ylabel('x2', fontsize=12)
    ax1.set_title(f'Failure Risk (100 steps) | Iter {idx} | Cost: {history[idx]:.6f}',
                  fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax2.plot(history[:idx+1], 'b-', linewidth=2, label='cost')
    ax2.scatter([idx], [history[idx]], c='blue', s=60, zorder=3)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Uniform Failure Probability', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(history))
    ax2.set_ylim(0, cost_max * 1.1)

anim = FuncAnimation(fig, animate, frames=len(sampled_indices), interval=50, repeat=True)
print(f"Saving GIF with {len(sampled_indices)} frames...")
writer = PillowWriter(fps=20)
anim.save('dynamics-optimize.gif', writer=writer)
plt.close()

