# Libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

CENTER = np.array([7.0, 7.0])

# Define the dynamics
def dynamics(x, alpha=0.1, center=CENTER):
    x = np.asarray(x, dtype=float)
    u = x - center  # shift so center is at 0
    u1, u2 = u
    gradV = np.array([u1 + u1**3,
                      u2 + u2**3])
    return x - alpha * gradV

def jacobian_dynamics(x, alpha=0.1, center=CENTER):
    x = np.asarray(x, dtype=float)
    u = x - center
    u1, u2 = u
    df1_dx1 = 1.0 - alpha * (1.0 + 3.0 * u1**2)
    df2_dx2 = 1.0 - alpha * (1.0 + 3.0 * u2**2)
    return np.array([
        [df1_dx1, 0.0],
        [0.0,     df2_dx2]
    ])


def objective_function(params1, params2):
    cost = 0.0
    for i in range(len(params1) - 1):
        for j in range(len(params2) - 1):
            corners = np.array([
                [params1[i],   params2[j]],
                [params1[i+1], params2[j]],
                [params1[i],   params2[j+1]],
                [params1[i+1], params2[j+1]],
            ])
            transformed = np.array([dynamics(c) for c in corners])   # shape (4, 2)
            avg = transformed.mean(axis=0)                           # shape (2,)
            sq_dists = np.sum((transformed - avg)**2, axis=1)        # shape (4,)
            cost += 0.5 * sq_dists.sum()
    return cost


# Define gradient of the objective function
def gradient_objective_function(params1, params2, alpha=0.1):
    params1 = np.asarray(params1, dtype=float)
    params2 = np.asarray(params2, dtype=float)

    n = len(params1)
    m = len(params2)

    grad_cost_1 = np.zeros(n)  # dJ/da_i
    grad_cost_2 = np.zeros(m)  # dJ/db_j

    for i in range(n - 1):
        for j in range(m - 1):
            # --- 1. Corners of cell (i,j) ---
            v1 = np.array([params1[i],   params2[j]])
            v2 = np.array([params1[i+1], params2[j]])
            v3 = np.array([params1[i],   params2[j+1]])
            v4 = np.array([params1[i+1], params2[j+1]])
            vs = [v1, v2, v3, v4]

            # --- 2. Forward dynamics: y_l = f(v_l) ---
            ys = [dynamics(v, alpha=alpha) for v in vs]   # 4 x (2,)
            ys_arr = np.stack(ys, axis=0)                 # shape (4, 2)

            # --- 3. Mean of transformed corners ---
            avg = ys_arr.mean(axis=0)                     # μ_ij

            # --- 4. dJ/dy_l = y_l - μ ---
            g = ys_arr - avg                              # shape (4, 2)

            # --- 5. dJ/dv_l = J_f(v_l)^T * g_l ---
            u = []
            for l in range(4):
                Jf = jacobian_dynamics(vs[l], alpha=alpha)  # 2x2
                u_l = Jf.T @ g[l]                           # shape (2,)
                u.append(u_l)
            u = np.stack(u, axis=0)  # shape (4, 2)

            # --- 6. Accumulate into grad wrt a's and b's ---
            # corner 0: (a_i,     b_j)
            grad_cost_1[i]   += u[0, 0]   # x1 component
            grad_cost_2[j]   += u[0, 1]   # x2 component

            # corner 1: (a_{i+1}, b_j)
            grad_cost_1[i+1] += u[1, 0]
            grad_cost_2[j]   += u[1, 1]

            # corner 2: (a_i,     b_{j+1})
            grad_cost_1[i]   += u[2, 0]
            grad_cost_2[j+1] += u[2, 1]

            # corner 3: (a_{i+1}, b_{j+1})
            grad_cost_1[i+1] += u[3, 0]
            grad_cost_2[j+1] += u[3, 1]

    # Fix endpoints
    grad_cost_1[0]  = 0.0
    grad_cost_1[-1] = 0.0
    grad_cost_2[0]  = 0.0
    grad_cost_2[-1] = 0.0

    return grad_cost_1, grad_cost_2


# Gradient descent optimization
def gradient_descent(params1, params2, learning_rate=0.1, max_iters=5000, tol=1e-3, verbose=True):
    # Make a copy so we don't overwrite the original list
    params1 = params1[:]
    params2 = params2[:]
    history = []
    params1_history = [params1[:]]  # Track parameter evolution
    params2_history = [params2[:]]

    for it in range(max_iters):
        cost = objective_function(params1, params2)
        
        # Check for NaN - early termination if unstable
        if np.isnan(cost):
            print(f"WARNING: NaN encountered at iteration {it}. Stopping.")
            print("Try reducing the learning rate.")
            break
            
        history.append(cost)

        if verbose and it % 100 == 0:
            print(f"Iter {it:5d} | cost = {cost:.6e}")

        # Stopping criterion: cost not changing much
        grad1, grad2 = gradient_objective_function(params1, params2)
        grad_norm = np.sqrt(np.sum(np.array(grad1)**2) + np.sum(np.array(grad2)**2))
        if grad_norm < tol:
            if verbose:
                print(f"Converged at iter {it} with cost {cost:.6e}")
            break

        grad = gradient_objective_function(params1, params2)

        # Update only interior parameters (keep endpoints fixed)
        for k in range(1, len(params1) - 1):
            params1[k] -= learning_rate * grad[0][k]
        for k in range(1, len(params2) - 1):
            params2[k] -= learning_rate * grad[1][k]
            
        params1_history.append(params1[:])
        params2_history.append(params2[:])

    return params1, params2, history, params1_history, params2_history


x1min, x1max = -10, 10
x2min, x2max = -10, 10
params1 = np.linspace(x1min, x1max, 52).tolist()
params2 = np.linspace(x2min, x2max, 52).tolist()

cost = objective_function(params1, params2)
print("Initial cost:", cost)
grad_cost1, grad_cost2 = gradient_objective_function(params1, params2)
print("Initial gradient (params1):", grad_cost1)
print("Initial gradient (params2):", grad_cost2)

# Run gradient descent
lr = 0.0000015
final_params1, final_params2, history, params1_history, params2_history = gradient_descent(params1, params2, learning_rate=lr, max_iters=50000)
print("\nFinal params1:", final_params1)
print("Final params2:", final_params2)
print("Final cost:", objective_function(final_params1, final_params2))





# Create animated GIF

# Sample frames to avoid huge GIF
sample_rate = max(1, len(params1_history) // 100)  # Target ~100 frames
sampled_indices = list(range(0, len(params1_history), sample_rate))
if sampled_indices[-1] != len(params1_history) - 1:
    sampled_indices.append(len(params1_history) - 1)  # Always include final frame

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

def animate(frame_idx):
    idx = sampled_indices[frame_idx]
    p1 = params1_history[idx]
    p2 = params2_history[idx]
    
    # Clear axes
    ax1.clear()
    ax2.clear()
    
    # Left plot: 2D grid with moving grid lines
    # Draw vertical lines (params1)
    for i, x in enumerate(p1):
        color = 'red' if i == 0 or i == len(p1) - 1 else 'blue'
        linewidth = 3 if i == 0 or i == len(p1) - 1 else 1.5
        ax1.axvline(x, color=color, linewidth=linewidth, alpha=0.7)
    
    # Draw horizontal lines (params2)
    for j, y in enumerate(p2):
        color = 'red' if j == 0 or j == len(p2) - 1 else 'green'
        linewidth = 3 if j == 0 or j == len(p2) - 1 else 1.5
        ax1.axhline(y, color=color, linewidth=linewidth, alpha=0.7)
    
    ax1.set_xlim(x1min - 0.5, x1max + 0.5)
    ax1.set_ylim(x2min - 0.5, x2max + 0.5)
    ax1.set_xlabel('position (p)', fontsize=12)
    ax1.set_ylabel('velocity (v)', fontsize=12)
    ax1.set_title(f'2D Grid Evolution | Iteration {idx} | Cost: {history[idx]:.6f}', 
                  fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Right plot: Cost history
    ax2.plot(history[:idx+1], 'b-', linewidth=2)
    ax2.scatter([idx], [history[idx]], c='red', s=100, zorder=3)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cost', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(history))
    ax2.set_ylim(0, max(history) * 1.1)

# Create animation
anim = FuncAnimation(fig, animate, frames=len(sampled_indices), interval=50, repeat=True)

# Save as GIF
print(f"Saving GIF with {len(sampled_indices)} frames...")
writer = PillowWriter(fps=20)
anim.save('dynamics-optimize.gif', writer=writer)

plt.close()


