# Libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

CENTER = np.array([5.0, 5.0])

# Define the dynamics
def dynamics(x, center=CENTER):
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])  # contraction + mild rotation
    x = np.asarray(x, dtype=float)
    return center + A @ (x - center)

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


# Average number of successor states
# Helper: map a point (x,y) to cell indices (i,j); returns None if outside grid.
def _point_to_cell(x, y, params1, params2):
    if x < params1[0] or x > params1[-1] or y < params2[0] or y > params2[-1]:
        return None
    i = np.searchsorted(params1, x, side='right') - 1
    j = np.searchsorted(params2, y, side='right') - 1
    # Ensure indices refer to a valid cell (exclude last line index)
    if i < 0 or i >= len(params1) - 1 or j < 0 or j >= len(params2) - 1:
        return None
    return (i, j)

def meta_objective_function(params1, params2, include_self=True, ignore_out_of_bounds=True):

    params1 = np.asarray(params1, dtype=float)
    params2 = np.asarray(params2, dtype=float)
    n_cells = (len(params1) - 1) * (len(params2) - 1)
    if n_cells == 0:
        return 0.0

    total_successors = 0

    for i in range(len(params1) - 1):
        for j in range(len(params2) - 1):
            corners = np.array([
                [params1[i],   params2[j]],
                [params1[i+1], params2[j]],
                [params1[i],   params2[j+1]],
                [params1[i+1], params2[j+1]],
            ])
            transformed = np.array([dynamics(c) for c in corners])  # (4,2)

            successor_cells = set()
            if include_self:
                successor_cells.add((i, j))

            for (tx, ty) in transformed:
                cell = _point_to_cell(tx, ty, params1, params2)
                if cell is None:
                    if ignore_out_of_bounds:
                        continue
                    else:
                        continue  # placeholder for potential future handling
                successor_cells.add(cell)

            total_successors += len(successor_cells)

    return total_successors / n_cells


# Differentiable replacement of the desired objective
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


# # Define gradient of the objective function
# def gradient_objective_function(params1, params2, alpha=0.1):
#     params1 = np.asarray(params1, dtype=float)
#     params2 = np.asarray(params2, dtype=float)

#     n = len(params1)
#     m = len(params2)

#     grad_cost_1 = np.zeros(n)  # dJ/da_i
#     grad_cost_2 = np.zeros(m)  # dJ/db_j

#     for i in range(n - 1):
#         for j in range(m - 1):
#             # --- 1. Corners of cell (i,j) ---
#             v1 = np.array([params1[i],   params2[j]])
#             v2 = np.array([params1[i+1], params2[j]])
#             v3 = np.array([params1[i],   params2[j+1]])
#             v4 = np.array([params1[i+1], params2[j+1]])
#             vs = [v1, v2, v3, v4]

#             # --- 2. Forward dynamics: y_l = f(v_l) ---
#             ys = [dynamics(v, alpha=alpha) for v in vs]   # 4 x (2,)
#             ys_arr = np.stack(ys, axis=0)                 # shape (4, 2)

#             # --- 3. Mean of transformed corners ---
#             avg = ys_arr.mean(axis=0)                     # μ_ij

#             # --- 4. dJ/dy_l = y_l - μ ---
#             g = ys_arr - avg                              # shape (4, 2)

#             # --- 5. dJ/dv_l = J_f(v_l)^T * g_l ---
#             u = []
#             for l in range(4):
#                 Jf = jacobian_dynamics(vs[l], alpha=alpha)  # 2x2
#                 u_l = Jf.T @ g[l]                           # shape (2,)
#                 u.append(u_l)
#             u = np.stack(u, axis=0)  # shape (4, 2)

#             # --- 6. Accumulate into grad wrt a's and b's ---
#             # corner 0: (a_i,     b_j)
#             grad_cost_1[i]   += u[0, 0]   # x1 component
#             grad_cost_2[j]   += u[0, 1]   # x2 component

#             # corner 1: (a_{i+1}, b_j)
#             grad_cost_1[i+1] += u[1, 0]
#             grad_cost_2[j]   += u[1, 1]

#             # corner 2: (a_i,     b_{j+1})
#             grad_cost_1[i]   += u[2, 0]
#             grad_cost_2[j+1] += u[2, 1]

#             # corner 3: (a_{i+1}, b_{j+1})
#             grad_cost_1[i+1] += u[3, 0]
#             grad_cost_2[j+1] += u[3, 1]

#     # Fix endpoints
#     grad_cost_1[0]  = 0.0
#     grad_cost_1[-1] = 0.0
#     grad_cost_2[0]  = 0.0
#     grad_cost_2[-1] = 0.0

#     return grad_cost_1, grad_cost_2

def gradient_objective_function(params1, params2):
    """
    params1: list/array of a's (x1 grid positions)
    params2: list/array of b's (x2 grid positions)

    Returns:
        grad_cost_1: gradient dJ/da_i, same length as params1
        grad_cost_2: gradient dJ/db_j, same length as params2
    """
    params1 = np.asarray(params1, dtype=float)
    params2 = np.asarray(params2, dtype=float)

    n = len(params1)  # number of a's
    m = len(params2)  # number of b's

    grad_cost_1 = np.zeros(n)
    grad_cost_2 = np.zeros(m)

    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    AT = A.T

    # Loop over all cells (i,j)
    for i in range(n - 1):
        for j in range(m - 1):
            # Corners of cell (i,j)
            corners = np.array([
                [params1[i],   params2[j]],     # v1
                [params1[i+1], params2[j]],     # v2
                [params1[i],   params2[j+1]],   # v3
                [params1[i+1], params2[j+1]],   # v4
            ])  # shape (4, 2)

            # Transform corners through dynamics: y_l = A v_l
            transformed = corners @ AT  # shape (4, 2); AT so rows are v^T * A^T = (A v)^T

            # Mean of transformed corners
            avg = transformed.mean(axis=0)  # μ_ij, shape (2,)

            # Gradient wrt y_l: g_l = y_l - μ
            g = transformed - avg  # shape (4, 2)

            # Gradient wrt v_l: u_l = A^T g_l
            # (row-wise: each u_l = AT @ g_l)
            u = g @ AT  # shape (4, 2)

            # Distribute contributions to a's and b's:
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

    # Optionally fix endpoints: zero gradient so you don't move boundaries
    grad_cost_1[0]  = 0.0
    grad_cost_1[-1] = 0.0
    grad_cost_2[0]  = 0.0
    grad_cost_2[-1] = 0.0

    return grad_cost_1.tolist(), grad_cost_2.tolist()


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
        meta_cost = meta_objective_function(params1, params2)
        
        # Check for NaN - early termination if unstable
        if np.isnan(cost):
            print(f"WARNING: NaN encountered at iteration {it}. Stopping.")
            print("Try reducing the learning rate.")
            break
            
        history.append(cost)

        if verbose and it % 100 == 0:
            print(f"Iter {it:5d} | cost = {cost:.6e} | meta cost = {meta_cost:.3e}")

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
params1 = np.linspace(x1min, x1max, 22).tolist()
params2 = np.linspace(x2min, x2max, 22).tolist()

cost = objective_function(params1, params2)
print("Initial cost:", cost)
grad_cost1, grad_cost2 = gradient_objective_function(params1, params2)
meta_cost = meta_objective_function(params1, params2)
print("Initial gradient (params1):", grad_cost1)
print("Initial gradient (params2):", grad_cost2)
print("Initial meta cost:", meta_cost)

# Run gradient descent
lr = 0.01
final_params1, final_params2, history, params1_history, params2_history = gradient_descent(params1, params2, learning_rate=lr, tol=0, max_iters=10000)
print("\nFinal params1:", final_params1)
print("Final params2:", final_params2)
print("Final cost:", objective_function(final_params1, final_params2))

# # Compute meta cost history corresponding to each stored params history
# meta_history = [meta_objective_function(p1, p2) for p1, p2 in zip(params1_history, params2_history)]
# print("Initial meta cost:", meta_history[0])
# print("Final   meta cost:", meta_history[-1])

# # Create animated GIF

# # Sample frames to avoid huge GIF
# sample_rate = max(1, len(params1_history) // 100)  # Target ~100 frames
# sampled_indices = list(range(0, len(params1_history), sample_rate))
# if sampled_indices[-1] != len(params1_history) - 1:
#     sampled_indices.append(len(params1_history) - 1)  # Always include final frame

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
# ax2b = ax2.twinx()  # twin axis for meta cost

# # Precompute y-lims for consistent scaling across frames
# cost_max = max(history) if len(history) > 0 else 1.0
# meta_max = max(meta_history) if len(meta_history) > 0 else 1.0

# def animate(frame_idx):
#     idx = sampled_indices[frame_idx]
#     p1 = params1_history[idx]
#     p2 = params2_history[idx]
    
#     # Clear axes
#     ax1.clear()
#     ax2.clear()
#     ax2b.clear()
    
#     # Left plot: 2D grid with moving grid lines
#     # Draw vertical lines (params1)
#     for i, x in enumerate(p1):
#         color = 'red' if i == 0 or i == len(p1) - 1 else 'blue'
#         linewidth = 3 if i == 0 or i == len(p1) - 1 else 1.5
#         ax1.axvline(x, color=color, linewidth=linewidth, alpha=0.7)
    
#     # Draw horizontal lines (params2)
#     for j, y in enumerate(p2):
#         color = 'red' if j == 0 or j == len(p2) - 1 else 'green'
#         linewidth = 3 if j == 0 or j == len(p2) - 1 else 1.5
#         ax1.axhline(y, color=color, linewidth=linewidth, alpha=0.7)
    
#     ax1.set_xlim(x1min - 0.5, x1max + 0.5)
#     ax1.set_ylim(x2min - 0.5, x2max + 0.5)
#     ax1.set_xlabel('position (p)', fontsize=12)
#     ax1.set_ylabel('velocity (v)', fontsize=12)
#     ax1.set_title(f'2D Grid Evolution | Iteration {idx} | Cost: {history[idx]:.6f} | Meta: {meta_history[idx]:.6f}', 
#                   fontsize=13, fontweight='bold')
#     ax1.set_aspect('equal')
#     ax1.grid(True, alpha=0.2, linestyle='--')
    
#     # Right plot: Cost and Meta cost history
#     ax2.plot(history[:idx+1], 'b-', linewidth=2, label='cost')
#     ax2.scatter([idx], [history[idx]], c='blue', s=60, zorder=3)
#     ax2.set_xlabel('Iteration', fontsize=12)
#     # ax2.set_ylabel('Cost', fontsize=12, color='blue')
#     ax2.tick_params(axis='y', labelcolor='blue')
#     ax2.grid(True, alpha=0.3)
#     ax2.set_xlim(0, len(history))
#     ax2.set_ylim(0, cost_max * 1.1)

#     ax2b.plot(meta_history[:idx+1], 'orange', linewidth=2, label='meta cost')
#     ax2b.scatter([idx], [meta_history[idx]], c='orange', s=60, zorder=3)
#     # ax2b.set_ylabel('Meta cost', fontsize=12, color='orange')
#     ax2b.tick_params(axis='y', labelcolor='orange')
#     ax2b.set_ylim(0, meta_max * 1.1)

#     # Optional combined legend
#     # Create a single legend by combining handles
#     handles1, labels1 = ax2.get_legend_handles_labels()
#     handles2, labels2 = ax2b.get_legend_handles_labels()
#     if handles1 or handles2:
#         ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

# # Create animation
# anim = FuncAnimation(fig, animate, frames=len(sampled_indices), interval=50, repeat=True)

# # Save as GIF
# print(f"Saving GIF with {len(sampled_indices)} frames...")
# writer = PillowWriter(fps=20)
# anim.save('dynamics-optimize.gif', writer=writer)

# plt.close()


