# Libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter




# Define the objective function
def objective_function(params1, params2):
    x1min, x1max = params1[0], params1[-1]
    x2min, x2max = params2[0], params2[-1]
    L = (x1max - x1min) * (x2max - x2min) / ((len(params1) - 1) * (len(params2) - 1))
    cost = 0.0
    for i in range (len(params1) - 1):
        for j in range(len(params2) - 1):
            cell_area = (params1[i + 1] - params1[i]) * (params2[j + 1] - params2[j])
            cost += 0.5 * (cell_area - L) ** 2
    return cost


# Define gradient of the objective function
def gradient_objective_function(params1, params2):
    grad_cost_1 = [0.0] * len(params1)  # gradients for a's
    grad_cost_2 = [0.0] * len(params2)  # gradients for b's

    n = len(params1)  # n = M + 2
    m = len(params2)  # m = K + 2

    # Segment lengths in x and y
    deltas_x = [params1[i+1] - params1[i] for i in range(n - 1)]  # i = 0..M
    deltas_y = [params2[j+1] - params2[j] for j in range(m - 1)]  # j = 0..K

    # Target cell area L (endpoints assumed fixed)
    width = params1[-1] - params1[0]
    height = params2[-1] - params2[0]
    num_cells = (n - 1) * (m - 1)  # (M+1)*(K+1)
    L = (width * height) / num_cells

    # dJ/d(Δx_i) and dJ/d(Δy_j)
    Gx = [0.0] * (n - 1)  # gradient wrt deltas_x[i]
    Gy = [0.0] * (m - 1)  # gradient wrt deltas_y[j]

    # Loop over all cells (i, j)
    for i in range(n - 1):
        for j in range(m - 1):
            area_ij = deltas_x[i] * deltas_y[j]
            r = area_ij - L  # residual

            # dJ/d(Δx_i) accumulates over j
            Gx[i] += r * deltas_y[j]

            # dJ/d(Δy_j) accumulates over i
            Gy[j] += r * deltas_x[i]

    # Now backpropagate from Δx to a's:
    for i in range(1, n - 1):  # interior a's only
        grad_cost_1[i] = Gx[i - 1] - Gx[i]

    # Similarly from Δy to b's:
    for j in range(1, m - 1):  # interior b's only
        grad_cost_2[j] = Gy[j - 1] - Gy[j]

    # endpoints stay 0.0 (fixed boundaries)
    return grad_cost_1, grad_cost_2

# Gradient descent optimization
def gradient_descent(params1, params2, learning_rate=0.001, max_iters=5000, tol=1e-2, verbose=True):
    # Make a copy so we don't overwrite the original list
    params1 = params1[:]
    params2 = params2[:]
    history = []
    params1_history = [params1[:]]  # Track parameter evolution
    params2_history = [params2[:]]

    for it in range(max_iters):
        cost = objective_function(params1, params2)
        history.append(cost)

        if verbose and it % 100 == 0:
            print(f"Iter {it:5d} | cost = {cost:.6e} | params1 = {params1} | params2 = {params2}")

        # Stopping criterion: cost not changing much
        if it > 0 and abs(history[-2] - history[-1]) < tol:
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
params1 = [x1min, -9.5, -9, -8.5, -8, -7.5, 7, 8.5, 9, 9.5, x1max] # initial parameters for x1
params2 = [x2min, -9.5, -9, -8.5, -8, -7.5, 7, 8.5, 9, 9.5, x2max] # initial parameters for x2

# Run gradient descent
final_params1, final_params2, history, params1_history, params2_history = gradient_descent(params1, params2, learning_rate=0.001, max_iters=5000)
print("\nFinal params1:", final_params1)
print("Final params2:", final_params2)
print("Final cost:", objective_function(final_params1, final_params2))


# Create animated GIF
print("\nCreating animation...")

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
    
    # Optionally: shade cells by their area to show variance
    target_area = (p1[-1] - p1[0]) * (p2[-1] - p2[0]) / ((len(p1) - 1) * (len(p2) - 1))
    
    for i in range(len(p1) - 1):
        for j in range(len(p2) - 1):
            x_left, x_right = p1[i], p1[i + 1]
            y_bottom, y_top = p2[j], p2[j + 1]
            cell_area = (x_right - x_left) * (y_top - y_bottom)
            
            # Color cells by deviation from target area
            deviation = abs(cell_area - target_area) / target_area
            color_intensity = min(1.0, deviation * 5)  # Scale for visibility
            
            # Rectangle with color showing deviation
            rect = plt.Rectangle((x_left, y_bottom), x_right - x_left, y_top - y_bottom,
                                facecolor=plt.cm.RdYlGn_r(color_intensity), 
                                edgecolor='none', alpha=0.3)
            ax1.add_patch(rect)
            
            # Optionally add area text in center of cell (for small grids)
            if len(p1) <= 6 and len(p2) <= 6:
                cx = (x_left + x_right) / 2
                cy = (y_bottom + y_top) / 2
                ax1.text(cx, cy, f'{cell_area:.2f}', ha='center', va='center', 
                        fontsize=8, color='black', weight='bold')
    
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
anim.save('2d_grid_optimization.gif', writer=writer)
print("Animation saved as '2d_grid_optimization.gif'")

plt.close()


