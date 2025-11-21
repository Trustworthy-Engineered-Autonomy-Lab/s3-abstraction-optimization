# Libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter



# Define the objective function
def objective_function(params):
    L = (params[-1] - params[0]) / (len(params) - 1)
    cost = 0.0
    for k in range(len(params) - 1):
        segment_length = params[k + 1] - params[k]
        cost += 0.5 * (segment_length - L) ** 2
    return cost

# Define gradient of the objective function
def gradient_objective_function(params):
    grad_cost = []
    n = len(params)
    for k in range(n):
        if k == 0 or k == n - 1:
            grad_cost.append(0.0)  # fixed endpoints
        else:
            grad_cost.append(2*params[k] - params[k-1] - params[k+1])
    return grad_cost

# Gradient descent optimization
def gradient_descent(params, learning_rate=0.1, max_iters=5000, tol=1e-8, verbose=True):
    # Make a copy so we don't overwrite the original list
    params = params[:]
    history = []
    params_history = [params[:]]  # Track parameter evolution

    for it in range(max_iters):
        cost = objective_function(params)
        history.append(cost)

        if verbose and it % 10 == 0:
            print(f"Iter {it:5d} | cost = {cost:.6e} | params = {params}")

        # Stopping criterion: cost not changing much
        if it > 0 and abs(history[-2] - history[-1]) < tol:
            if verbose:
                print(f"Converged at iter {it} with cost {cost:.6e}")
            break

        grad = gradient_objective_function(params)

        # Update only interior parameters (keep endpoints fixed)
        for k in range(1, len(params) - 1):
            params[k] -= learning_rate * grad[k]
        
        params_history.append(params[:])

    return params, history, params_history

# Setup the state space
xmin, xmax = -10, 10
params0 = [xmin, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, xmax] # initial parameters





# Run gradient descent
final_params, history, params_history = gradient_descent(params0, learning_rate=0.1, max_iters=5000)
print("\nFinal params:", final_params)
print("Final cost:", objective_function(final_params))

# Create animated GIF
print("\nCreating animation...")

# Sample frames to avoid huge GIF (take every Nth frame)
sample_rate = max(1, len(params_history) // 100)  # Target ~100 frames
sampled_indices = list(range(0, len(params_history), sample_rate))
if sampled_indices[-1] != len(params_history) - 1:
    sampled_indices.append(len(params_history) - 1)  # Always include final frame

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

def animate(frame_idx):
    idx = sampled_indices[frame_idx]
    params = params_history[idx]
    
    # Clear axes
    ax1.clear()
    ax2.clear()
    
    # Top plot: Number line with intervals
    ax1.plot([xmin, xmax], [0, 0], 'k-', linewidth=2, alpha=0.3)
    
    # Plot interval markers
    for k in range(len(params)):
        color = 'red' if k == 0 or k == len(params) - 1 else 'blue'
        ax1.scatter(params[k], 0, s=200, c=color, zorder=3, edgecolors='black', linewidth=2)
    
    # Draw intervals with length labels
    for k in range(len(params) - 1):
        interval_length = params[k + 1] - params[k]
        mid_point = (params[k] + params[k + 1]) / 2
        ax1.plot([params[k], params[k + 1]], [0, 0], 'b-', linewidth=4, alpha=0.6)
        ax1.text(mid_point, -0.15, f'{interval_length:.2f}', ha='center', fontsize=10, color='blue')
    
    ax1.set_xlim(xmin - 1, xmax + 1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_title(f'Iteration {idx} | Cost: {history[idx]:.6f}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_yticks([])
    
    # Bottom plot: Cost history
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
anim.save('interval_optimization.gif', writer=writer)
print("Animation saved as 'interval_optimization.gif'")

plt.close()
