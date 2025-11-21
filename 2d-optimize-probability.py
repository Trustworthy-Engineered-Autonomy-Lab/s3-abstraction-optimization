# Libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm

# Define the dynamics
def dynamics(x):
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    return A @ x

# Compute Markovian transition probabilities
def compute_probabilities(params1, params2, sigma1=1.0, sigma2=1.0):
    M = len(params1) - 1
    N = len(params2) - 1
    probabilities = np.empty((M, N, M, N), dtype=float)

    # Outer loop loops over current cells (i,j)
    for i in range(M):
        for j in range(N):
            a_lo = params1[i]
            a_hi = params1[i+1]
            b_lo = params2[j]
            b_hi = params2[j+1]
            mid_x = [0.5 * (a_lo + a_hi), 0.5 * (b_lo + b_hi)]
            mu = dynamics(np.array(mid_x))

            # Inner loop loops over target cells (k,l)
            for m in range(M):
                for n in range(N):
                    Am = norm.cdf(params1[m+1], loc=mu[0], scale=sigma1) - norm.cdf(params1[m], loc=mu[0], scale=sigma1)
                    Bn = norm.cdf(params2[n+1], loc=mu[1], scale=sigma2) - norm.cdf(params2[n], loc=mu[1], scale=sigma2)
                    probabilities[i, j, m, n] = Am * Bn
    return probabilities

def failure_reach_probabilities(P):
    num_states_with_failure = P.shape[0]
    num_states = num_states_with_failure - 1  # all but last

    Q = P[:num_states, :num_states]
    r = P[:num_states, -1]

    I = np.eye(num_states)
    v = np.linalg.solve(I - Q, r)  # length S

    v_full = np.zeros(num_states_with_failure)
    v_full[:num_states] = v
    v_full[-1] = 1.0  # starting in failure -> prob 1 of "reaching failure"
    return v_full

x1min, x1max = -10, 10
x2min, x2max = -10, 10
params1 = np.linspace(x1min, x1max, 11).tolist()
params2 = np.linspace(x2min, x2max, 11).tolist()

# Compute transition probabilities; unroll into transition matrix
probabilities = compute_probabilities(params1, params2, sigma1=0.5, sigma2=0.5)

# Convert 4D array to 2D transition matrix
M = len(params1) - 1
N = len(params2) - 1
num_states = M * N
transition_matrix_partial = probabilities.reshape(num_states, num_states)

# Add absorbing/failure state
row_sums = transition_matrix_partial.sum(axis=1)
missing_mass = 1.0 - row_sums
transition_matrix = np.zeros((num_states + 1, num_states + 1))
transition_matrix[:num_states, :num_states] = transition_matrix_partial
transition_matrix[:num_states, -1] = missing_mass
transition_matrix[-1, -1] = 1.0
print(transition_matrix)

# # Quantiative reachability analysis
# v_full = failure_reach_probabilities(transition_matrix)
# M = len(params1) - 1
# N = len(params2) - 1
# grid_probs = v_full[:-1].reshape(M, N)
# print("Probability of eventual failure from center cell:", grid_probs[M//2, N//2])
