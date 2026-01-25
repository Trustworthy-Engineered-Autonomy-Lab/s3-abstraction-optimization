# Libraries
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# Load data
with open('synthetic_training_history.pkl', 'rb') as f:
    training_history = pkl.load(f)

cost_history = training_history['cost_history']
sr_history = training_history['sr_history']
tnp_history = training_history['tnp_history']
fnr_history = training_history['fnr_history']

# These histories are recorded every N gradient steps in `mountaincar.py`.
record_every = 1
iters_cost = np.arange(len(cost_history)) * record_every
iters_rate = np.arange(len(cost_history)) * record_every

fig, ax = plt.subplots(figsize=(7, 5))
ax2 = ax.twinx()

# Left axis: rates (0..1)
l1, = ax.plot(iters_rate, sr_history, color='blue', label='SR', lw=2)
l2, = ax.plot(iters_rate, tnp_history, color='red', label='TNP', lw=2)
l3, = ax.plot(iters_rate, fnr_history, color='green', label='FNR', lw=2)
ax.set_ylim(-0.02, 1.02)
# ax.set_ylabel('Rate')

# Right axis: cost
l4, = ax2.plot(iters_cost, cost_history, label='Cost', lw=2.5, color='black', alpha=0.85)
# ax2.set_ylabel('Cost')

ax.set_xlabel('Training iterations')
# ax.set_title('Training history')
ax.grid(True, alpha=0.25)

lines = [l1, l2, l3, l4]
labels = [ln.get_label() for ln in lines]
# ax.legend(lines, labels, loc='best')

plt.tight_layout()
plt.show()

