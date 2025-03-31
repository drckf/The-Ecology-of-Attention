import os
import torch
import numpy as np
import ecoattention.memory as memory
from generate import SyntheticDataGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set the random seed for reproducibility
torch.manual_seed(42)
L = 100

generator = SyntheticDataGenerator(
    N=20, 
    D_v=10, 
    D_k=10, 
    rho=0.5, 
    sim=0.9, 
    device='cpu'
)

X, Q, K, V = generator.generate_QKV(L)

invasion_memory = memory.InvadeAndAdjustLotkaVolterra(
    d_k=10,
    d_v=10,
    L=L,
    device='cpu'
)

greedy_memory = memory.GreedyInvasionMemory(
    d_k=10,
    d_v=10,
    device='cpu'
)

linear_memory = memory.LinearAttentionMemory(
    d_k=10,
    d_v=10,
    device='cpu'
)

fit_steps = 100000
step_size = 1e-2

invade_costs = []
greedy_costs = []
linear_costs = []
invade_count = []
greedy_count = []
linear_count = []
for i in range(L):
    print(i)
    invade_cost, invade_bool = invasion_memory.update(Q[i], K[i], V[i], t_max=fit_steps * step_size, dt=step_size)
    greedy_cost, greedy_bool = greedy_memory.update(Q[i], K[i], V[i])
    linear_cost, linear_bool = linear_memory.update(Q[i], K[i], V[i])
    invade_costs.append(invade_cost.item())
    greedy_costs.append(greedy_cost.item())
    linear_costs.append(linear_cost.item())
    invade_count.append(invade_bool)
    greedy_count.append(greedy_bool)
    linear_count.append(linear_bool)

# Create a publication-quality figure with two panels
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Top panel: Costs
ax1.plot(invade_costs, linewidth=2, label='Invade and Adjust')
ax1.plot(greedy_costs, linewidth=2, label='Greedy Invasion')
ax1.plot(linear_costs, linewidth=2, label='Linear Attention')

# Add labels and title for top panel
ax1.set_ylabel('Cost', fontsize=14)
ax1.set_title('Comparison of Memory Update Methods', fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)
ax1.tick_params(labelsize=12)

# Bottom panel: Cumulative memories
ax2.plot(np.cumsum(invade_count), linewidth=2, label='Invade and Adjust')
ax2.plot(np.cumsum(greedy_count), linewidth=2, label='Greedy Invasion')
ax2.plot(np.cumsum(linear_count), linewidth=2, label='Linear Attention')

# Add labels for bottom panel
ax2.set_xlabel('Token Position', fontsize=14)
ax2.set_ylabel('Cumulative Memories', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)
ax2.tick_params(labelsize=12)

# Tight layout for better spacing
plt.tight_layout()

# Create figures directory if it doesn't exist
os.makedirs('simulations/figures', exist_ok=True)

# Save the figure in high resolution
plt.savefig('simulations/figures/online_memory_cost_comparison.png', dpi=300, bbox_inches='tight')

# Display the figure
plt.show()