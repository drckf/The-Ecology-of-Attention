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

invasion_memory = memory.InvadeAndAdjustReplicator(
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
step_size = 1e-3

invade_costs = []
greedy_costs = []
linear_costs = []
for i in range(L):
    print(i)
    invade_cost = invasion_memory.update(Q[i], K[i], V[i], t_max=fit_steps * step_size, dt=step_size)
    greedy_cost = greedy_memory.update(Q[i], K[i], V[i])
    linear_cost = linear_memory.update(Q[i], K[i], V[i])
    invade_costs.append(invade_cost.item())
    greedy_costs.append(greedy_cost.item())
    linear_costs.append(linear_cost.item())

# Create a publication-quality figure
plt.figure(figsize=(10, 6))
plt.plot(invade_costs, linewidth=2, label='Invade and Adjust')
plt.plot(greedy_costs, linewidth=2, label='Greedy Invasion')
plt.plot(linear_costs, linewidth=2, label='Linear Attention')

# Add labels and title
plt.xlabel('Token Position', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.title('Cost Comparison of Memory Update Methods', fontsize=16)

# Add grid and legend
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Tight layout for better spacing
plt.tight_layout()

# Create figures directory if it doesn't exist
os.makedirs('simulations/   figures', exist_ok=True)

# Save the figure in high resolution
plt.savefig('simulations/figures/online_memory_cost_comparison.png', dpi=300, bbox_inches='tight')

# Display the figure
plt.show()