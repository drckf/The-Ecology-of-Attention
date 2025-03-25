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

memory = memory.InvadeAndAdjustReplicator(
    d_k=10,
    d_v=10,
    L=L,
    device='cpu'
)

# greedy_memory = memory.GreedyInvasionMemory(
#     d_k=10,
#     d_v=10,
#     L=L,
#     device='cpu'
# )

fit_steps = 100000
step_size = 1e-3

invade_costs = []
for i in range(L):
    print(i)
    invade_cost = memory.update(Q[i], K[i], V[i], t_max=fit_steps * step_size, dt=step_size)
    # greedy_cost = greedy_memory.update(Q[i], V[i])
    invade_costs.append(invade_cost.item())

plt.plot(invade_costs)
plt.show()