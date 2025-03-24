import os
import torch
import numpy as np
import ecoattention.memory as memory
from generate import SyntheticDataGenerator
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
torch.manual_seed(42)
iterations = 1000

results = np.zeros((iterations, 2))

for i in range(iterations):

    generator = SyntheticDataGenerator(
        N=20, 
        D_v=10, 
        D_k=10, 
        rho=0.5, 
        sim=0.5, 
        device='cpu'
    )

    L = 100
    X, Q, K, V = generator.generate_QKV(L)

    gradient_descent_memory = memory.GradientDescentMemory(
        K=K, 
        V=V
    )

    correlation_memory = memory.CorrelationBasedMemory(
        K=K, 
        V=V
    )

    w = torch.randn(L, device=correlation_memory.device)
    gradient_descent_memory.w.data = w
    grad_cost = gradient_descent_memory.compute_cost(Q, V)

    correlation_memory.w = w
    correlation_memory.compute_correlations(Q, V)
    corr_cost = correlation_memory.compute_cost()

    results[i, 0] = np.log(corr_cost.cpu().detach().numpy())
    results[i, 1] = np.log(grad_cost.cpu().detach().numpy())

# Create a scatter plot of correlation cost vs gradient descent cost
plt.figure(figsize=(8, 8))
plt.scatter(results[:, 0], results[:, 1], color='blue', alpha=0.7, s=80, label='Cost comparison')
plt.plot([0, 5], [0, 5], 'k--', alpha=0.5, label='y = x')  # Diagonal dashed line

# Set equal aspect ratio to properly show y = x relationship
plt.axis('equal')

# Add labels and title
plt.xlabel('Log Correlation-based Cost', fontsize=12)
plt.ylabel('Log Gradient Descent Cost', fontsize=12)
plt.title('Comparison of Cost Functions', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Adjust limits to ensure all points are visible with typical values between 10 and 15
plt.xlim(0, 5)
plt.ylim(0, 5)

plt.tight_layout()
os.makedirs('simulations/figures', exist_ok=True)
plt.savefig('simulations/figures/cost_comparison.png')
plt.show()
