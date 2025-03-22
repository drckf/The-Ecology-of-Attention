import os
import torch
import numpy as np
import ecoattention.memory as memory
from generate import SyntheticDataGenerator
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
torch.manual_seed(42)
L = 100

generator = SyntheticDataGenerator(
    N=20, 
    D_v=10, 
    D_k=10, 
    rho=0.5, 
    sim=0.5, 
    device='cpu'
)

X, Q, K, V = generator.generate_QKV(L)

gradient_descent_memory = memory.GradientDescentMemory(
    d_k=10, 
    d_v=10, 
    L=L
)

correlation_memory = memory.CorrelationBasedMemory(
    d_k=10, 
    d_v=10, 
    L=L
)

correlation_memory.compute_correlations(Q, K, V)
correlation_memory.compute_ecological_params(K, V)

losses = gradient_descent_memory.fit(Q, K, V, lr=1e-9, n_steps=5)

print(losses)

# Create a figure with 5 subplots: 3 for correlation matrices, 1 for growth rates, 1 for interaction coefficients
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot Sigma_vv
heatmap1 = axes[0, 0].imshow(correlation_memory.Sigma_vv.detach().numpy(), cmap='viridis')
fig.colorbar(heatmap1, ax=axes[0, 0], label='Correlation')
axes[0, 0].set_title('Value-Value Correlation Matrix (Sigma_vv)')
axes[0, 0].set_xlabel('Value Dimension')
axes[0, 0].set_ylabel('Value Dimension')

# Plot Sigma_qq
heatmap2 = axes[0, 1].imshow(correlation_memory.Sigma_qq.detach().numpy(), cmap='viridis')
fig.colorbar(heatmap2, ax=axes[0, 1], label='Correlation')
axes[0, 1].set_title('Query-Query Correlation Matrix (Sigma_qq)')
axes[0, 1].set_xlabel('Query Dimension')
axes[0, 1].set_ylabel('Query Dimension')

# Plot Sigma_qv
heatmap3 = axes[0, 2].imshow(correlation_memory.Sigma_qv.detach().numpy(), cmap='viridis')
fig.colorbar(heatmap3, ax=axes[0, 2], label='Correlation')
axes[0, 2].set_title('Query-Value Correlation Matrix (Sigma_qv)')
axes[0, 2].set_xlabel('Value Dimension')
axes[0, 2].set_ylabel('Query Dimension')

# Plot growth rates (s)
s_values = correlation_memory.s.detach().numpy()
axes[1, 0].bar(np.arange(len(s_values)), s_values)
axes[1, 0].set_title('Growth Rates (s)')
axes[1, 0].set_xlabel('Token Index')
axes[1, 0].set_ylabel('Growth Rate')
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot interaction coefficients (A)
heatmap4 = axes[1, 1].imshow(correlation_memory.A.detach().numpy(), cmap='coolwarm', 
                            vmin=-np.abs(correlation_memory.A.detach().numpy()).max(), 
                            vmax=np.abs(correlation_memory.A.detach().numpy()).max())
fig.colorbar(heatmap4, ax=axes[1, 1], label='Interaction Strength')
axes[1, 1].set_title('Interaction Coefficients (A)')
axes[1, 1].set_xlabel('Token Index')
axes[1, 1].set_ylabel('Token Index')

# Hide the unused subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
