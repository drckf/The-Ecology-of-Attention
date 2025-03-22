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
    K=K, 
    V=V
)

correlation_memory = memory.CorrelationBasedMemory(
    K=K, 
    V=V
)

correlation_memory.compute_correlations(Q, V)
correlation_memory.compute_ecological_params()

losses = gradient_descent_memory.fit(Q, V, lr=1e-6, n_steps=1000)
corr_loss = correlation_memory.fit(Q, V, t_max=1000 * 1e-6, dt=1e-6, store_losses=True)

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

# Plot losses from gradient descent
axes[1, 2].plot(losses.detach().numpy(), label='Gradient Descent')
axes[1, 2].plot(corr_loss.detach().numpy(), label='Correlation-Based')
axes[1, 2].set_title('Loss Comparison')
axes[1, 2].set_xlabel('Optimization Step')
axes[1, 2].set_ylabel('Loss')
axes[1, 2].grid(True, linestyle='--', alpha=0.7)
axes[1, 2].legend()

# Adjust spacing to prevent legend overlap
plt.tight_layout(pad=3.0)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Save the figure
os.makedirs('simulations/figures', exist_ok=True)
plt.savefig('simulations/figures/correlation_analysis.png', dpi=300, bbox_inches='tight')

plt.show()
