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

gradient_descent_memory = memory.GradientDescentMemory(
    K=K, 
    V=V
)

lotka_volterra_memory = memory.LotkaVolterraMemory(
    K=K, 
    V=V
)

lotka_volterra_memory.compute_correlations(Q, V)
lotka_volterra_memory.compute_ecological_params()

fit_steps = 200000
step_size = 1e-2
losses = gradient_descent_memory.fit(Q, V, lr=step_size, n_steps=fit_steps)
lv_loss = lotka_volterra_memory.fit(Q, V, t_max=fit_steps * step_size, dt=step_size, store_losses=True)

# Create figure with GridSpec
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.5)  # Increased spacing between subplots
axes = gs.subplots()

# Plot Sigma_vv
heatmap1 = axes[0, 0].imshow(lotka_volterra_memory.Sigma_vv.detach().numpy(), cmap='viridis', aspect='equal')
divider = make_axes_locatable(axes[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(heatmap1, cax=cax, label='Correlation')
axes[0, 0].set_title('A. Value-Value Correlation Matrix (Sigma_vv)')
axes[0, 0].set_xlabel('Value Dimension')
axes[0, 0].set_ylabel('Value Dimension')

# Plot Sigma_qq
heatmap2 = axes[0, 1].imshow(lotka_volterra_memory.Sigma_qq.detach().numpy(), cmap='viridis', aspect='equal')
divider = make_axes_locatable(axes[0, 1])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(heatmap2, cax=cax, label='Correlation')
axes[0, 1].set_title('B. Query-Query Correlation Matrix (Sigma_qq)')
axes[0, 1].set_xlabel('Query Dimension')
axes[0, 1].set_ylabel('Query Dimension')

# Plot Sigma_qv
heatmap3 = axes[0, 2].imshow(lotka_volterra_memory.Sigma_qv.detach().numpy(), cmap='viridis', aspect='equal')
divider = make_axes_locatable(axes[0, 2])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(heatmap3, cax=cax, label='Correlation')
axes[0, 2].set_title('C. Query-Value Correlation Matrix (Sigma_qv)')
axes[0, 2].set_xlabel('Value Dimension')
axes[0, 2].set_ylabel('Query Dimension')

# Plot growth rates (s)
s_values = lotka_volterra_memory.s.detach().numpy()
axes[1, 0].bar(np.arange(len(s_values)), s_values)
axes[1, 0].set_title('D. Growth Rates (s)')
axes[1, 0].set_xlabel('Token Index')
axes[1, 0].set_ylabel('Growth Rate')
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot interaction coefficients (A)
heatmap4 = axes[1, 1].imshow(lotka_volterra_memory.A.detach().numpy(), cmap='coolwarm', 
                            vmin=-np.abs(lotka_volterra_memory.A.detach().numpy()).max(), 
                            vmax=np.abs(lotka_volterra_memory.A.detach().numpy()).max(),
                            aspect='equal')
divider = make_axes_locatable(axes[1, 1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(heatmap4, cax=cax, label='Interaction Strength')
# Format the colorbar with decimal values
cbar.formatter = plt.FormatStrFormatter('%.2f')
cbar.update_ticks()

axes[1, 1].set_title('E. Interaction Coefficients (A)')
axes[1, 1].set_xlabel('Token Index')
axes[1, 1].set_ylabel('Token Index')

# Plot losses from gradient descent
axes[1, 2].plot(lv_loss.detach().numpy(), label='Lotka-Volterra')
axes[1, 2].plot(losses.detach().numpy(), label='Gradient Descent',  linestyle='--')
axes[1, 2].set_title('F. Loss Comparison')
axes[1, 2].set_xlabel('Optimization Step')
axes[1, 2].set_ylabel('Loss')
axes[1, 2].set_ylim(0, 0.5)
axes[1, 2].grid(True, linestyle='--', alpha=0.7)
# Reduce number of x-ticks to prevent overlapping
axes[1, 2].xaxis.set_major_locator(plt.MaxNLocator(5))
# Place legend inside the plot with padding
axes[1, 2].legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=True, framealpha=1)

# Make all axes the same size but with padding
ax_width = 0.22  # Reduced width to allow for more space between plots
ax_height = 0.32  # Reduced height to allow for more space between plots
for i in range(2):
    for j in range(3):
        pos = axes[i, j].get_position()
        axes[i, j].set_position([pos.x0, pos.y0, ax_width, ax_height])

# Adjust the figure layout to ensure all elements are visible
plt.tight_layout(pad=3.0)  # Add extra padding around the entire figure

# Save the figure
os.makedirs('simulations/figures', exist_ok=True)
plt.savefig('simulations/figures/correlation_analysis.png', dpi=300, bbox_inches='tight')

plt.show()
