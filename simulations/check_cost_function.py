import torch
import ecoattention.memory as memory
from generate import SyntheticDataGenerator

# Set the random seed for reproducibility
torch.manual_seed(42)

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
    d_k=10, 
    d_v=10, 
    L=L
)

correlation_memory = memory.CorrelationBasedMemory(
    d_k=10, 
    d_v=10, 
    L=L
)

w = torch.randn(L, device=correlation_memory.device)
gradient_descent_memory.w.data = w
gradient_descent_memory.set_memory(K, V)
grad_cost = gradient_descent_memory.compute_cost(Q, V)

correlation_memory.w = w
correlation_memory.set_memory(K, V)
correlation_memory.compute_correlations(Q, K, V)
corr_cost = correlation_memory.compute_cost()

print(corr_cost)
print(grad_cost)

