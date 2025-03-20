import torch
import ecoattention.memory as memory
from generate import SyntheticDataGenerator

# Set the random seed for reproducibility
torch.manual_seed(42)

generator = SyntheticDataGenerator(
    N=100, 
    D_v=10, 
    D_k=10, 
    rho=0.5, 
    sim=0.5, 
    device='cpu'
)

L = 100
X, Q, K, V = generator.generate_QKV(L)