import torch
import torch.nn as nn
from . import integrate



def squared_retrieval_error(
         q: torch.Tensor, 
         v: torch.Tensor,
         associative_memory: nn.Module
    ):
    """
    Compute the retrieval error of the associative memory.
    """
    return (associative_memory.forward(q) - v).pow(2).sum(dim=1).mean()


class GradientDescentMemory(nn.Module):
    """
    A class for updating and storing a memory of data.
    """
    def __init__(self, d_k: int, d_v: int, L: int):
        """
        Initialize the memory.

        Args:
            d_k: The dimension of the keys.
            d_v: The dimension of the values.
            L: The number of data points in the memory.
        """
        super().__init__()
        self.L = L
        self.J = torch.zeros(d_v, d_k)
        self.w = nn.Parameter(torch.zeros(L))

    def set_memory(self, k: torch.Tensor, v: torch.Tensor):
        """
        Set the memory matrix J using the provided keys and values.

        This is a vectorized implementation that computes J = sum_i w_i * v_i * k_i^T.

        Args:
            k: A tensor of shape (L, d_k) containing the keys
            v: A tensor of shape (L, d_v) containing the values

        Returns:
            The computed memory matrix J of shape (d_v, d_k)
        """
        # v: (L, d_v), k: (L, d_k), w: (L,)
        # Reshape w to (L, 1, 1) for broadcasting
        w_expanded = self.w.view(-1, 1, 1)
        # v: (L, d_v, 1), k: (L, 1, d_k)
        self.J = (w_expanded * v.unsqueeze(-1) * k.unsqueeze(1)).sum(dim=0)
        return self.J

    def forward(self, q: torch.Tensor):
        """
        Forward pass of the memory.

        Args:
            q: A tensor of shape (L, d_k) containing the queries.

        Returns:
            A tensor of shape (L, d_v) containing the values.
        """
        return q @ self.J.T
    
    def fit(
            self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor,
            lr: float = 1e-3,
            n_steps: int = 100
        ) -> torch.Tensor:
        """
        Fit the memory to the data.

        Args:
            q: A tensor of shape (L, d_k) containing the queries.
            k: A tensor of shape (L, d_k) containing the keys.
            v: A tensor of shape (L, d_v) containing the values.
            lr: Learning rate for optimization
            n_steps: Number of optimization steps

        Returns:
            torch.Tensor: Vector of loss values at each optimization step
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = torch.zeros(n_steps)
        for i in range(n_steps):
            optimizer.zero_grad()
            self.set_memory(k, v)
            loss = squared_retrieval_error(q, v, self)
            losses[i] = loss.item()
            loss.backward()
            optimizer.step()
        return losses


class CorrelationBasedMemory(nn.Module):
    """
    A class for updating and storing a memory of data using correlation-based optimization.
    """
    def __init__(self, d_k: int, d_v: int, L: int, device: torch.device = None):
        """
        Initialize the memory.  

        Args:
            d_k: The dimension of the keys
            d_v: The dimension of the values
            L: The number of data points in the memory
            device: Device to store tensors on (default: cpu)
        """
        super().__init__()
        self.device = device or torch.device('cpu')
        self.d_k = d_k
        self.d_v = d_v
        self.L = L
        
        # Initialize tensors on specified device
        self.J = torch.zeros(d_v, d_k, device=self.device)
        self.w = torch.zeros(L, device=self.device)
        # Correlation matrices
        self.Sigma_vv = torch.zeros(d_v, d_v, device=self.device)
        self.Sigma_qv = torch.zeros(d_k, d_v, device=self.device)
        self.Sigma_qq = torch.zeros(d_k, d_k, device=self.device)
        # Growth rates and interaction coefficients
        self.s = torch.zeros(L, device=self.device)
        self.A = torch.zeros(L, L, device=self.device)

    def compute_correlations(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Compute correlation matrices Sigma_vv, Sigma_qv, and Sigma_qq.

        Args:
            q: queries of shape (L, d_k)
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)

        Raises:
            ValueError: If input tensors have incorrect shapes
        """
        if not (q.shape[0] == k.shape[0] == v.shape[0] == self.L):
            raise ValueError(f"Expected batch size {self.L}, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        if not (q.shape[1] == k.shape[1] == self.d_k):
            raise ValueError(f"Expected key dim {self.d_k}, got {q.shape[1]}, {k.shape[1]}")
        if v.shape[1] != self.d_v:
            raise ValueError(f"Expected value dim {self.d_v}, got {v.shape[1]}")

        self.Sigma_vv = (v.T @ v) / self.L
        self.Sigma_qv = (q.T @ v) / self.L
        self.Sigma_qq = (q.T @ q) / self.L

    def compute_ecological_params(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Compute growth rates s_l and interaction coefficients A_ll'.
        Vectorized implementation for better efficiency.

        Args:
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)
        """
        # s_l = k_l^T Sigma_qv v_l
        self.s = torch.sum(k @ self.Sigma_qv * v, dim=1)
        
        # A_ll' = v_l^T v_l' k_l'^T Sigma_qq k_l
        v_outer = v @ v.T  # (L, L)
        k_sigma_k = k @ self.Sigma_qq @ k.T  # (L, L)
        self.A = v_outer * k_sigma_k

    def compute_optimal_weights(self) -> None:
        """
        Compute optimal weights as w = A^(-1) s.
        """
        try:
            self.w = torch.linalg.solve(self.A, self.s)
        except RuntimeError:
            # Fallback if A is not invertible
            self.w = torch.linalg.lstsq(self.A, self.s.unsqueeze(-1)).solution.squeeze()

    def compute_cost(self) -> torch.Tensor:
        """
        Compute the cost function value:
        C(w) = 1/2 Tr(Sigma_vv) - Tr(J Sigma_qv) + 1/2 Tr(J Sigma_qq J^T)

        Returns:
            torch.Tensor: Current value of the cost function
        """
        term1 = 0.5 * torch.trace(self.Sigma_vv)
        term2 = -torch.trace(self.J @ self.Sigma_qv)
        term3 = 0.5 * torch.trace(self.J @ self.Sigma_qq @ self.J.T)
        return term1 + term2 + term3

    def set_memory(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Set the memory matrix J using the provided keys and values.

        Args:
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)

        Returns:
            The computed memory matrix J of shape (d_v, d_k)
        """
        w_expanded = self.w.view(-1, 1, 1)
        self.J = (w_expanded * v.unsqueeze(-1) * k.unsqueeze(1)).sum(dim=0)
        return self.J

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the memory.

        Args:
            q: queries of shape (L, d_k)

        Returns:
            Retrieved values of shape (L, d_v)
        """
        return q @ self.J.T

    def fit(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Fit the memory by computing optimal weights directly.

        Args:
            q: queries of shape (L, d_k)
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)

        Returns:
            torch.Tensor: Final cost value
        """
        self.compute_correlations(q, k, v)
        self.compute_ecological_params(k, v)
        self.compute_optimal_weights()
        self.set_memory(k, v)
        return self.compute_cost()

    def fit_with_dynamics(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            t_max: float = 10.0,
            dt: float = 0.01
        ) -> torch.Tensor:
        """
        Fit using dynamical systems integration.

        Args:
            q: queries of shape (L, d_k)
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)
            dynamics: One of ['linear', 'lotka_volterra', 'replicator']
            t_max: Maximum integration time
            dt: Integration time step

        Returns:
            torch.Tensor: Final cost value
        """
        self.compute_correlations(q, k, v)
        self.compute_ecological_params(k, v)
        
        self.w = integrate.integrate_linear(
            w_0=torch.zeros_like(self.w),
            s=self.s,
            A=self.A,
            t_max=t_max,
            dt=dt
        )
        
        self.set_memory(k, v)
        return self.compute_cost()
    
    
class LotkaVolterraMemory(nn.Module):
    """
    A class for updating and storing a memory of data using correlation-based optimization.
    """
    def __init__(self, d_k: int, d_v: int, L: int, device: torch.device = None):
        """
        Initialize the memory.  

        Args:
            d_k: The dimension of the keys
            d_v: The dimension of the values
            L: The number of data points in the memory
            device: Device to store tensors on (default: cpu)
        """
        super().__init__()
        self.device = device or torch.device('cpu')
        self.d_k = d_k
        self.d_v = d_v
        self.L = L
        
        # Initialize tensors on specified device
        self.J = torch.zeros(d_v, d_k, device=self.device)
        self.w = torch.zeros(L, device=self.device)
        # Correlation matrices
        self.Sigma_vv = torch.zeros(d_v, d_v, device=self.device)
        self.Sigma_qv = torch.zeros(d_k, d_v, device=self.device)
        self.Sigma_qq = torch.zeros(d_k, d_k, device=self.device)
        # Growth rates and interaction coefficients
        self.s = torch.zeros(L, device=self.device)
        self.A = torch.zeros(L, L, device=self.device)

    def compute_correlations(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Compute correlation matrices Sigma_vv, Sigma_qv, and Sigma_qq.

        Args:
            q: queries of shape (L, d_k)
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)

        Raises:
            ValueError: If input tensors have incorrect shapes
        """
        if not (q.shape[0] == k.shape[0] == v.shape[0] == self.L):
            raise ValueError(f"Expected batch size {self.L}, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        if not (q.shape[1] == k.shape[1] == self.d_k):
            raise ValueError(f"Expected key dim {self.d_k}, got {q.shape[1]}, {k.shape[1]}")
        if v.shape[1] != self.d_v:
            raise ValueError(f"Expected value dim {self.d_v}, got {v.shape[1]}")

        self.Sigma_vv = (v.T @ v) / self.L
        self.Sigma_qv = (q.T @ v) / self.L
        self.Sigma_qq = (q.T @ q) / self.L

    def compute_ecological_params(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Compute growth rates s_l and interaction coefficients A_ll'.
        Vectorized implementation for better efficiency.

        Args:
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)
        """
        # s_l = k_l^T Sigma_qv v_l
        self.s = torch.sum(k @ self.Sigma_qv * v, dim=1)
        
        # A_ll' = v_l^T v_l' k_l'^T Sigma_qq k_l
        v_outer = v @ v.T  # (L, L)
        k_sigma_k = k @ self.Sigma_qq @ k.T  # (L, L)
        self.A = v_outer * k_sigma_k

    def compute_cost(self) -> torch.Tensor:
        """
        Compute the cost function value:
        C(w) = 1/2 Tr(Sigma_vv) - Tr(J Sigma_qv) + 1/2 Tr(J Sigma_qq J^T)

        Returns:
            torch.Tensor: Current value of the cost function
        """
        term1 = 0.5 * torch.trace(self.Sigma_vv)
        term2 = -torch.trace(self.J @ self.Sigma_qv)
        term3 = 0.5 * torch.trace(self.J @ self.Sigma_qq @ self.J.T)
        return term1 + term2 + term3

    def set_memory(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Set the memory matrix J using the provided keys and values.

        Args:
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)

        Returns:
            The computed memory matrix J of shape (d_v, d_k)
        """
        w_expanded = self.w.view(-1, 1, 1)
        self.J = (w_expanded * v.unsqueeze(-1) * k.unsqueeze(1)).sum(dim=0)
        return self.J

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the memory.

        Args:
            q: queries of shape (L, d_k)

        Returns:
            Retrieved values of shape (L, d_v)
        """
        return q @ self.J.T

    def fit(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dynamics: str = 'linear',
            t_max: float = 10.0,
            dt: float = 0.01
        ) -> torch.Tensor:
        """
        Fit using dynamical systems integration.

        Args:
            q: queries of shape (L, d_k)
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)
            dynamics: One of ['linear', 'lotka_volterra', 'replicator']
            t_max: Maximum integration time
            dt: Integration time step

        Returns:
            torch.Tensor: Final cost value
        """
        self.compute_correlations(q, k, v)
        self.compute_ecological_params(k, v)        
        
        self.w = integrate.integrate_lotka_volterra(
            w_0=torch.zeros_like(self.w),
            s=self.s,
            A=self.A,
            t_max=t_max,
            dt=dt
        )
        
        self.set_memory(k, v)
        return self.compute_cost()
    
    
class ReplicatorMemory(nn.Module):
    """
    A class for updating and storing a memory of data using correlation-based optimization.
    """
    def __init__(self, d_k: int, d_v: int, L: int, device: torch.device = None):
        """
        Initialize the memory.  

        Args:
            d_k: The dimension of the keys
            d_v: The dimension of the values
            L: The number of data points in the memory
            device: Device to store tensors on (default: cpu)
        """
        super().__init__()
        self.device = device or torch.device('cpu')
        self.d_k = d_k
        self.d_v = d_v
        self.L = L
        
        # Initialize tensors on specified device
        self.J = torch.zeros(d_v, d_k, device=self.device)
        self.w = torch.ones(L, device=self.device) / L
        # Correlation matrices
        self.Sigma_vv = torch.zeros(d_v, d_v, device=self.device)
        self.Sigma_qv = torch.zeros(d_k, d_v, device=self.device)
        self.Sigma_qq = torch.zeros(d_k, d_k, device=self.device)
        # Growth rates and interaction coefficients
        self.s = torch.zeros(L, device=self.device)
        self.A = torch.zeros(L, L, device=self.device)

    def compute_correlations(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Compute correlation matrices Sigma_vv, Sigma_qv, and Sigma_qq.

        Args:
            q: queries of shape (L, d_k)
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)

        Raises:
            ValueError: If input tensors have incorrect shapes
        """
        if not (q.shape[0] == k.shape[0] == v.shape[0] == self.L):
            raise ValueError(f"Expected batch size {self.L}, got {q.shape[0]}, {k.shape[0]}, {v.shape[0]}")
        if not (q.shape[1] == k.shape[1] == self.d_k):
            raise ValueError(f"Expected key dim {self.d_k}, got {q.shape[1]}, {k.shape[1]}")
        if v.shape[1] != self.d_v:
            raise ValueError(f"Expected value dim {self.d_v}, got {v.shape[1]}")

        self.Sigma_vv = (v.T @ v) / self.L
        self.Sigma_qv = (q.T @ v) / self.L
        self.Sigma_qq = (q.T @ q) / self.L

    def compute_ecological_params(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Compute growth rates s_l and interaction coefficients A_ll'.
        Vectorized implementation for better efficiency.

        Args:
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)
        """
        # s_l = k_l^T Sigma_qv v_l
        self.s = torch.sum(k @ self.Sigma_qv * v, dim=1)
        
        # A_ll' = v_l^T v_l' k_l'^T Sigma_qq k_l
        v_outer = v @ v.T  # (L, L)
        k_sigma_k = k @ self.Sigma_qq @ k.T  # (L, L)
        self.A = v_outer * k_sigma_k

    def compute_cost(self) -> torch.Tensor:
        """
        Compute the cost function value:
        C(w) = 1/2 Tr(Sigma_vv) - Tr(J Sigma_qv) + 1/2 Tr(J Sigma_qq J^T)

        Returns:
            torch.Tensor: Current value of the cost function
        """
        term1 = 0.5 * torch.trace(self.Sigma_vv)
        term2 = -torch.trace(self.J @ self.Sigma_qv)
        term3 = 0.5 * torch.trace(self.J @ self.Sigma_qq @ self.J.T)
        return term1 + term2 + term3

    def set_memory(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Set the memory matrix J using the provided keys and values.

        Args:
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)

        Returns:
            The computed memory matrix J of shape (d_v, d_k)
        """
        w_expanded = self.w.view(-1, 1, 1)
        self.J = (w_expanded * v.unsqueeze(-1) * k.unsqueeze(1)).sum(dim=0)
        return self.J

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the memory.

        Args:
            q: queries of shape (L, d_k)

        Returns:
            Retrieved values of shape (L, d_v)
        """
        return q @ self.J.T

    def fit(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dynamics: str = 'linear',
            t_max: float = 10.0,
            dt: float = 0.01
        ) -> torch.Tensor:
        """
        Fit using dynamical systems integration.

        Args:
            q: queries of shape (L, d_k)
            k: keys of shape (L, d_k)
            v: values of shape (L, d_v)
            dynamics: One of ['linear', 'lotka_volterra', 'replicator']
            t_max: Maximum integration time
            dt: Integration time step

        Returns:
            torch.Tensor: Final cost value
        """
        self.compute_correlations(q, k, v)
        self.compute_ecological_params(k, v)        
        
        self.w = integrate.integrate_replicator_equation(
            w_0=torch.zeros_like(self.w),
            s=self.s,
            A=self.A,
            t_max=t_max,
            dt=dt
        )
        
        self.set_memory(k, v)
        return self.compute_cost()
    
    
