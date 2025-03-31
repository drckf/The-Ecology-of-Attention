import torch
import torch.nn as nn
from . import integrate
from .expo import ExponentiatedSGD


################################################################################
# Gradient Descent Memory
################################################################################

def squared_retrieval_error(
         q: torch.Tensor, 
         v: torch.Tensor,
         J: torch.Tensor
    ):
    """
    Compute the retrieval error of the associative memory with memory matrix J.
    """
    return 0.5 * (q @ J.T - v).pow(2).sum(dim=1).mean()


class GradientDescentMemory(nn.Module):
    """
    A class for updating and storing a memory of data.
    """
    def __init__(self, K: torch.Tensor, V: torch.Tensor, device: torch.device = None):
        """
        Initialize the memory.

        Args:
            K: The key tensor of shape (L, d_k)
            V: The value tensor of shape (L, d_v)
        """
        super().__init__()
        self.device = device or torch.device('cpu')
        self.K = K
        self.V = V
        self.L = K.shape[0]
        self.d_k = K.shape[1]
        self.d_v = V.shape[1]
        self.w = nn.Parameter(torch.ones(self.L, device=self.device) / self.L)

    @property
    def J(self):
        """
        Compute the memory matrix J using the current weights.
        
        The memory matrix J is calculated as the weighted sum of outer products
        between value and key vectors: J = sum_i w_i * v_i * k_i^T
        
        Returns:
            torch.Tensor: The computed memory matrix J of shape (d_v, d_k)
        """
        w_expanded = self.w.view(-1, 1, 1)
        return (w_expanded * self.V.unsqueeze(-1) * self.K.unsqueeze(1)).sum(dim=0)

    def forward(self, q: torch.Tensor):
        """
        Forward pass of the memory.

        Args:
            q: A tensor of shape (L, d_k) containing the queries.

        Returns:
            A tensor of shape (L, d_v) containing the values.
        """
        return q @ self.J
    
    def compute_cost(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the cost function value:
        C(w) = 1/2 Tr(Sigma_vv) - Tr(J Sigma_qv) + 1/2 Tr(J Sigma_qq J^T)
        """
        return squared_retrieval_error(q, v, self.J)
    
    def fit(
            self, 
            q: torch.Tensor, 
            v: torch.Tensor,
            lr: float = 1e-3,
            n_steps: int = 100
        ) -> torch.Tensor:
        """
        Fit the memory to the data.

        Args:
            q: A tensor of shape (L, d_k) containing the queries.
            v: A tensor of shape (L, d_v) containing the values.
            lr: Learning rate for optimization
            n_steps: Number of optimization steps

        Returns:
            torch.Tensor: Vector of loss values at each optimization step
        """
        optimizer = ExponentiatedSGD(self.parameters(), lr=lr)
        losses = torch.zeros(n_steps)
        for i in range(n_steps):
            optimizer.zero_grad()
            loss = squared_retrieval_error(q, v, self.J)
            losses[i] = loss.item()
            loss.backward()
            optimizer.step()
        return losses


################################################################################
# Correlation-Based Memory
################################################################################

class BaseCorrelationMemory(nn.Module):
    """
    Base class for correlation-based associative memories that optimize weights using
    ecological dynamics derived from correlation statistics.
    """
    def __init__(self, K: torch.Tensor, V: torch.Tensor, device: torch.device = None):
        """
        Initialize the memory.  

        Args:
            K: The key tensor of shape (L, d_k)
            V: The value tensor of shape (L, d_v)
            device: Device to store tensors on (default: cpu)
        """
        super().__init__()
        self.device = device or torch.device('cpu')
        self.K = K
        self.V = V

        self.L = K.shape[0]
        self.d_k = K.shape[1]
        self.d_v = V.shape[1]
        
        # Initialize tensors on specified device
        self.w = self._initialize_weights()

        # Correlation matrices
        self.Sigma_vv = torch.zeros(self.d_v, self.d_v, device=self.device)
        self.Sigma_qv = torch.zeros(self.d_k, self.d_v, device=self.device)
        self.Sigma_qq = torch.zeros(self.d_k, self.d_k, device=self.device)
        
        # Growth rates and interaction coefficients
        self.s = torch.zeros(self.L, device=self.device)
        self.A = torch.zeros(self.L, self.L, device=self.device)

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weights (can be overridden by subclasses)"""
        return torch.zeros(self.L, device=self.device)

    @property
    def J(self):
        """
        Compute the memory matrix J using the current weights.
        
        The memory matrix J is calculated as the weighted sum of outer products
        between value and key vectors: J = sum_i w_i * v_i * k_i^T
        
        Returns:
            torch.Tensor: The computed memory matrix J of shape (d_v, d_k)
        """
        w_expanded = self.w.view(-1, 1, 1)
        return (w_expanded * self.V.unsqueeze(-1) * self.K.unsqueeze(1)).sum(dim=0)

    def compute_correlations(self, q: torch.Tensor, v: torch.Tensor) -> None:
        """
        Compute correlation matrices Sigma_vv, Sigma_qv, and Sigma_qq.

        Args:
            q: queries of shape (L, d_k)
            v: values of shape (L, d_v)

        Raises:
            ValueError: If input tensors have incorrect shapes
        """
        if not (q.shape[0] == v.shape[0] == self.L):
            raise ValueError(f"Expected batch size {self.L}, got {q.shape[0]}, {v.shape[0]}")
        if not (q.shape[1] == self.d_k):
            raise ValueError(f"Expected key dim {self.d_k}, got {q.shape[1]}")
        if v.shape[1] != self.d_v:
            raise ValueError(f"Expected value dim {self.d_v}, got {v.shape[1]}")

        self.Sigma_vv = (v.T @ v) / self.L
        self.Sigma_qv = (q.T @ v) / self.L
        self.Sigma_qq = (q.T @ q) / self.L

    def compute_ecological_params(self) -> None:
        """
        Compute growth rates s_l and interaction coefficients A_ll'.
        Vectorized implementation for better efficiency.
        """            
        # All parameters are always active for this class
        # Compute parameters
        k_sigma_qv = self.K @ self.Sigma_qv
        s_values = (k_sigma_qv * self.V).sum(dim=1)
        
        # Update s
        self.s = s_values
        
        # Update A
        v_outer = self.V @ self.V.T
        k_sigma_k = self.K @ self.Sigma_qq @ self.K.T
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

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the memory.

        Args:
            q: queries of shape (L, d_k)

        Returns:
            Retrieved values of shape (L, d_v)
        """
        return q @ self.J.T

    def _validate_integration_params(self, t_max: float, dt: float) -> None:
        """Validate integration parameters"""
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        if dt <= 0 or dt >= t_max:
            raise ValueError("dt must be positive and less than t_max")

    def fit(
            self,
            q: torch.Tensor,
            v: torch.Tensor,
            t_max: float = 10.0,
            dt: float = 0.01,
            store_losses: bool = False
        ) -> torch.Tensor:
        """
        Fit using dynamical systems integration. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement fit method")

class LotkaVolterraMemory(BaseCorrelationMemory):
    """
    Correlation-based memory with non-negative weights using Lotka-Volterra dynamics.
    """
    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weights uniformly"""
        return torch.ones(self.L, device=self.device) / self.L
        
    def fit(
            self,
            q: torch.Tensor,
            v: torch.Tensor,
            t_max: float = 10.0,
            dt: float = 0.01,
            store_losses: bool = False
        ) -> torch.Tensor:
        """
        Fit using Lotka-Volterra dynamics integration.
        """
        self._validate_integration_params(t_max, dt)
        self.compute_correlations(q, v)
        self.compute_ecological_params()

        if store_losses:
            n_steps = int(t_max / dt) + 1
            losses = torch.zeros(n_steps)
            
            def store_loss(w, t):
                self.w = w
                idx = min(round(t / dt), n_steps - 1)
                losses[idx] = self.compute_cost().item()

            store_loss(self.w, 0)
            self.w = integrate.integrate_lotka_volterra(
                w_0=self.w,
                s=self.s,
                A=self.A,
                t_max=t_max,
                dt=dt,
                callback=store_loss
            )
            assert (self.w >= 0).all(), "Weights must be non-negative"
            return losses
        else:
            self.w = integrate.integrate_lotka_volterra(
                w_0=self.w,
                s=self.s,
                A=self.A,
                t_max=t_max,
                dt=dt
            )
            assert (self.w >= 0).all(), "Weights must be non-negative"
            return self.compute_cost()


class ReplicatorMemory(BaseCorrelationMemory):
    """
    Correlation-based memory with non-negative weights that sum to one using replicator dynamics.
    """
    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weights uniformly"""
        return torch.ones(self.L, device=self.device) / self.L

    def fit(
            self,
            q: torch.Tensor,
            v: torch.Tensor,
            t_max: float = 10.0,
            dt: float = 0.01,
            store_losses: bool = False
        ) -> torch.Tensor:
        """
        Fit using replicator dynamics integration.
        """
        self._validate_integration_params(t_max, dt)
        self.compute_correlations(q, v)
        self.compute_ecological_params()

        if store_losses:
            n_steps = int(t_max / dt) + 1
            losses = torch.zeros(n_steps)
            
            def store_loss(w, t):
                self.w = w
                idx = min(round(t / dt), n_steps - 1)
                losses[idx] = self.compute_cost().item()

            store_loss(self.w, 0)
            self.w = integrate.integrate_replicator_equation(
                w_0=self.w,
                s=self.s,
                A=self.A,
                t_max=t_max,
                dt=dt,
                callback=store_loss
            )
            assert (self.w >= 0).all(), "Weights must be non-negative"
            assert torch.allclose(self.w.sum(), torch.tensor(1.0)), "Weights must sum to 1"
            return losses
        else:
            self.w = integrate.integrate_replicator_equation(
                w_0=self.w,
                s=self.s,
                A=self.A,
                t_max=t_max,
                dt=dt
            )
            assert (self.w >= 0).all(), "Weights must be non-negative"
            assert torch.allclose(self.w.sum(), torch.tensor(1.0)), "Weights must sum to 1"
            return self.compute_cost()
    

################################################################################
# Invade and Adjust
################################################################################

class InvadeAndAdjustMemory(nn.Module):
    """
    Base class for online memory updates using invade-and-adjust dynamics.
    """
    def __init__(self, d_k: int, d_v: int, L: int, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.d_k = d_k
        self.d_v = d_v
        self.L = L

        # Keep track of number of tokens seen
        self.current_L = 0

        # Initialize correlation matrices
        self.reset_correlations()
        
        # Initialize tensors on specified device
        self.w = self._initialize_weights()
        self.K = torch.zeros(L, d_k, device=self.device)
        self.V = torch.zeros(L, d_v, device=self.device)
        
        # Growth rates and interaction coefficients
        self.s = torch.zeros(self.L, device=self.device)
        self.A = torch.zeros(self.L, self.L, device=self.device)

    def reset_correlations(self):
        """Reset correlation matrices to zeros"""
        self.Sigma_vv = torch.zeros(self.d_v, self.d_v, device=self.device)
        self.Sigma_qv = torch.zeros(self.d_k, self.d_v, device=self.device)
        self.Sigma_qq = torch.zeros(self.d_k, self.d_k, device=self.device)

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weights (can be overridden by subclasses)"""
        return torch.zeros(self.L, device=self.device)
    
    def update_length(self) -> None:
        """Update the length of the memory"""
        if self.current_L < self.L:
            self.current_L += 1
        else:
            raise ValueError("Memory is full")
    
    def update_correlations(self, q: torch.Tensor, v: torch.Tensor) -> None:
        """
        Update correlation matrices with a new token.

        Args:
            q: query vector of shape (d_k,)
            v: value vector of shape (d_v,)
        """
        l = self.current_L
        self.Sigma_vv = (l-1)/l * self.Sigma_vv + (v.outer(v)) / l
        self.Sigma_qv = (l-1)/l * self.Sigma_qv + (q.outer(v)) / l
        self.Sigma_qq = (l-1)/l * self.Sigma_qq + (q.outer(q)) / l

    def update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Update the memory matrix with a new token.
        """
        self.K[self.current_L-1] = k
        self.V[self.current_L-1] = v    

    @property
    def J(self):
        """
        Compute the memory matrix J using the current weights.
        
        The memory matrix J is calculated as the weighted sum of outer products
        between value and key vectors: J = sum_i w_i * v_i * k_i^T
        
        Returns:
            torch.Tensor: The computed memory matrix J of shape (d_v, d_k)
        """
        # Only use the first current_L entries which are non-zero
        if self.current_L == 0:
            return torch.zeros(self.d_v, self.d_k, device=self.device)
        
        w_active = self.w[:self.current_L].view(-1, 1, 1)
        V_active = self.V[:self.current_L]
        K_active = self.K[:self.current_L]
        
        return (w_active * V_active.unsqueeze(-1) * K_active.unsqueeze(1)).sum(dim=0)
    
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

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the memory.

        Args:
            q: queries of shape (d_k,)

        Returns:
            Retrieved values of shape (d_v,)
        """
        return self.J @ q

    def compute_ecological_params(self) -> None:
        """
        Compute growth rates s_l and interaction coefficients A_ll'.
        Vectorized implementation for better efficiency.
        """
        if self.current_L == 0:
            self.s.zero_()
            self.A.zero_()
            return
            
        # Cache active slices for better readability and potentially better performance
        active_slice = slice(None, self.current_L)
        K_active = self.K[active_slice]
        V_active = self.V[active_slice]
        
        # Compute parameters
        k_sigma_qv = K_active @ self.Sigma_qv
        s_active = (k_sigma_qv * V_active).sum(dim=1)
        
        # Update s with zero initialization
        self.s.zero_()
        self.s[active_slice] = s_active
        
        # Update A with zero initialization
        self.A.zero_()
        v_outer = V_active @ V_active.T
        k_sigma_k = K_active @ self.Sigma_qq @ K_active.T
        self.A[active_slice, active_slice] = v_outer * k_sigma_k

    def compute_invasion_criterion(self, q: torch.Tensor, v: torch.Tensor) -> bool:
        """
        Check if new token can invade. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement invasion criterion")

    def _validate_integration_params(self, t_max: float, dt: float) -> None:
        """Validate integration parameters"""
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        if dt <= 0 or dt >= t_max:
            raise ValueError("dt must be positive and less than t_max")

    def _adjust_weights(self, t_max: float, dt: float) -> None:
        """
        Adjust weights using dynamics. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement weight adjustment")
    
    def update(
            self, 
            q: torch.Tensor, 
            k: torch.Tensor,
            v: torch.Tensor, 
            t_max: float = 10.0, 
            dt: float = 0.01,
            epsilon: float = 1e-6
        ) -> torch.Tensor:
        """
        Update memory with new token using invade-and-adjust dynamics.

        Args:
            q: query vector of shape (d_k,)
            k: key vector of shape (d_k,)
            v: value vector of shape (d_v,)
            t_max: Maximum integration time if invasion occurs
            dt: Integration time step if invasion occurs
            epsilon: Small initial weight for invasion

        Returns:
            tuple: Current cost value and boolean indicating if invasion occurred
        """
        self.update_length()
        self.update_correlations(q, v)
        self.update_memory(k, v)
        self.compute_ecological_params()
        
        can_invade = self.compute_invasion_criterion(q, v)
        if can_invade:
            self.w[self.current_L-1] = epsilon
            self._adjust_weights(t_max, dt)
        else:
            self.w[self.current_L-1] = 0.0
                 
        return self.compute_cost(), can_invade


class InvadeAndAdjustLotkaVolterra(InvadeAndAdjustMemory):
    """
    Online memory updates using Lotka-Volterra dynamics with invasion criterion.
    """
    def compute_invasion_criterion(self, q: torch.Tensor, v: torch.Tensor) -> bool:
        """
        Check Lotka-Volterra invasion criterion:
        s_l - sum_{l'} A_{l,l'} w_{l'} <= 0
        """
        # s_l for new token
        s_l = self.s[self.current_L-1]
        
        # sum_{l'} A_{l,l'} w_{l'} for existing tokens
        A_sum = 0.0
        if self.current_L > 1:
            A_sum = torch.sum(self.A[self.current_L-1, :self.current_L-1] * self.w[:self.current_L-1])
            
        return (s_l - A_sum) > 0

    def _adjust_weights(self, t_max: float, dt: float) -> None:
        """Adjust weights using Lotka-Volterra dynamics"""
        self._validate_integration_params(t_max, dt)
        self.w[:self.current_L] = integrate.integrate_lotka_volterra(
            w_0=self.w[:self.current_L],
            s=self.s[:self.current_L],
            A=self.A[:self.current_L, :self.current_L],
            t_max=t_max,
            dt=dt
        )


class InvadeAndAdjustReplicator(InvadeAndAdjustMemory):
    """
    Online memory updates using replicator dynamics with invasion criterion.
    """
    def compute_invasion_criterion(self, q: torch.Tensor, v: torch.Tensor) -> bool:
        """
        Check replicator invasion criterion:
        s_l - mean(s) - sum_{l'} A_{l,l'} w_{l'} + mean(sum_{l'} A_{l',l''} w_{l''}) <= 0
        """
        if self.current_L <= 1:
            return True
            
        # s_l for new token
        s_l = self.s[self.current_L-1]
        
        # mean of existing s values
        s_mean = torch.mean(self.s[:self.current_L-1])
        
        # sum_{l'} A_{l,l'} w_{l'} for existing tokens
        A_sum = torch.sum(self.A[self.current_L-1, :self.current_L-1] * self.w[:self.current_L-1])
        
        # mean of sum_{l'} A_{l',l''} w_{l''} for existing tokens
        A_mean = torch.mean(
            torch.sum(
                self.A[:self.current_L-1, :self.current_L-1] * 
                self.w[:self.current_L-1], 
                dim=1
            )
        )
        
        return (s_l - s_mean - A_sum + A_mean) > 0

    def _adjust_weights(self, t_max: float, dt: float) -> None:
        """Adjust weights using replicator dynamics"""
        self._validate_integration_params(t_max, dt)
        if self.current_L == 1:
            self.w[0] = 1.0
        else:
            self.w[:self.current_L] = integrate.integrate_replicator_equation(
                w_0=self.w[:self.current_L],
                s=self.s[:self.current_L],
                A=self.A[:self.current_L, :self.current_L],
                t_max=t_max,
                dt=dt
            )


################################################################################
# Greedy Invasion
################################################################################

class GreedyInvasionMemory(nn.Module):
    """
    Online memory updates using greedy invasion based on Lotka-Volterra dynamics.
    
    The memory matrix is updated as:
    J_l = omega_f * J_{l-1} + omega_i * v_l * k_l^T
    
    where omega_i and omega_f are computed from ecological parameters.
    """
    def __init__(self, d_k: int, d_v: int, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.d_k = d_k
        self.d_v = d_v
        self.current_L = 0
        self.reset()

    def reset(self):
        """Reset memory state"""
        self.J = torch.zeros(self.d_v, self.d_k, device=self.device)
        self.reset_correlations()

    def reset_correlations(self):
        """Reset correlation matrices"""
        self.Sigma_vv = torch.zeros(self.d_v, self.d_v, device=self.device)
        self.Sigma_qv = torch.zeros(self.d_k, self.d_v, device=self.device)
        self.Sigma_qq = torch.zeros(self.d_k, self.d_k, device=self.device)

    def update_correlations(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Update correlation matrices with new token"""
        self.current_L += 1
        l = self.current_L
        self.Sigma_vv = (l-1)/l * self.Sigma_vv + v.outer(v) / l
        self.Sigma_qv = (l-1)/l * self.Sigma_qv + q.outer(v) / l
        self.Sigma_qq = (l-1)/l * self.Sigma_qq + q.outer(q) / l

    def compute_ecological_params(self, q, k, v):
        """Compute s_J, s_l, A_JJ, A_ll, A_Jl"""
        s_J = torch.trace(self.J @ self.Sigma_qv)
        s_l = (k @ self.Sigma_qv @ v).item()
        A_JJ = torch.trace(self.J @ self.Sigma_qq @ self.J.T)
        A_ll = (v @ v) * (k @ self.Sigma_qq @ k)
        A_Jl = (v @ self.J @ self.Sigma_qq @ k).item()
        return s_J, s_l, A_JJ, A_ll, A_Jl

    def compute_lv_gates(self, s_J, s_l, A_JJ, A_ll, A_Jl):
        """Lotka-Volterra equilibrium with nonnegativity constraints"""
        denom = A_JJ * A_ll - A_Jl**2
        omega_f = (A_ll * s_J - A_Jl * s_l) / denom
        omega_i = (A_JJ * s_l - A_Jl * s_J) / denom

        # Apply ecological invasion logic
        if omega_i <= 0:
            omega_f = s_J / A_JJ
            omega_i = 0.0
        elif omega_f <= 0:
            omega_i = s_l / A_ll
            omega_f = 0.0

        return omega_f, omega_i
    
    def compute_cost(self) -> torch.Tensor:
        """
        Compute the cost function:
        C(w) = 1/2 Tr(Sigma_vv) - Tr(J Sigma_qv) + 1/2 Tr(J Sigma_qq J^T)
        Returns:
            torch.Tensor: Cost value.
        """
        term1 = 0.5 * torch.trace(self.Sigma_vv)
        term2 = -torch.trace(self.J @ self.Sigma_qv)
        term3 = 0.5 * torch.trace(self.J @ self.Sigma_qq @ self.J.T)
        return term1 + term2 + term3

    def update(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Update memory with new token using ecological greedy invasion"""
        self.update_correlations(q, k, v)

        # First token: directly add
        if self.current_L == 1:
            self.J = v.outer(k)
            return self.compute_cost(), True

        # Compute approximate invasion margin
        s_J, s_l, A_JJ, A_ll, A_Jl = self.compute_ecological_params(q, k, v)
        margin = s_l - A_Jl * (s_J / A_JJ)

        if margin <= 0:
            # New token cannot invade → memory unchanged
            return self.compute_cost(), False

        # New token can invade → compute gated update
        omega_f, omega_i = self.compute_lv_gates(s_J, s_l, A_JJ, A_ll, A_Jl)
        self.J = omega_f * self.J + omega_i * v.outer(k)

        return self.compute_cost(), True
    

class LinearAttentionMemory(nn.Module):
    """
    Linear attention memory that assigns equal weight (1/L) to each token.
    This implements a simple running average of outer products.
    """
    
    def __init__(
        self,
        d_k: int,
        d_v: int,
        device: str = "cuda",
    ):
        """
        Initialize the linear attention memory.
        
        Args:
            d_k: Dimension of key/query vectors
            d_v: Dimension of value vectors
            device: Device to use for computations
        """
        super().__init__()
        self.device = device or torch.device('cpu')
        self.d_k = d_k
        self.d_v = d_v
        self.current_L = 0
        self.reset()

    def reset(self):
        """Reset memory state"""
        self.J = torch.zeros(self.d_v, self.d_k, device=self.device)
        self.reset_correlations()

    def reset_correlations(self):
        """Reset correlation matrices"""
        self.Sigma_vv = torch.zeros(self.d_v, self.d_v, device=self.device)
        self.Sigma_qv = torch.zeros(self.d_k, self.d_v, device=self.device)
        self.Sigma_qq = torch.zeros(self.d_k, self.d_k, device=self.device)

    def update_correlations(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Update correlation matrices with new token
        
        Args:
            q: query vector of shape (d_k,)
            k: key vector of shape (d_k,)
            v: value vector of shape (d_v,)
        """
        self.current_L += 1
        l = self.current_L
        self.Sigma_vv = (l-1)/l * self.Sigma_vv + (v.outer(v)) / l
        self.Sigma_qv = (l-1)/l * self.Sigma_qv + (q.outer(v)) / l
        self.Sigma_qq = (l-1)/l * self.Sigma_qq + (q.outer(q)) / l
        
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
    
    def update(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Update memory with new token using linear attention (equal weighting).
        
        Args:
            q: query vector of shape (d_k,)
            k: key vector of shape (d_k,)
            v: value vector of shape (d_v,)
            
        Returns:
            torch.Tensor: Current cost value
        """
        self.update_correlations(k, q, v)
        
        # Linear attention: weight = 1/L for all tokens
        omega_i = 1.0 / self.current_L
        omega_f = 1.0 - omega_i
        
        # Update memory
        self.J = omega_f * self.J + omega_i * v.outer(k)
        
        return self.compute_cost(), True
