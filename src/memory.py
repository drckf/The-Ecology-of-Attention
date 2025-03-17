import torch
import torch.nn as nn



def squared_retrieval_error(
         q: torch.Tensor, 
         v: torch.Tensor,
         associative_memory: nn.Module
    ):
    """
    Compute the retrieval error of the associative memory.
    """
    return (associative_memory.forward(q) - v).pow(2).sum(dim=1).mean()


class FullBatchMemory(nn.Module):
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
        return self.J @ q
    
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


class EmpiricalCovarianceMatrix:
    """
    A class for updating and storing a covariance matrix.
    """
    def __init__(self, n_features: int):
        """
        Initialize the covariance matrix.

        Args:
            n_features: The number of features in the data.
        """ 
        self.L = 0
        self.n_features = n_features
        self.cov = torch.zeros(n_features, n_features)

    def compute(self, x: torch.Tensor):
        """
        Update the covariance matrix with a batch of data.

        Args:
            x: A tensor of shape (batch_size, n_features) 
            containing the data to update the covariance matrix with. 

        Returns:
            The updated covariance matrix.
        """
        self.L = x.shape[0]
        self.cov = x.T @ x
        self.cov /= self.L
        return self.cov

    def update(self, x: torch.Tensor):
        """
        Update the covariance matrix with a single data point.

        Args:
            x: A tensor of shape (n_features,) containing the data to 
            update the covariance matrix with.

        Returns:
            The updated covariance matrix.  
        """
        self.L += 1
        self.cov = (self.L - 1) / self.L * self.cov + x.T @ x / self.L
        return self.cov
    
    
