import torch
from math import sqrt



class SyntheticDataGenerator:
    """
    A class for generating synthetic data for memory models.
    
    This class creates synthetic token data with controlled correlation structure,
    and generates corresponding query, key, and value projections.
    
    Attributes:
        N (int): Dimension of the latent space
        D_v (int): Dimension of value vectors
        D_k (int): Dimension of key/query vectors
        rho (float): Correlation parameter controlling the structure of the covariance matrix
        sim (float): Correlation parameter between 0 and 1.
                     Higher values increase correlation between key and query vectors.
                     Default: 0.
        W_v (torch.Tensor): Value projection matrix of shape (D_v, N)
        W_k (torch.Tensor): Key projection matrix of shape (D_k, N)
        W_q (torch.Tensor): Query projection matrix of shape (D_k, N)
        cov (torch.Tensor): Covariance matrix of shape (N, N)
    """
    def __init__(self, N, D_v, D_k, rho, sim=0, device='cpu'):
        """
        Initialize the synthetic data generator.
        
        Args:
            N (int): Dimension of the latent space
            D_v (int): Dimension of value vectors
            D_k (int): Dimension of key/query vectors
            rho (float): Correlation parameter controlling the structure of the covariance matrix
            sim (float): Correlation parameter between 0 and 1.
                         Higher values increase correlation between key and query vectors.
                         Default: 0.
            device (str): Device to store the tensors on.
        """
        self.N = N
        self.D_v = D_v
        self.D_k = D_k
        self.rho = rho
        self.sim = sim
        self.device = device
        self.generate_cov(self.rho)
        self.generate_projections(self.sim)

    def generate_cov(self, rho, dim=None):
        """
        Generate a covariance matrix with controlled correlation structure.
        
        Creates a covariance matrix with a structure controlled by the rho parameter.
        When rho=1, it's an identity matrix (diagonal). As rho decreases toward 0,
        off-diagonal elements increase, creating correlations between dimensions.
        The resulting matrix is guaranteed to be positive semi-definite.
        
        Args:
            rho (float): Correlation parameter between 0 and 1.
                         Higher values increase diagonal dominance.
            dim (int, optional): Dimension of the random projection matrix.
                                If None, uses self.N. Default: None.
                                
        Returns:
            None: Updates the instance variable cov in-place.
        """
        eye = torch.eye(self.N, device=self.device)
        V = torch.randn(self.N, self.N if dim is None else dim, device=self.device)
        self.cov = rho * eye + (1 - rho) * V @ V.T / sqrt(self.N)
    
    def generate_projections(self, sim=0):
        """
        Generate random projection matrices for values, keys, and queries.
        
        Creates three random projection matrices (W_v, W_k, W_q) that map from
        the latent space (dimension N) to the value space (dimension D_v) and 
        key/query space (dimension D_k).
        
        These matrices are used to generate correlated query, key, and value vectors
        from the same underlying latent representation.

        Args:
            sim (float): Correlation parameter between 0 and 1.
                         Higher values increase correlation between key and query vectors.
                         Default: 0.
        
        Returns:
            None: Updates the instance variables W_v, W_k, and W_q in-place.
        """
        self.W_v = torch.randn(self.D_v, self.N, device=self.device ) / sqrt(self.D_v)
        self.W_k = torch.randn(self.D_k, self.N, device=self.device) / sqrt(self.D_k)
        self.W_q = sim * self.W_k + (1 - sim) * torch.randn(self.D_k, self.N, device=self.device) / sqrt(self.D_k)

    @property
    def SigmaVV(self):
        """
        Calculate the value-value covariance matrix.
        
        Computes the covariance matrix between value vectors by projecting
        the latent covariance matrix through the value projection matrix.
        
        Returns:
            torch.Tensor: Covariance matrix of shape (D_v, D_v) representing
                         correlations between value dimensions.
        """
        return self.W_v @ self.cov @ self.W_v.T
    
    @property
    def SigmaVQ(self):
        """
        Calculate the value-query covariance matrix.
        
        Computes the covariance matrix between value and query vectors by projecting
        the latent covariance matrix through the respective projection matrices.
        
        Returns:
            torch.Tensor: Covariance matrix of shape (D_v, D_k) representing
                         correlations between value and query dimensions.
        """
        return self.W_v @ self.cov @ self.W_q.T
    
    @property
    def SigmaQK(self):
        """
        Calculate the query-key covariance matrix.
        
        Computes the covariance matrix between query and key vectors by projecting
        the latent covariance matrix through the respective projection matrices.
        
        Returns:
            torch.Tensor: Covariance matrix of shape (D_k, D_k) representing
                         correlations between query and key dimensions.
        """
        return self.W_q @ self.cov @ self.W_k.T
    
    @property
    def SigmaKK(self):
        """
        Calculate the key-key covariance matrix.
        
        Computes the covariance matrix between key vectors by projecting
        the latent covariance matrix through the key projection matrix.
        
        Returns:
            torch.Tensor: Covariance matrix of shape (D_k, D_k) representing
                         correlations between key dimensions.
        """
        return self.W_k @ self.cov @ self.W_k.T
    
    @property
    def SigmaQQ(self):
        """
        Calculate the query-query covariance matrix.
        
        Computes the covariance matrix between query vectors by projecting
        the latent covariance matrix through the query projection matrix.
        
        Returns:
            torch.Tensor: Covariance matrix of shape (D_k, D_k) representing
                         correlations between query dimensions.
        """
        return self.W_q @ self.cov @ self.W_q.T

    def generate_tokens(self, L):
        """
        Generate random token embeddings from the latent distribution.
        
        Samples L token embeddings from the multivariate normal distribution
        defined by the latent covariance matrix.
        
        Args:
            L (int): Number of tokens to generate.
            
        Returns:
            torch.Tensor: Generated token embeddings of shape (L, N) where N
                         is the dimension of the latent space.
        """
        r = torch.randn(L, self.N, device=self.device)
        chol = torch.linalg.cholesky(self.cov)
        return r @ chol.T
    
    def generate_QKV(self, L):
        """
        Generate query, key, and value vectors for a sequence of tokens.
        
        Samples L token embeddings from the latent distribution and projects them
        through the query, key, and value matrices to generate the corresponding
        attention vectors.
        
        Args:
            L (int): Number of tokens to generate.
            
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Original token embeddings of shape (L, N)
                - torch.Tensor: Query vectors of shape (L, D_k)
                - torch.Tensor: Key vectors of shape (L, D_k)
                - torch.Tensor: Value vectors of shape (L, D_v)
        """
        tokens = self.generate_tokens(L)
        return tokens, tokens @ self.W_q.T, tokens @ self.W_k.T, tokens @ self.W_v.T