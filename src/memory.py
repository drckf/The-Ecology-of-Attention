import torch


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
    
    
