import torch



class SyntheticDataGenerator:
    def __init__(self, N, D_v, D_k):
        self.N = N
        self.D_v = D_v
        self.D_k = D_k

        self.W_v = torch.zeros(D_v, N)
        self.W_k = torch.zeros(D_k, N)
        self.W_q = torch.zeros(D_k, N)
        self.cov = torch.zeros(N, N)

    def generate_cov(self, rho, dim):
        eye = torch.eye(self.N)
        V = torch.randn(self.N, dim)
        return rho * eye + (1 - rho) * V @ V.T
    
    def generate_projections(self):
        self.W_v = torch.randn(self.D_v, self.N)
        self.W_k = torch.randn(self.D_k, self.N)
        self.W_q = torch.randn(self.D_k, self.N)

    @property
    def SigmaVV(self):
        return self.W_v @ self.cov @ self.W_v.T
    
    @property
    def SigmaVQ(self):
        return self.W_v @ self.cov @ self.W_q.T
    
    @property
    def SigmaQK(self):
        return self.W_q @ self.cov @ self.W_k.T
    
    @property
    def SigmaKK(self):
        return self.W_k @ self.cov @ self.W_k.T
    
    @property
    def SigmaQQ(self):
        return self.W_q @ self.cov @ self.W_q.T

    def generate_tokens(self, L):
        r = torch.randn(L, self.N)
        chol = torch.linalg.cholesky(self.cov)
        return r @ chol.T
    
    def generate_QKV(self, L):
        tokens = self.generate_tokens(L)
        return tokens, tokens @ self.W_q.T, tokens @ self.W_k.T, tokens @ self.W_v.T
    
    