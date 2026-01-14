import torch
import torch.nn as nn
import numpy as np
import networkx as nx

class GALF_ADMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, lambda1=1.0, lambda2=0.5, alpha=2.0, rho=1.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.rho = rho
        self.W = nn.Linear(hidden_dim, num_labels)
        self.embedding = nn.Parameter(torch.randn(input_dim, hidden_dim))
        
    def forward(self, X):
        Z = X @ self.embedding
        logits = self.W(Z)
        return Z, logits
    
    def admm_step(self, X_noisy, X_clean, Y, D_kappa, D_sigma, V, Y_dual, lr=0.01):
        X_noisy = torch.tensor(X_noisy, dtype=torch.float32)
        X_clean = torch.tensor(X_clean, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        D_kappa = D_kappa
        D_sigma = D_sigma
        V = V
        Y_dual = Y_dual
        
        Z = X_noisy @ self.embedding
        logits = self.W(Z)
        cls_loss = nn.BCEWithLogitsLoss()(logits, Y)
        term1 = self.lambda1 * torch.norm(D_kappa.sqrt() @ (Z - X_clean) @ D_sigma, p='fro')**2
        term2 = (self.rho / 2) * torch.norm(D_kappa @ Z - V + Y_dual / self.rho, p='fro')**2
        loss = cls_loss + term1 + term2
        loss.backward()
        
        with torch.no_grad():
            self.embedding -= lr * self.embedding.grad
            self.W.weight -= lr * self.W.weight.grad
            self.W.bias -= lr * self.W.bias.grad
            
        self.embedding.grad.zero_()
        self.W.weight.grad.zero_()
        self.W.bias.grad.zero_()
        
        Z_new = X_noisy @ self.embedding
        M = D_kappa @ Z_new + Y_dual / self.rho
        U, S, Vt = torch.svd(M)
        S_thresh = torch.clamp(S - self.lambda2 / self.rho, min=0)
        V_new = U @ torch.diag(S_thresh) @ Vt.T
        Y_dual_new = Y_dual + self.rho * (D_kappa @ Z_new - V_new)
        
        return Z_new.detach(), V_new.detach(), Y_dual_new.detach(), loss.item()

def compute_forman_ricci(A):
    n = A.shape[0]
    G = nx.from_numpy_array(A)
    degrees = np.array([G.degree(i) for i in range(n)])
    curvature = np.zeros((n, n))
    for i in range(n):
        for j in G.neighbors(i):
            if i < j:
                w_i = degrees[i]
                w_j = degrees[j]
                w_e = 1.0
                sum_term = 0
                for k in G.neighbors(i):
                    if k != j:
                        sum_term += 1.0 / w_i
                for k in G.neighbors(j):
                    if k != i:
                        sum_term += 1.0 / w_j
                kappa = w_e * (w_i/w_e + w_j/w_e - sum_term)
                curvature[i, j] = kappa
                curvature[j, i] = kappa
    return curvature
