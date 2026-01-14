import os
import numpy as np
import torch
from sklearn.metrics import f1_score, label_ranking_loss
from datasets import DATASETS
from galf_admm import GALF_ADMM, compute_forman_ricci
import pandas as pd

os.makedirs("tables", exist_ok=True)

def inject_heterogeneous_noise(X, A, p_e=0.3, p_f=0.2):
    n = X.shape[0]
    degrees = A.sum(1)
    edge_prob = p_e * (1 - degrees / (degrees.max() + 1e-8))
    A_noisy = A.copy()
    for i in range(n):
        for j in range(i+1, n):
            if np.random.rand() < edge_prob[i] / 2:
                A_noisy[i, j] = 1 - A_noisy[i, j]
                A_noisy[j, i] = A_noisy[i, j]
    feat_var = X.var(1)
    feat_prob = p_f * (feat_var / (feat_var.max() + 1e-8))
    X_noisy = X.copy()
    for i in range(n):
        mask = np.random.rand(X.shape[1]) < feat_prob[i]
        X_noisy[i, mask] = 0
    return X_noisy, A_noisy

def evaluate_metrics(Y_true, Y_pred):
    micro_f1 = f1_score(Y_true, Y_pred, average='micro')
    macro_f1 = f1_score(Y_true, Y_pred, average='macro')
    lrl = label_ranking_loss(Y_true, Y_pred)
    return micro_f1, macro_f1, lrl

def train_galf_admm(X_noisy, A, Y, X_clean, r=128):
    n, d = X_noisy.shape
    L = Y.shape[1]
    curvature = compute_forman_ricci(A)
    avg_curv = curvature.sum(1) / (A.sum(1) + 1e-8)
    D_kappa = torch.diag(1 / (1 + torch.exp(-torch.tensor(avg_curv * 2.0))))
    sigma = np.zeros(n)
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        if len(neighbors) > 0:
            sigma[i] = np.mean([np.linalg.norm(X_noisy[i] - X_noisy[j])**2 for j in neighbors])
        else:
            sigma[i] = 1.0
    D_sigma = torch.diag(torch.tensor(sigma).sqrt())
    
    model = GALF_ADMM(d, r, L)
    V = torch.zeros(n, r)
    Y_dual = torch.zeros(n, r)
    
    for epoch in range(200):
        Z, V, Y_dual, loss = model.admm_step(X_noisy, X_clean, Y, D_kappa, D_sigma, V, Y_dual, lr=0.01)
    
    with torch.no_grad():
        Z_cpu = Z.cpu()
        logits = model.W(Z_cpu)
        preds = (torch.sigmoid(logits) > 0.5).numpy()
    return preds

# Run experiments
all_results = {}
for name in ['Cora-ML', 'PubMed-MC', 'PPI', 'Yeast', 'Delicious', 'EurLEX']:
    print(f"Processing {name}...")
    X, A, Y = DATASETS[name]()
    X_noisy, A_noisy = inject_heterogeneous_noise(X, A, p_e=0.3, p_f=0.2)
    r = 256 if name == 'PPI' else 128
    preds = train_galf_admm(X_noisy, A_noisy, Y, X, r=r)
    micro, macro, lrl = evaluate_metrics(Y, preds)
    all_results[name] = {'Micro-F1': micro, 'Macro-F1': macro, 'LRL': lrl}

# Save tables
df = pd.DataFrame(all_results).T
df.to_csv('tables/galf_results.csv')
print("✅ Results saved to tables/galf_results.csv")