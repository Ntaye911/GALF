# GALF: Geometrically Adaptive Low-Rank Fusion

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TPAMI-blue)](https://ieeexplore.ieee.org/document/XXXXXXX)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-orange)](LICENSE)

Official implementation of _"Geometrically Adaptive Low-Rank Fusion for Multi-Label Graph Classification under Heterogeneous Noise"_ (IEEE TPAMI, 2026).

## 🚀 Features
- ✅ **Heterogeneous noise modeling** (edge + feature corruption)
- ✅ **Forman–Ricci curvature** for geometric adaptation
- ✅ **ADMM-based low-rank fusion** with convergence guarantees
- ✅ Supports **6 real-world multi-label datasets**: Cora-ML, PubMed-MC, PPI, Yeast, Delicious, EurLEX

## 📊 Results
GALF achieves **up to 9.1% higher Micro-F1** than 12 baselines under severe noise.

## 🛠️ Installation
```bash
git clone https://github.com/Ntaye911/GALF.git
cd GALF
pip install -r requirements.txt