# U-TET: Universal Transformer-Enhanced Tracking

English | [简体中文](README_zh-CN.md)

Official implementation of:

> **U-TET: A Universal Transformer-Enhanced Tracking Framework for Heterogeneous radar-RF Fusion in Complex Low-Altitude Environments**
> Jingxian Liu, Ronglan Huang, Dehuan Wan, Zhi Quan, Hui Wang, Zhilin Mei, Lizhi You
> *IEEE Transactions on Vehicular Technology* (TVT)

<p align="center">
  <a href="https://doi.org/10.1109/TVT.2026.3753453"><img src="https://img.shields.io/badge/DOI-10.1109%2FTVT.2026.3753453-blue" alt="DOI"></a>
  <a href="https://ieeexplore.ieee.org/document/11795160"><img src="https://img.shields.io/badge/IEEE_Xplore-Early_Access-green" alt="IEEE Xplore"></a>
  <img src="https://img.shields.io/badge/Journal-IEEE_Transactions_on_Vehicular_Technology-red" alt="Journal">
</p>

## 📰 News

- **[2026-11] The paper has been accepted for publication in *IEEE Transactions on Vehicular Technology* and is now available in the "Early Access" area on IEEE Xplore!** 🎉
  - Paper page: <https://ieeexplore.ieee.org/document/11795160>
  - DOI: `10.1109/TVT.2026.3753453`

## ✨ Overview

In low-altitude security, traditional multi-sensor trackers require pre-defined kinematic models and operate reliably only under moderate clutter. **U-TET** decouples tracking into two stages:

1. **Deep Purifier — Physics-Aware RF-Guided Transformer (RFG-Trans)**: uses cross-modal attention so that passive RF detections guide the extraction of valid radar observations from severe clutter, regularized by a physics-aware (jerk) loss with two-stage curriculum learning. A clip-sequential sliding-window inference strategy with buffer-match-average fusion turns the network output into a stream of high-fidelity **"virtual measurements"** — a standardized, tracker-agnostic interface.
2. **Bayesian Tracker**: an arbitrary classical backend (GLMB-based filter, GNN-MN, ...) consumes the virtual measurements for association, smoothing, and lifecycle management, retaining rigorous kinematic constraints and real-time performance.

Simulations on a high-fidelity digital twin (10 highly maneuvering targets, clutter intensity up to λ = 40 per scan per sensor, plus state-dependent RF multipath ghosts) show that the hybrid architecture — especially with the GLMB backend — sustains **MOTA > 0.86** where traditional baselines collapse to **negative MOTA**, while keeping the per-frame latency at ≈ 5 ms.

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026utet,
  author  = {Liu, Jingxian and Huang, Ronglan and Wan, Dehuan and Quan, Zhi and Wang, Hui and Mei, Zhilin and You, Lizhi},
  title   = {U-TET: A Universal Transformer-Enhanced Tracking Framework for Heterogeneous radar-{RF} Fusion in Complex Low-Altitude Environments},
  journal = {IEEE Transactions on Vehicular Technology},
  year    = {2026},
  note    = {Early Access},
  doi     = {10.1109/TVT.2026.3753453}
}
```

## 📁 Repository Structure

```
U-TET/
├── Maneuvering_target_simulation_v2_r.py   # Digital-twin simulation: 3D maneuvering targets, radar/RF measurements, clutter & multipath ghosts
├── Observation_Preprocessing_v2.py         # Radar/RF observation preprocessing & tensorization (normalization, truncation/zero-padding, validity masks)
├── Training_Pipeline_v3_r.py               # RFG-Trans training (physics-aware bipartite-matching loss, two-stage curriculum)
├── v3_checkpoint_best_r.pth                # Pre-trained RFG-Trans checkpoint
├── Comprehensive_Tracker_Comparison_v2.py  # Five-paradigm comparison: Pure GLMB / Standard GNN / Pure Transformer / Hybrid GLMB / Hybrid GNN
├── Ultimate_Evaluation_System.py           # Evaluation: OSPA, MOTA, Precision/Recall/F1, cardinality error, latency, OPS
├── Ultimate_Comparison_Final.py            # Final cross-method comparison & dashboard generation
├── plot_comparison.m                       # MATLAB plotting helper for comparison results
├── Comparison_Metrics.csv                  # Comparison metrics table
├── comparison_data.mat                     # Comparison data (mat)
└── filterpy/                               # Bundled filterpy library used by the Bayesian backends
```

## ⚙️ Requirements

- Python 3.x
- PyTorch (CUDA recommended; experiments were run on a single NVIDIA GeForce RTX 4090)
- NumPy, SciPy, Matplotlib

```bash
pip install torch numpy scipy matplotlib
```

## 🚀 Usage

A typical pipeline runs in the following order:

```bash
# 1. Simulate maneuvering targets and heterogeneous radar/RF observations (digital twin)
python Maneuvering_target_simulation_v2_r.py

# 2. Preprocess observations into spatiotemporal tensors
python Observation_Preprocessing_v2.py

# 3. Train the RFG-Trans deep purifier (or skip with the provided checkpoint)
python Training_Pipeline_v3_r.py

# 4. Run the five-paradigm tracking comparison and the evaluation system
python Comprehensive_Tracker_Comparison_v2.py
python Ultimate_Evaluation_System.py
python Ultimate_Comparison_Final.py
```

The pre-trained checkpoint `v3_checkpoint_best_r.pth` can be loaded directly by the comparison/evaluation scripts to reproduce the results reported in the paper.

## 📄 License

Released for academic and research use. Please cite the paper above if this repository contributes to your research.
