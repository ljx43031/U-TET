# U-TET:通用 Transformer 增强跟踪框架

[English](README.md) | 简体中文

以下论文的官方实现:

> **U-TET: A Universal Transformer-Enhanced Tracking Framework for Heterogeneous radar-RF Fusion in Complex Low-Altitude Environments**
> (U-TET:面向复杂低空环境异构雷达-射频融合的通用 Transformer 增强跟踪框架)
> Jingxian Liu, Ronglan Huang, Dehuan Wan, Zhi Quan, Hui Wang, Zhilin Mei, Lizhi You
> *IEEE Transactions on Vehicular Technology*(IEEE 车辆技术汇刊,TVT)

<p align="center">
  <a href="https://doi.org/10.1109/TVT.2026.3753453"><img src="https://img.shields.io/badge/DOI-10.1109%2FTVT.2026.3753453-blue" alt="DOI"></a>
  <a href="https://ieeexplore.ieee.org/document/11795160"><img src="https://img.shields.io/badge/IEEE_Xplore-Early_Access-green" alt="IEEE Xplore"></a>
  <img src="https://img.shields.io/badge/Journal-IEEE_Transactions_on_Vehicular_Technology-red" alt="Journal">
</p>

## 📰 新闻动态

- **【2026-11】论文已被《IEEE Transactions on Vehicular Technology》正式接收,现已在 IEEE Xplore "Early Access"(抢先出版)专区上线!** 🎉
  - 论文页面:<https://ieeexplore.ieee.org/document/11795160>
  - DOI:`10.1109/TVT.2026.3753453`

## ✨ 方法概述

在低空安防场景中,传统多传感器跟踪器需要预先定义的运动学模型,且只能在中等杂波环境下可靠工作。**U-TET** 将跟踪解耦为两个阶段:

1. **深度净化器——物理感知的 RF 引导 Transformer(RFG-Trans)**:利用跨模态注意力机制,让被动 RF 检测结果引导从严重杂波中提取有效雷达观测,并以物理感知(jerk)损失配合两阶段课程学习进行正则化。采用 clip-sequential 滑动窗推理与缓冲匹配平均融合策略,将网络输出转化为高保真的**"虚拟量测"流**——一个与跟踪器无关的标准化接口。
2. **贝叶斯跟踪器**:任意经典后端(GLMB 滤波器、GNN-MN 等)消费虚拟量测,完成关联、平滑与航迹生命周期管理,保留严格的运动学约束与实时性能。

在高保真数字孪生仿真上(10 个高机动目标、每传感器每扫描杂波强度高达 λ = 40、外加状态相关的 RF 多径鬼影),混合架构——尤其是 GLMB 后端——在传统基线 MOTA 崩溃为**负值**的场景下仍保持 **MOTA > 0.86**,同时将单帧延迟控制在约 5 ms。

## 📖 引用

如果本工作对您的研究有帮助,请引用:

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

## 📁 仓库结构

```
U-TET/
├── Maneuvering_target_simulation_v2_r.py   # 数字孪生仿真:3D 机动目标、雷达/RF 量测、杂波与多径鬼影
├── Observation_Preprocessing_v2.py         # 雷达/RF 观测预处理与张量化(归一化、截断/零填充、有效性掩码)
├── Training_Pipeline_v3_r.py               # RFG-Trans 训练(物理感知二分匹配损失、两阶段课程学习)
├── v3_checkpoint_best_r.pth                # 预训练 RFG-Trans 权重
├── Comprehensive_Tracker_Comparison_v2.py  # 五种范式对比:Pure GLMB / Standard GNN / Pure Transformer / Hybrid GLMB / Hybrid GNN
├── Ultimate_Evaluation_System.py           # 评估:OSPA、MOTA、Precision/Recall/F1、cardinality 误差、延迟、OPS
├── Ultimate_Comparison_Final.py            # 最终跨方法对比与结果面板生成
├── plot_comparison.m                       # MATLAB 对比结果绘图辅助脚本
├── Comparison_Metrics.csv                  # 对比指标表
├── comparison_data.mat                     # 对比数据(mat)
└── filterpy/                               # 贝叶斯后端所用的 filterpy 库副本
```

## ⚙️ 环境要求

- Python 3.x
- PyTorch(建议 CUDA 版;论文实验在单块 NVIDIA GeForce RTX 4090 上完成)
- NumPy、SciPy、Matplotlib

```bash
pip install torch numpy scipy matplotlib
```

## 🚀 使用方法

典型流水线按以下顺序运行:

```bash
# 1. 仿真机动目标与异构雷达/RF 观测(数字孪生)
python Maneuvering_target_simulation_v2_r.py

# 2. 将观测预处理为时空张量
python Observation_Preprocessing_v2.py

# 3. 训练 RFG-Trans 深度净化器(或直接使用提供的权重)
python Training_Pipeline_v3_r.py

# 4. 运行五范式跟踪对比与评估系统
python Comprehensive_Tracker_Comparison_v2.py
python Ultimate_Evaluation_System.py
python Ultimate_Comparison_Final.py
```

预训练权重 `v3_checkpoint_best_r.pth` 可被对比/评估脚本直接加载,复现论文中的结果。

## 📄 许可

仅供学术与研究用途发布。若本仓库对您的研究有所帮助,请引用上述论文。
