"""
Compact_Comparison.py
精简对比脚本：
- 直接从 Ultimate_Comparison_Final.py 和 Ultimate_Evaluation_System.py 导入算法
- 对比5种方法
- 可调整的传感器误差参数
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
import scipy.io as sio
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 从两个脚本导入算法类
# ==========================================
# 从 Ultimate_Comparison_Final.py 导入
from Ultimate_Comparison_Final import (
    Algo_Pure_GLMB,
    Algo_Pure_Transformer as Algo_Trans_Final,
    Algo_Hybrid_Fused as Algo_Hybrid_GLMB,
    ospa_dist,
    Denormalizer
)

# 从 Ultimate_Evaluation_System.py 导入
from Ultimate_Evaluation_System import (
    Algo_Standard_Radar_Tracker as Algo_GNN_MN,
    Algo_Pure_Transformer as Algo_Trans_Eval,
    Algo_Hybrid_Fused as Algo_Hybrid_GNN_MN,
    Evaluator,
    TrackState
)

# 共用模块
from Training_Pipeline_v3_r import RFGuidedTransformerTracker
from Maneuvering_target_simulation_v2_r import Trajectory_auto_generator_multitarget
from Observation_Preprocessing_v2 import PointCloudPreprocessor


# ==========================================
# 配置参数 (方便调整)
# ==========================================
class Config:
    """集中管理所有可调参数"""

    # 场景参数
    TOTAL_FRAMES = 500
    NUM_TARGETS = 10
    CLUTTER_MEAN = 20

    # ============================================
    # Radar 传感器参数
    # 格式: [r_min, r_max, angle_std(度), range_std(米), P_D]
    # angle_std: 角度测量误差标准差 (度)
    # range_std: 距离测量误差标准差 (米)
    # P_D: 检测概率
    # ============================================
    RADAR_PARAMS = [1, 10000, 0.5, 3.5, 0.85]

    # ============================================
    # RF 传感器参数
    # 格式: [r_max, angle_std(度), P_D]
    # ============================================
    RF_PARAMS = [10000, 2.5, 0.95]

    # 模型参数
    MAX_RANGE = 10000
    MODEL_PATH = 'v3_checkpoint_best_r.pth'

    # 评估参数
    OSPA_C = 100.0
    OSPA_P = 2.0
    MATCH_THRESH = 50.0


def create_sensor_params(radar_angle_std=0.5, radar_range_std=3.5, radar_pd=0.85,
                         rf_angle_std=2.5, rf_pd=0.95):
    """
    便捷函数：创建传感器参数

    Args:
        radar_angle_std: 雷达角度误差标准差 (度)
        radar_range_std: 雷达距离误差标准差 (米)
        radar_pd: 雷达检测概率
        rf_angle_std: RF角度误差标准差 (度)
        rf_pd: RF检测概率

    Returns:
        radar_params, rf_params
    """
    radar_params = [1, 10000, radar_angle_std, radar_range_std, radar_pd]
    rf_params = [10000, rf_angle_std, rf_pd]
    return radar_params, rf_params


# ==========================================
# 数据生成
# ==========================================
def generate_data(config):
    """生成仿真数据"""
    print(f"Generating {config.TOTAL_FRAMES} frames with {config.NUM_TARGETS} targets...")
    print(
        f"  Radar: angle_std={config.RADAR_PARAMS[2]}°, range_std={config.RADAR_PARAMS[3]}m, P_D={config.RADAR_PARAMS[4]}")
    print(f"  RF: angle_std={config.RF_PARAMS[1]}°, P_D={config.RF_PARAMS[2]}")
    print(f"  Clutter: {config.CLUTTER_MEAN} points/frame")

    sim = Trajectory_auto_generator_multitarget(TargetNum=config.NUM_TARGETS)
    _, Observations, _, Radar_obs, RF_obs, Clutter_rad, Clutter_rf = \
        sim.show_data(
            TimeStepRange=[0, config.TOTAL_FRAMES // 20 + 1],
            Camera=[10000, 0.002],
            Radar=config.RADAR_PARAMS,
            RFsensor=config.RF_PARAMS,
            clutter_num_mean=config.CLUTTER_MEAN
        )

    # 构建GT
    gt_list = []
    gt_array = []

    for t in range(config.TOTAL_FRAMES):
        pos, ids = [], []
        for i, obs in enumerate(Observations):
            path, st = obs[0], int(obs[1])
            if st <= t < st + len(path):
                p_dat = path[t - st]
                p, a, r = np.deg2rad(p_dat[0]), np.deg2rad(p_dat[1]), p_dat[2]
                x = r * np.cos(p) * np.cos(a)
                y = r * np.cos(p) * np.sin(a)
                z = r * np.sin(p)
                pos.append([x, y, z])
                ids.append(i + 1)

        pos_arr = np.array(pos) if pos else np.empty((0, 3))
        ids_arr = np.array(ids) if ids else np.array([])

        gt_list.append({'pos': pos_arr, 'ids': ids_arr})
        gt_array.append(pos_arr)

    # 构建输入
    radar_in, rf_in = [], []
    for t in range(config.TOTAL_FRAMES):
        r_f, rf_f = [], []
        if t < len(Clutter_rad):
            r_f.extend(Clutter_rad[t])
        if t < len(Clutter_rf):
            rf_f.extend(Clutter_rf[t])
        for i, obs in enumerate(Radar_obs):
            st = int(Observations[i][1])
            if st <= t < st + len(obs) and len(obs[t - st]) > 0:
                r_f.append(obs[t - st])
        for i, obs in enumerate(RF_obs):
            st = int(Observations[i][1])
            if st <= t < st + len(obs) and len(obs[t - st]) > 0:
                rf_f.append(obs[t - st])
        radar_in.append(r_f)
        rf_in.append(rf_f)

    return {
        'gt_list': gt_list,
        'gt_array': gt_array,
        'radar_in': radar_in,
        'rf_in': rf_in,
        'observations': Observations
    }


# ==========================================
# 结果格式转换
# ==========================================
def convert_array_to_dict(results_array):
    """将 Final 脚本的数组格式转换为字典格式"""
    results_dict = []
    for arr in results_array:
        results_dict.append({
            'pos': arr if len(arr) > 0 else np.empty((0, 3)),
            'ids': np.arange(len(arr)) if len(arr) > 0 else np.array([])
        })
    return results_dict


# ==========================================
# 统一评估函数
# ==========================================
def evaluate_all(gt_list, results_dict, latency, name, config):
    """统一评估函数"""
    tp, fp, fn = 0, 0, 0
    total_gt = 0
    ospa_hist = []
    card_gt, card_est = [], []

    for t, (gt, res) in enumerate(zip(gt_list, results_dict)):
        gt_pos = gt['pos']
        res_pos = res['pos']

        # OSPA
        ospa_hist.append(ospa_dist(gt_pos, res_pos, config.OSPA_C, config.OSPA_P))

        # Cardinality
        card_gt.append(len(gt_pos))
        card_est.append(len(res_pos))

        # MOTA
        total_gt += len(gt_pos)

        if len(gt_pos) > 0 and len(res_pos) > 0:
            d = np.linalg.norm(gt_pos[:, None, :] - res_pos[None, :, :], axis=2)
            r, c = linear_sum_assignment(d)

            matched_gt, matched_res = set(), set()
            for ri, ci in zip(r, c):
                if d[ri, ci] < config.MATCH_THRESH:
                    tp += 1
                    matched_gt.add(ri)
                    matched_res.add(ci)

            fn += len(gt_pos) - len(matched_gt)
            fp += len(res_pos) - len(matched_res)
        else:
            fn += len(gt_pos)
            fp += len(res_pos)

    mota = 1 - (fn + fp) / max(1, total_gt)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    card_error = np.mean(np.abs(np.array(card_gt) - np.array(card_est)))

    return {
        'Name': name,
        'MOTA': mota,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Avg_OSPA': np.mean(ospa_hist),
        'Card_Error': card_error,
        'Latency_ms': np.mean(latency),
        'ospa_curve': ospa_hist,
        'card_gt': card_gt,
        'card_est': card_est
    }


# ==========================================
# 主比较函数
# ==========================================
def run_comparison(config, device='cuda'):
    """运行5种算法的对比"""

    if not torch.cuda.is_available():
        device = 'cpu'
    print(f"Using device: {device}")

    # 1. 生成数据
    data = generate_data(config)

    # 2. 加载模型
    print("\nLoading Transformer model...")
    model = RFGuidedTransformerTracker(30, 128, 4, 3, 3, 256, 0.1, 3).to(device)
    try:
        ckpt = torch.load(config.MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model ({e})")
        return None, None, None

    prep = PointCloudPreprocessor(32, 100, 50, 30, config.MAX_RANGE)

    # 3. 定义算法
    algorithms = {
        '1_Pure_GLMB': {
            'class': Algo_Pure_GLMB,
            'args': [],
            'use_rf': False,
            'source': 'Final'
        },
        '2_Pure_Transformer': {
            'class': Algo_Trans_Final,
            'args': [model, prep, device, config.MAX_RANGE],
            'use_rf': True,
            'source': 'Final'
        },
        '3_Hybrid_GLMB': {
            'class': Algo_Hybrid_GLMB,
            'args': [model, prep, device, config.MAX_RANGE],
            'use_rf': True,
            'source': 'Final'
        },
        '4_GNN_MN': {
            'class': Algo_GNN_MN,
            'args': [],
            'use_rf': False,
            'source': 'Evaluation'
        },
        '5_Hybrid_GNN_MN': {
            'class': Algo_Hybrid_GNN_MN,
            'args': [model, prep, device, config.MAX_RANGE],
            'use_rf': True,
            'source': 'Evaluation'
        }
    }

    # 4. 运行算法
    all_results = {}
    all_metrics = []

    print("\n" + "=" * 60)
    print("Running Algorithms...")
    print("=" * 60)

    for name, algo_info in algorithms.items():
        print(f"\n--- {name} (from {algo_info['source']}) ---")

        # 实例化算法 (修正语法错误)
        algo_class = algo_info['class']
        algo_args = algo_info['args']
        algo = algo_class(*algo_args)

        # 运行
        if algo_info['use_rf']:
            if algo_info['source'] == 'Final':
                results, latency = algo.run(data['radar_in'], data['rf_in'], config.TOTAL_FRAMES)
                results_dict = convert_array_to_dict(results)
            else:
                results_dict, latency = algo.run(data['radar_in'], data['rf_in'], config.TOTAL_FRAMES)
        else:
            if algo_info['source'] == 'Final':
                results, latency = algo.run(data['radar_in'], config.TOTAL_FRAMES)
                results_dict = convert_array_to_dict(results)
            else:
                results_dict, latency = algo.run(data['radar_in'], config.TOTAL_FRAMES)

        # 评估
        metrics = evaluate_all(data['gt_list'], results_dict, latency, name, config)
        all_results[name] = {
            'results': results_dict,
            'latency': latency,
            'metrics': metrics
        }
        all_metrics.append({k: v for k, v in metrics.items()
                            if k not in ['ospa_curve', 'card_gt', 'card_est']})

        print(f"  MOTA: {metrics['MOTA']:.3f}, OSPA: {metrics['Avg_OSPA']:.1f}m, "
              f"Precision: {metrics['Precision']:.3f}, Recall: {metrics['Recall']:.3f}, "
              f"Latency: {metrics['Latency_ms']:.1f}ms")

    return all_results, all_metrics, data


# ==========================================
# 可视化
# ==========================================
def plot_results(all_results, all_metrics, data, config, save_path="Comparison_Result.png"):
    """绘制对比结果"""

    fig = plt.figure(figsize=(18, 14))

    colors = {
        '1_Pure_GLMB': 'blue',
        '2_Pure_Transformer': 'green',
        '3_Hybrid_GLMB': 'orange',
        '4_GNN_MN': 'cyan',
        '5_Hybrid_GNN_MN': 'red'
    }

    # 1. OSPA时间曲线
    ax1 = fig.add_subplot(2, 3, 1)
    for name, res in all_results.items():
        ospa_curve = res['metrics']['ospa_curve']
        ax1.plot(ospa_curve, color=colors.get(name, 'gray'), alpha=0.7,
                 label=name.split('_', 1)[1], linewidth=1)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('OSPA (m)')
    ax1.set_title('OSPA Distance Over Time')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 势估计
    ax2 = fig.add_subplot(2, 3, 2)
    card_gt = all_results[list(all_results.keys())[0]]['metrics']['card_gt']
    ax2.plot(card_gt, 'k-', linewidth=2, label='Ground Truth')
    for name, res in all_results.items():
        card_est = res['metrics']['card_est']
        ax2.plot(card_est, color=colors.get(name, 'gray'), alpha=0.7,
                 label=name.split('_', 1)[1], linewidth=1)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Number of Tracks')
    ax2.set_title('Cardinality Estimation')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. 指标柱状图
    ax3 = fig.add_subplot(2, 3, 3)
    df = pd.DataFrame(all_metrics)
    x = np.arange(len(df))
    width = 0.2
    metrics_to_plot = ['MOTA', 'Precision', 'Recall']
    for i, metric in enumerate(metrics_to_plot):
        ax3.bar(x + i * width, df[metric], width, label=metric)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([n.split('_', 1)[1] for n in df['Name']], rotation=45, ha='right')
    ax3.set_ylabel('Score')
    ax3.set_title('MOTA / Precision / Recall')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)

    # 4. OSPA和延迟柱状图
    ax4 = fig.add_subplot(2, 3, 4)
    ax4_twin = ax4.twinx()
    x = np.arange(len(df))
    ax4.bar(x - 0.2, df['Avg_OSPA'], 0.4, label='OSPA', color='steelblue')
    ax4_twin.bar(x + 0.2, df['Latency_ms'], 0.4, label='Latency', color='coral')
    ax4.set_xticks(x)
    ax4.set_xticklabels([n.split('_', 1)[1] for n in df['Name']], rotation=45, ha='right')
    ax4.set_ylabel('OSPA (m)', color='steelblue')
    ax4_twin.set_ylabel('Latency (ms)', color='coral')
    ax4.set_title('OSPA and Latency')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')

    # 5. 3D轨迹
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    for i, obs in enumerate(data['observations']):
        path, st = obs[0], int(obs[1])
        valid_idx = np.where((np.arange(len(path)) + st) < config.TOTAL_FRAMES)[0]
        if len(valid_idx) > 1:
            p_dat = path[valid_idx]
            p, a, r = np.deg2rad(p_dat[:, 0]), np.deg2rad(p_dat[:, 1]), p_dat[:, 2]
            x = r * np.cos(p) * np.cos(a)
            y = r * np.cos(p) * np.sin(a)
            z = r * np.sin(p)
            ax5.plot(x, y, z, 'k-', linewidth=1.5, alpha=0.7, label='GT' if i == 0 else "")

    best_algo = min(all_results.keys(), key=lambda k: all_results[k]['metrics']['Avg_OSPA'])
    best_res = all_results[best_algo]['results']
    all_x, all_y, all_z = [], [], []
    for t in range(0, config.TOTAL_FRAMES, 2):
        pts = best_res[t]['pos']
        if len(pts) > 0:
            all_x.extend(pts[:, 0])
            all_y.extend(pts[:, 1])
            all_z.extend(pts[:, 2])
    ax5.scatter(all_x, all_y, all_z, c='red', s=2, alpha=0.5,
                label=f'Best: {best_algo.split("_", 1)[1]}')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_zlabel('Z (m)')
    ax5.set_title('3D Trajectory (GT vs Best)')
    ax5.legend()

    # 6. 综合评分
    ax6 = fig.add_subplot(2, 3, 6)
    composite = []
    for _, row in df.iterrows():
        score = (0.3 * max(0, row['MOTA']) +
                 0.3 * (1 - row['Avg_OSPA'] / 100) +
                 0.2 * row['F1'] +
                 0.2 * (1 - min(row['Latency_ms'], 100) / 100))
        composite.append(score)

    bars = ax6.barh([n.split('_', 1)[1] for n in df['Name']], composite,
                    color=[colors.get(n, 'gray') for n in df['Name']])
    ax6.set_xlabel('Composite Score')
    ax6.set_title('Overall Performance Score')
    ax6.set_xlim([0, 1])
    ax6.grid(True, axis='x', alpha=0.3)
    for bar, score in zip(bars, composite):
        ax6.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {save_path}")
    plt.show()


def save_mat_data(all_results, all_metrics, data, config, save_path="comparison_data.mat"):
    """将绘图所需数据保存为 .mat 文件，供 MATLAB 读取绘图"""

    df = pd.DataFrame(all_metrics)
    algo_names_full = df['Name'].tolist()
    algo_names_short = [n.split('_', 1)[1] for n in algo_names_full]
    n_algos = len(algo_names_short)
    n_frames = config.TOTAL_FRAMES

    # ---------- 子图1: OSPA 随帧数变化曲线 ----------
    ospa_curves = np.zeros((n_frames, n_algos))
    for i, name in enumerate(algo_names_full):
        curve = all_results[name]['metrics']['ospa_curve']
        ospa_curves[:len(curve), i] = curve

    # ---------- 子图2: 势估计（目标数量） ----------
    first_key = algo_names_full[0]
    card_gt = np.array(all_results[first_key]['metrics']['card_gt'], dtype=float).reshape(-1, 1)
    card_est = np.zeros((n_frames, n_algos))
    for i, name in enumerate(algo_names_full):
        card_est[:, i] = np.array(all_results[name]['metrics']['card_est'])

    # ---------- 子图3: MOTA / 精确率 / 召回率 ----------
    mota       = df['MOTA'].values.astype(float)
    precision_v = df['Precision'].values.astype(float)
    recall_v   = df['Recall'].values.astype(float)
    f1         = df['F1'].values.astype(float)

    # ---------- 子图4: OSPA 与延迟 ----------
    avg_ospa   = df['Avg_OSPA'].values.astype(float)
    latency_ms = df['Latency_ms'].values.astype(float)

    # ---------- 子图5: 三维轨迹 ----------
    gt_trajs_x, gt_trajs_y, gt_trajs_z = [], [], []
    for obs in data['observations']:
        path, st = obs[0], int(obs[1])
        valid_idx = np.where((np.arange(len(path)) + st) < n_frames)[0]
        if len(valid_idx) > 1:
            p_dat = path[valid_idx]
            p_ang = np.deg2rad(p_dat[:, 0])
            a_ang = np.deg2rad(p_dat[:, 1])
            r_val = p_dat[:, 2]
            gt_trajs_x.append(r_val * np.cos(p_ang) * np.cos(a_ang))
            gt_trajs_y.append(r_val * np.cos(p_ang) * np.sin(a_ang))
            gt_trajs_z.append(r_val * np.sin(p_ang))

    n_gt = len(gt_trajs_x)
    if n_gt > 0:
        max_len_gt = max(len(t) for t in gt_trajs_x)
        gt_x_mat = np.full((max_len_gt, n_gt), np.nan)
        gt_y_mat = np.full((max_len_gt, n_gt), np.nan)
        gt_z_mat = np.full((max_len_gt, n_gt), np.nan)
        for i, (xi, yi, zi) in enumerate(zip(gt_trajs_x, gt_trajs_y, gt_trajs_z)):
            gt_x_mat[:len(xi), i] = xi
            gt_y_mat[:len(yi), i] = yi
            gt_z_mat[:len(zi), i] = zi
    else:
        gt_x_mat = np.empty((0, 0))
        gt_y_mat = np.empty((0, 0))
        gt_z_mat = np.empty((0, 0))

    best_algo_name_full = min(all_results.keys(),
                              key=lambda k: all_results[k]['metrics']['Avg_OSPA'])
    best_res = all_results[best_algo_name_full]['results']
    best_x, best_y, best_z = [], [], []
    for t in range(0, n_frames, 2):
        pts = best_res[t]['pos']
        if len(pts) > 0:
            best_x.extend(pts[:, 0])
            best_y.extend(pts[:, 1])
            best_z.extend(pts[:, 2])

    best_pts_x = np.array(best_x, dtype=float) if best_x else np.empty((0,))
    best_pts_y = np.array(best_y, dtype=float) if best_y else np.empty((0,))
    best_pts_z = np.array(best_z, dtype=float) if best_z else np.empty((0,))
    best_algo_short = best_algo_name_full.split('_', 1)[1]
    best_algo_idx = algo_names_full.index(best_algo_name_full) + 1  # 1-indexed for MATLAB

    # ---------- 子图6: 综合性能评分 ----------
    composite = np.array([
        0.3 * max(0, row['MOTA']) +
        0.3 * (1 - row['Avg_OSPA'] / 100) +
        0.2 * row['F1'] +
        0.2 * (1 - min(row['Latency_ms'], 100) / 100)
        for _, row in df.iterrows()
    ], dtype=float)

    # ---------- 算法名称 cell 数组 ----------
    algo_names_cell = np.empty((1, n_algos), dtype=object)
    for i, name in enumerate(algo_names_short):
        algo_names_cell[0, i] = name

    mat_dict = {
        'algo_names':   algo_names_cell,
        'n_algos':      float(n_algos),
        'n_frames':     float(n_frames),
        'ospa_curves':  ospa_curves,
        'card_gt':      card_gt,
        'card_est':     card_est,
        'mota':         mota.reshape(-1, 1),
        'precision_v':  precision_v.reshape(-1, 1),
        'recall_v':     recall_v.reshape(-1, 1),
        'f1':           f1.reshape(-1, 1),
        'avg_ospa':     avg_ospa.reshape(-1, 1),
        'latency_ms':   latency_ms.reshape(-1, 1),
        'composite':    composite.reshape(-1, 1),
        'best_algo_idx': float(best_algo_idx),
        'best_algo_name': best_algo_short,
        'gt_x':         gt_x_mat,
        'gt_y':         gt_y_mat,
        'gt_z':         gt_z_mat,
        'n_gt_traj':    float(n_gt),
        'best_pts_x':   best_pts_x.reshape(-1, 1),
        'best_pts_y':   best_pts_y.reshape(-1, 1),
        'best_pts_z':   best_pts_z.reshape(-1, 1),
    }

    sio.savemat(save_path, mat_dict)
    print(f"\nSaved plot data to: {save_path}")


def print_summary(all_metrics):
    """打印汇总表格"""
    df = pd.DataFrame(all_metrics)
    df_display = df.copy()
    df_display['Name'] = df_display['Name'].apply(lambda x: x.split('_', 1)[1])

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(df_display.to_string(index=False))

    print("\n" + "-" * 40)
    print("Best by Metric:")
    print("-" * 40)

    best_mota = df.loc[df['MOTA'].idxmax()]
    best_ospa = df.loc[df['Avg_OSPA'].idxmin()]
    best_f1 = df.loc[df['F1'].idxmax()]
    best_lat = df.loc[df['Latency_ms'].idxmin()]

    print(f"  MOTA:    {best_mota['Name'].split('_', 1)[1]:20s} = {best_mota['MOTA']:.3f}")
    print(f"  OSPA:    {best_ospa['Name'].split('_', 1)[1]:20s} = {best_ospa['Avg_OSPA']:.1f}m")
    print(f"  F1:      {best_f1['Name'].split('_', 1)[1]:20s} = {best_f1['F1']:.3f}")
    print(f"  Latency: {best_lat['Name'].split('_', 1)[1]:20s} = {best_lat['Latency_ms']:.1f}ms")


# ==========================================
# 多场景测试
# ==========================================
def run_multi_scenario_test():
    """运行多个传感器参数场景的测试"""

    scenarios = [
        {
            'name': 'Low_Noise',
            # Radar: [r_min, r_max, angle_std(度), range_std(米), P_D]
            'radar': [1, 10000, 0.2, 3.5, 0.90],
            # RF: [r_max, angle_std(度), P_D]
            'rf': [10000, 2.5, 0.95],
            'clutter': 20
        },
        {
            'name': 'Medium_Noise',
            'radar': [1, 10000, 0.35, 5.5, 0.85],
            'rf': [10000, 3.5, 0.90],
            'clutter': 30
        },
        {
            'name': 'High_Noise',
            'radar': [1, 10000, 0.5, 7.5, 0.75],
            'rf': [10000, 4.5, 0.85],
            'clutter': 40
        },
    ]

    all_scenario_results = []

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario['name']}")
        print(
            f"  Radar: angle_std={scenario['radar'][2]}°, range_std={scenario['radar'][3]}m, P_D={scenario['radar'][4]}")
        print(f"  RF: angle_std={scenario['rf'][1]}°, P_D={scenario['rf'][2]}")
        print(f"  Clutter: {scenario['clutter']} pts/frame")
        print(f"{'=' * 60}")

        config = Config()
        config.TOTAL_FRAMES = 300
        config.RADAR_PARAMS = scenario['radar']
        config.RF_PARAMS = scenario['rf']
        config.CLUTTER_MEAN = scenario['clutter']

        results, metrics, _ = run_comparison(config)

        if metrics:
            for m in metrics:
                m['Scenario'] = scenario['name']
            all_scenario_results.extend(metrics)

    # 汇总
    if all_scenario_results:
        df = pd.DataFrame(all_scenario_results)
        print("\n" + "=" * 80)
        print("MULTI-SCENARIO SUMMARY")
        print("=" * 80)

        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            scenario_df = df[df['Scenario'] == scenario['name']]
            display_df = scenario_df[['Name', 'MOTA', 'Avg_OSPA', 'Precision', 'Recall', 'Latency_ms']].copy()
            display_df['Name'] = display_df['Name'].apply(lambda x: x.split('_', 1)[1])
            print(display_df.to_string(index=False))

        return df
    return None


# ==========================================
# 主程序
# ==========================================
def main():
    """主程序入口"""

    # 创建配置
    config = Config()

    # ============================================
    # 在这里方便地调整参数
    # ============================================
    config.TOTAL_FRAMES = 200
    config.NUM_TARGETS = 10
    config.CLUTTER_MEAN = 20

    # ============================================
    # 调整传感器误差 - 方法1: 直接设置
    # Radar: [r_min, r_max, angle_std(度), range_std(米), P_D]
    # ============================================
    config.RADAR_PARAMS = [1, 10000, 0.25, 3.5, 0.85]

    # RF: [r_max, angle_std(度), P_D]
    config.RF_PARAMS = [10000, 2.0, 0.95]

    # # ============================================
    # # 调整传感器误差 - 方法2: 使用便捷函数
    # # ============================================
    # config.RADAR_PARAMS, config.RF_PARAMS = create_sensor_params(
    #     radar_angle_std=0.55,    # 雷达角度误差 (度)
    #     radar_range_std=8.5,    # 雷达距离误差 (米)
    #     radar_pd=0.80,          # 雷达检测概率
    #     rf_angle_std=5.5,       # RF角度误差 (度)
    #     rf_pd=0.85              # RF检测概率
    # )

    # 运行比较
    results, metrics, data = run_comparison(config)

    if results is not None:
        print_summary(metrics)
        save_mat_data(results, metrics, data, config, "comparison_data.mat")

        df = pd.DataFrame(metrics)
        df.to_csv("Comparison_Metrics.csv", index=False)
        print(f"\nSaved metrics to: Comparison_Metrics.csv")


if __name__ == "__main__":
    main()

    # # 可选: 运行多场景测试 (取消注释即可运行)
    # run_multi_scenario_test()
