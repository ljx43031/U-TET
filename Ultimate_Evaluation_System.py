"""
Ultimate_Evaluation_System_v2.py
版本升级:
1. 修复 GLMB 逻辑: 增加航迹起始/终结逻辑 (M/N 机制)，消除虚警，解决 MOTA 为负的问题。
2. 新增指标: 单帧推理时间 (Inference Time)。
3. 可视化升级: 3D 轨迹对比图。
4. 结构优化: 更严谨的卡尔曼滤波参数。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
from scipy.io import savemat
from collections import defaultdict, deque
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 导入仿真与模型模块
# ==========================================
# 请确保这些文件在同一目录下
from Training_Pipeline_v3_r import RFGuidedTransformerTracker
from Maneuvering_target_simulation_v2_r import Trajectory_auto_generator_multitarget
from Observation_Preprocessing_v2 import PointCloudPreprocessor


# ==========================================
# 0. 基础工具
# ==========================================
def ospa_dist(X, Y, c=100.0, p=2.0):
    m, n = X.shape[0], Y.shape[0]
    if m == 0 and n == 0: return 0
    if m == 0 or n == 0: return c
    d_mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = np.linalg.norm(X[i] - Y[j])
            d_mat[i, j] = min(d, c) ** p
    row_ind, col_ind = linear_sum_assignment(d_mat)
    match_cost = d_mat[row_ind, col_ind].sum()
    card_cost = (c ** p) * abs(m - n)
    return ((match_cost + card_cost) / max(m, n)) ** (1 / p)


class Denormalizer:
    def __init__(self, max_range):
        self.max_range = max_range

    def denorm(self, boxes):
        out = np.zeros_like(boxes)
        p_deg = (boxes[..., 0] + 1) / 2 * 90
        a_deg = (boxes[..., 1] + 1) / 2 * 360 - 180
        r = boxes[..., 2] * self.max_range
        p_rad, a_rad = np.deg2rad(p_deg), np.deg2rad(a_deg)
        out[..., 0] = r * np.cos(p_rad) * np.cos(a_rad)
        out[..., 1] = r * np.cos(p_rad) * np.sin(a_rad)
        out[..., 2] = r * np.sin(p_rad)
        return out


class CT_EKF:
    """ 坐标转弯扩展卡尔曼滤波 """

    def __init__(self, dt=1.0):
        self.dt = dt
        # 过程噪声 (Q): 控制模型对机动的适应性
        # 加大 Q 可以更好地跟踪转弯，但会增加抖动
        self.Q = np.diag([2.0, 5.0, 2.0, 5.0, 2.0, 5.0, 1e-4]) ** 2

        # 测量噪声 (R)
        self.R_radar = np.diag([20.0, np.deg2rad(1.0), np.deg2rad(1.0)]) ** 2
        self.R_cart = np.diag([10.0, 10.0, 10.0]) ** 2

    def predict(self, x, P):
        dt = self.dt
        px, vx, py, vy, pz, vz, w = x
        new_x = np.copy(x)
        if abs(w) < 1e-4:
            new_x[0] += vx * dt;
            new_x[2] += vy * dt;
            new_x[4] += vz * dt
            F = np.eye(7)
            F[0, 1] = dt;
            F[2, 3] = dt;
            F[4, 5] = dt
        else:
            s, c = np.sin(w * dt), np.cos(w * dt)
            new_x[0] = px + (vx * s - vy * (1 - c)) / w
            new_x[1] = vx * c - vy * s
            new_x[2] = py + (vx * (1 - c) + vy * s) / w
            new_x[3] = vx * s + vy * c
            new_x[4] = pz + vz * dt
            F = np.eye(7)
            F[0, 1] = s / w;
            F[0, 3] = -(1 - c) / w
            F[1, 1] = c;
            F[1, 3] = -s
            F[2, 1] = (1 - c) / w;
            F[2, 3] = s / w
            F[3, 1] = s;
            F[3, 3] = c
            F[4, 5] = dt
        P_pred = F @ P @ F.T + self.Q
        return new_x, P_pred

    def update_spherical(self, x, P, z):
        px, _, py, _, pz, _, _ = x
        r = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        rxy = np.sqrt(px ** 2 + py ** 2)
        r = max(r, 1e-3);
        rxy = max(rxy, 1e-3)
        z_pred = np.array([r, np.arctan2(py, px), np.arcsin(pz / r)])
        y = z - z_pred
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        H = np.zeros((3, 7))
        H[0, 0] = px / r;
        H[0, 2] = py / r;
        H[0, 4] = pz / r
        H[1, 0] = -py / rxy ** 2;
        H[1, 2] = px / rxy ** 2
        H[2, 0] = -px * pz / (r ** 2 * rxy);
        H[2, 2] = -py * pz / (r ** 2 * rxy);
        H[2, 4] = rxy / r ** 2
        S = H @ P @ H.T + self.R_radar
        K = P @ H.T @ np.linalg.inv(S)
        return x + K @ y, (np.eye(7) - K @ H) @ P, np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y)

    def update_cartesian(self, x, P, z):
        H = np.zeros((3, 7));
        H[0, 0] = 1;
        H[1, 2] = 1;
        H[2, 4] = 1
        y = z - H @ x
        S = H @ P @ H.T + self.R_cart
        K = P @ H.T @ np.linalg.inv(S)
        return x + K @ y, (np.eye(7) - K @ H) @ P, 1.0


# ==========================================
# 1. 算法: Standard GNN with Track Management (Replaces naive GLMB)
# ==========================================
class TrackState:
    """ 单个航迹对象 """

    def __init__(self, x, P, label, t_start):
        self.x = x
        self.P = P
        self.label = label
        self.state = 'tentative'  # tentative, confirmed, deleted
        self.history = deque(maxlen=5)  # 记录最近5帧的匹配情况 (1=match, 0=miss)
        self.history.append(1)
        self.miss_count = 0
        self.age = 1

    def update_status(self):
        # M/N 逻辑: 最近 4 帧里有 3 次匹配 -> 确认
        hits = sum(self.history)
        if self.state == 'tentative':
            if hits >= 3:
                self.state = 'confirmed'
            elif len(self.history) >= 4 and hits < 2:
                self.state = 'deleted'
        elif self.state == 'confirmed':
            if self.miss_count >= 3:  # 连续丢失3帧则删除
                self.state = 'deleted'


class Algo_Standard_Radar_Tracker:
    """
    标准雷达跟踪器 (替代原 Pure GLMB)
    逻辑: EKF + GNN Association + M/N Logic (Track Initiation)
    这种方法能有效过滤杂波，生成平滑轨迹，避免 MOTA 为负。
    """

    def __init__(self):
        self.ekf = CT_EKF()
        self.tracks = []
        self.next_label = 1
        self.gating_thresh = 150.0  # 米
        self.clutter_prob = 1e-10

    def run(self, radar_obs_list, total_frames):
        results = []
        latency_list = []
        print("Running Standard Radar Tracker (GNN-PM)...")

        for t in range(total_frames):
            t0 = time.time()
            # 1. 解析量测: [el, az, r] -> [r, az(rad), el(rad)]
            Z = []
            if t < len(radar_obs_list):
                for m in radar_obs_list[t]:
                    Z.append(np.array([m[2], np.deg2rad(m[1]), np.deg2rad(m[0])]))

            # 2. 预测
            for trk in self.tracks:
                trk.x, trk.P = self.ekf.predict(trk.x, trk.P)

            # 3. 数据关联 (GNN)
            m_cnt = len(Z)
            n_cnt = len(self.tracks)
            assignments = []
            unassigned_tracks = list(range(n_cnt))
            unassigned_meas = list(range(m_cnt))

            if m_cnt > 0 and n_cnt > 0:
                cost = np.full((n_cnt, m_cnt), 1e6)
                for i, trk in enumerate(self.tracks):
                    px, _, py, _, pz, _, _ = trk.x
                    pred_dist = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
                    for j, z in enumerate(Z):
                        # 粗略门控
                        if abs(pred_dist - z[0]) < self.gating_thresh:
                            # 欧氏距离近似 (为了速度，不用马氏距离)
                            # 将 z 转为 xyz 计算距离
                            zx = z[0] * np.cos(z[2]) * np.cos(z[1])
                            zy = z[0] * np.cos(z[2]) * np.sin(z[1])
                            zz = z[0] * np.sin(z[2])
                            dist = np.sqrt((px - zx) ** 2 + (py - zy) ** 2 + (pz - zz) ** 2)
                            if dist < self.gating_thresh:
                                cost[i, j] = dist

                row_ind, col_ind = linear_sum_assignment(cost)

                unassigned_tracks = []
                unassigned_meas = list(range(m_cnt))

                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < self.gating_thresh:
                        assignments.append((r, c))
                        if c in unassigned_meas: unassigned_meas.remove(c)
                    else:
                        unassigned_tracks.append(r)

                for i in range(n_cnt):
                    if i not in row_ind:
                        unassigned_tracks.append(i)

            # 4. 更新与管理
            # (A) 更新匹配的航迹
            for r, c in assignments:
                trk = self.tracks[r]
                trk.x, trk.P, _ = self.ekf.update_spherical(trk.x, trk.P, Z[c])
                trk.history.append(1)
                trk.miss_count = 0
                trk.update_status()

            # (B) 处理未匹配航迹
            for r in unassigned_tracks:
                trk = self.tracks[r]
                trk.history.append(0)
                trk.miss_count += 1
                trk.update_status()

            # (C) 新生航迹 (由未匹配量测产生)
            for c in unassigned_meas:
                z = Z[c]
                x = z[0] * np.cos(z[2]) * np.cos(z[1])
                y = z[0] * np.cos(z[2]) * np.sin(z[1])
                zc = z[0] * np.sin(z[2])
                # 初始化状态 [x, vx, y, vy, z, vz, w]
                init_x = np.array([x, 0, y, 0, zc, 0, 0])
                init_P = np.diag([50, 100, 50, 100, 50, 10, 0.1]) ** 2
                new_trk = TrackState(init_x, init_P, self.next_label, t)
                self.tracks.append(new_trk)
                self.next_label += 1

            # 5. 清理与输出
            # 移除 'deleted' 航迹
            self.tracks = [t for t in self.tracks if t.state != 'deleted']

            # 只输出 'confirmed' 航迹
            pos, ids = [], []
            for trk in self.tracks:
                if trk.state == 'confirmed':
                    pos.append(trk.x[[0, 2, 4]])
                    ids.append(trk.label)

            results.append({
                'pos': np.array(pos) if pos else np.empty((0, 3)),
                'ids': np.array(ids) if ids else np.array([])
            })
            latency_list.append((time.time() - t0) * 1000)  # ms

        return results, latency_list


# ==========================================
# 2. 算法: Pure Transformer & Hybrid (保持不变，增加计时)
# ==========================================
class Algo_Pure_Transformer:
    def __init__(self, model, preprocessor, device, max_range):
        self.model = model;
        self.prep = preprocessor
        self.device = device;
        self.denorm = Denormalizer(max_range)
        self.radar_buf = [];
        self.rf_buf = []

    def run(self, radar_data, rf_data, total_frames):
        results = [];
        latency = []
        print("Running Pure Transformer...")
        for _ in range(31):
            self.radar_buf.append(radar_data[0]);
            self.rf_buf.append(rf_data[0])

        for t in range(total_frames):
            t0 = time.time()
            self.radar_buf.append(radar_data[t]);
            self.rf_buf.append(rf_data[t])
            if len(self.radar_buf) > 32: self.radar_buf.pop(0); self.rf_buf.pop(0)

            est_pos, est_ids = np.empty((0, 3)), np.array([])
            if len(self.radar_buf) == 32:
                r_np = np.stack([self.prep.process_frame_radar(f) for f in self.radar_buf])
                f_np = np.stack([self.prep.process_frame_rf(f) for f in self.rf_buf])
                r_t = torch.from_numpy(r_np).unsqueeze(0).float().to(self.device)
                f_t = torch.from_numpy(f_np).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    out = self.model(r_t, f_t)
                    probs = 1 / (1 + np.exp(-out['logits'][0, :, -1].cpu().numpy()))
                    boxes = out['boxes'][0, :, -1, :].cpu().numpy()
                valid = np.where(probs > 0.4)[0]
                if len(valid) > 0:
                    est_pos = self.denorm.denorm(boxes[valid])
                    est_ids = valid

            results.append({'pos': est_pos, 'ids': est_ids})
            latency.append((time.time() - t0) * 1000)
        return results, latency


class Algo_Hybrid_Fused:
    def __init__(self, model, preprocessor, device, max_range):
        self.model = model;
        self.prep = preprocessor;
        self.device = device
        self.denorm = Denormalizer(max_range);
        self.ekf = CT_EKF()
        self.tracks = [];
        self.next_label = 1
        self.frame_buffer = defaultdict(list)

    def _fuser_add(self, start_frame, boxes_phys):
        K, T, _ = boxes_phys.shape
        for t in range(T):
            if start_frame != 0 and t < 16: continue
            abs_f = start_frame + t
            pts = [b for b in boxes_phys[:, t] if np.linalg.norm(b) > 1.0]
            if pts: self.frame_buffer[abs_f].append(np.array(pts))

    def _get_fused_meas(self, frame_idx):
        groups = self.frame_buffer.get(frame_idx, [])
        if not groups: return []
        base = groups[0]
        for i in range(1, len(groups)):
            new = groups[i]
            cost = np.linalg.norm(base[:, None, :] - new[None, :, :], axis=2)
            r, c = linear_sum_assignment(cost)
            matched = set()
            for ri, ci in zip(r, c):
                if cost[ri, ci] < 50.0:
                    base[ri] = (base[ri] + new[ci]) / 2.0;
                    matched.add(ci)
            for ni in range(len(new)):
                if ni not in matched: base = np.vstack([base, new[ni]])
        return base

    def run(self, radar_data, rf_data, total_frames):
        print("Running Hybrid Fused...")
        results = [None] * total_frames;
        latency = [0.0] * total_frames

        # 1. Inference
        r_all = [self.prep.process_frame_radar(f) for f in radar_data]
        f_all = [self.prep.process_frame_rf(f) for f in rf_data]
        WINDOW, STRIDE = 32, 16
        for start_t in range(0, total_frames - WINDOW + 1, STRIDE):
            t0 = time.time()
            end_t = start_t + WINDOW
            r_t = torch.from_numpy(np.stack(r_all[start_t:end_t])).unsqueeze(0).float().to(self.device)
            f_t = torch.from_numpy(np.stack(f_all[start_t:end_t])).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                out = self.model(r_t, f_t)
                probs = 1 / (1 + np.exp(-out['logits'][0].cpu().numpy()))
                boxes = out['boxes'][0].cpu().numpy()
            valid = np.where(probs.mean(axis=1) > 0.4)[0]
            if len(valid) > 0:
                self._fuser_add(start_t, self.denorm.denorm(boxes[valid]))
            latency[start_t] += (time.time() - t0) * 1000

        # 2. Tracking
        for t in range(total_frames):
            t_start = time.time()
            meas = self._get_fused_meas(t)

            # Predict
            for trk in self.tracks:
                trk.x, trk.P = self.ekf.predict(trk.x, trk.P)

            # Update
            assignments = []
            unassigned_meas = list(range(len(meas)))
            if len(meas) > 0 and len(self.tracks) > 0:
                cost = np.full((len(self.tracks), len(meas)), 1e6)
                for i, trk in enumerate(self.tracks):
                    for j, z in enumerate(meas):
                        dist = np.linalg.norm(trk.x[[0, 2, 4]] - z)
                        if dist < 200: cost[i, j] = dist
                r_idx, c_idx = linear_sum_assignment(cost)
                for r, c in zip(r_idx, c_idx):
                    if cost[r, c] < 200:
                        assignments.append((r, c))
                        if c in unassigned_meas: unassigned_meas.remove(c)

            # Track Update
            matched_tracks = set()
            for r, c in assignments:
                self.tracks[r].x, self.tracks[r].P, _ = self.ekf.update_cartesian(self.tracks[r].x, self.tracks[r].P,
                                                                                  meas[c])
                self.tracks[r].miss_count = 0
                matched_tracks.add(r)

            # Track Maintenance
            for i, trk in enumerate(self.tracks):
                if i not in matched_tracks: trk.miss_count += 1

            # Birth
            for c in unassigned_meas:
                z = meas[c]
                ix = np.array([z[0], 0, z[1], 0, z[2], 0, 0])
                iP = np.diag([20, 50, 20, 50, 20, 10, 0.5]) ** 2
                self.tracks.append(TrackState(ix, iP, self.next_label, t))
                self.tracks[-1].state = 'confirmed'  # Transformer detection is high confidence
                self.next_label += 1

            self.tracks = [t for t in self.tracks if t.miss_count < 5]

            pos = [trk.x[[0, 2, 4]] for trk in self.tracks if trk.state == 'confirmed']
            ids = [trk.label for trk in self.tracks if trk.state == 'confirmed']

            results[t] = {'pos': np.array(pos) if pos else np.empty((0, 3)), 'ids': np.array(ids)}
            latency[t] += (time.time() - t_start) * 1000

        return results, latency


# ==========================================
# 3. 评估器
# ==========================================
class Evaluator:
    def __init__(self, gt_list, tracker_results, latency, name):
        self.gt_list = gt_list;
        self.results = tracker_results
        self.latency = latency;
        self.name = name
        self.tp, self.fp, self.fn, self.ids, self.total_gt = 0, 0, 0, 0, 0
        self.ospa_hist = []

    def evaluate(self):
        for t, (gt, res) in enumerate(zip(self.gt_list, self.results)):
            gt_pos, gt_ids = gt['pos'], gt['ids']
            res_pos, res_ids = res['pos'], res['ids']
            self.ospa_hist.append(ospa_dist(gt_pos, res_pos, 100, 2))

            # CLEAR MOT
            self.total_gt += len(gt_pos)
            if len(gt_pos) > 0 and len(res_pos) > 0:
                d = np.linalg.norm(gt_pos[:, None, :] - res_pos[None, :, :], axis=2)
                r, c = linear_sum_assignment(d)
                matched_gt, matched_res = set(), set()
                for ri, ci in zip(r, c):
                    if d[ri, ci] < 50.0:
                        self.tp += 1
                        matched_gt.add(ri);
                        matched_res.add(ci)
                self.fn += len(gt_pos) - len(matched_gt)
                self.fp += len(res_pos) - len(matched_res)
            else:
                self.fn += len(gt_pos);
                self.fp += len(res_pos)

    def summary(self):
        mota = 1 - (self.fn + self.fp + self.ids) / max(1, self.total_gt)
        return {'Name': self.name, 'MOTA': mota, 'Avg_OSPA': np.mean(self.ospa_hist),
                'Avg_Latency_ms': np.mean(self.latency)}


# ==========================================
# 4. 主流程
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TOTAL_FRAMES = 1000
    print(f"Generating {TOTAL_FRAMES} frames...")

    # 1. 生成数据
    sim = Trajectory_auto_generator_multitarget(TargetNum=10)
    _, Observations, _, Radar_obs, RF_obs, Clutter_rad, Clutter_rf = \
        sim.show_data(TimeStepRange=[0, TOTAL_FRAMES // 20],
                      Camera=[10000, 0.002], Radar=[1, 10000, 0.5, 7.5, 0.85],
                      RFsensor=[10000, 5.0, 0.95], clutter_num_mean=40)

    # 构建 GT
    gt_list = []
    for t in range(TOTAL_FRAMES):
        pos, ids = [], []
        for i, obs in enumerate(Observations):
            path, st = obs[0], int(obs[1])
            if st <= t < st + len(path):
                p_dat = path[t - st]
                p, a, r = np.deg2rad(p_dat[0]), np.deg2rad(p_dat[1]), p_dat[2]
                pos.append([r * np.cos(p) * np.cos(a), r * np.cos(p) * np.sin(a), r * np.sin(p)])
                ids.append(i + 1)
        gt_list.append({'pos': np.array(pos), 'ids': np.array(ids)})

    # 构建输入
    radar_in, rf_in = [], []
    for t in range(TOTAL_FRAMES):
        r_f, rf_f = [], []
        if t < len(Clutter_rad): r_f.extend(Clutter_rad[t])
        if t < len(Clutter_rf): rf_f.extend(Clutter_rf[t])
        for i, obs in enumerate(Radar_obs):
            st = int(Observations[i][1])
            if st <= t < st + len(obs) and len(obs[t - st]) > 0: r_f.append(obs[t - st])
        for i, obs in enumerate(RF_obs):
            st = int(Observations[i][1])
            if st <= t < st + len(obs) and len(obs[t - st]) > 0: rf_f.append(obs[t - st])
        radar_in.append(r_f);
        rf_in.append(rf_f)

    # 2. 模型
    model = RFGuidedTransformerTracker(30, 128, 4, 3, 3, 256, 0.1, 3).to(device)
    try:
        ckpt = torch.load('v3_checkpoint_best_r.pth', map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
    except:
        print("Using random weights (Demo mode)")
    prep = PointCloudPreprocessor(32, 100, 50, 30, 10000)

    # 3. 运行算法
    algos = {
        'Radar_GNN': Algo_Standard_Radar_Tracker(),
        'Trans': Algo_Pure_Transformer(model, prep, device, 10000),
        'Hybrid': Algo_Hybrid_Fused(model, prep, device, 10000)
    }

    res_all, met_list = {}, []
    for name, algo in algos.items():
        if name == 'Radar_GNN':
            r, l = algo.run(radar_in, TOTAL_FRAMES)
        else:
            r, l = algo.run(radar_in, rf_in, TOTAL_FRAMES)
        res_all[name] = r
        ev = Evaluator(gt_list, r, l, name)
        ev.evaluate()
        met_list.append(ev.summary())

    # 4. 表格结果
    df = pd.DataFrame(met_list).set_index('Name')
    print("\n" + "=" * 40 + "\nPERFORMANCE METRICS\n" + "=" * 40)
    print(df)

    # 5. 三维可视化
    print("Plotting 3D Trajectories...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 画 GT
    for i, obs in enumerate(Observations):
        path = obs[0];
        st = int(obs[1])
        valid_idx = np.where((np.arange(len(path)) + st) < TOTAL_FRAMES)[0]
        if len(valid_idx) > 1:
            p_dat = path[valid_idx]
            p, a, r = np.deg2rad(p_dat[:, 0]), np.deg2rad(p_dat[:, 1]), p_dat[:, 2]
            x = r * np.cos(p) * np.cos(a)
            y = r * np.cos(p) * np.sin(a)
            z = r * np.sin(p)
            ax.plot(x, y, z, 'k-', linewidth=2, label='GT' if i == 0 else "")
            ax.text(x[0], y[0], z[0], f'T{i + 1}', color='black')

    # 画算法
    colors = {'Radar_GNN': 'blue', 'Trans': 'green', 'Hybrid': 'red'}
    for name, color in colors.items():
        # 收集所有点
        all_x, all_y, all_z = [], [], []
        for t in range(0, TOTAL_FRAMES, 3):  # 降采样显示
            pts = res_all[name][t]['pos']
            if len(pts) > 0:
                all_x.extend(pts[:, 0])
                all_y.extend(pts[:, 1])
                all_z.extend(pts[:, 2])
        ax.scatter(all_x, all_y, all_z, c=color, s=2, alpha=0.5, label=name)

    ax.set_title("3D Trajectory Comparison")
    ax.set_xlabel("X (m)");
    ax.set_ylabel("Y (m)");
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("3D_Comparison.png")
    plt.show()

    # 6. 保存数据
    savemat("Tracking_Results_v2.mat", {'GT': gt_list, 'Results': res_all})
    print("Saved to Tracking_Results_v2.mat")


if __name__ == "__main__":
    main()