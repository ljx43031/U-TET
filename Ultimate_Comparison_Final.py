"""
Ultimate_Comparison_Final.py
功能:
1. 对比 Pure GLMB, Pure Transformer, Hybrid (Trans+GLMB) 三种算法。
2. 逻辑确认:
   - Pure GLMB: 逐帧 EKF-CT (球坐标更新)。
   - Pure Trans: Stride=1 滑动窗口 (单帧高精度)。
   - Hybrid: Stride=16 稀疏窗口 + 融合 + GLMB平滑 (笛卡尔坐标更新)。
3. 包含尾部帧处理逻辑，确保全时序覆盖。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import time

# 导入模块
from Training_Pipeline_v3_r import RFGuidedTransformerTracker
from Maneuvering_target_simulation_v2_r import Trajectory_auto_generator_multitarget, Observation_Lables
from Observation_Preprocessing_v2 import PointCloudPreprocessor


# ==========================================
# 0. 通用工具
# ==========================================
def ospa_dist(X, Y, c, p):
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
    card_cost = c ** p * abs(m - n)
    return (1 / max(m, n) * (match_cost + card_cost)) ** (1 / p)


class Denormalizer:
    def __init__(self, max_range):
        self.max_range = max_range

    def denorm(self, boxes):
        # boxes: [K, T, 3] normalized
        out = np.zeros_like(boxes)
        # Pitch: [-1, 1] -> [0, 90]
        p_deg = (boxes[..., 0] + 1) / 2 * 90
        # Azimuth: [-1, 1] -> [-180, 180]
        a_deg = (boxes[..., 1] + 1) / 2 * 360 - 180
        # Range: [0, 1] -> [0, max]
        r = boxes[..., 2] * self.max_range

        # Spherical -> Cartesian
        p_rad = np.deg2rad(p_deg)
        a_rad = np.deg2rad(a_deg)

        out[..., 0] = r * np.cos(p_rad) * np.cos(a_rad)
        out[..., 1] = r * np.cos(p_rad) * np.sin(a_rad)
        out[..., 2] = r * np.sin(p_rad)
        return out


# ==========================================
# 1. CT-EKF
# ==========================================
class CT_EKF:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.Q = np.diag([0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 1e-4]) ** 2
        # Pure GLMB 用 (雷达原始精度)
        self.R_radar = np.diag([10.0, np.deg2rad(0.5), np.deg2rad(0.5)]) ** 2
        # Hybrid 用 (Transformer 精度)
        self.R_cart = np.diag([5.0, 5.0, 5.0]) ** 2

    def predict(self, x, P):
        dt = self.dt
        px, vx, py, vy, pz, vz, w = x
        new_x = np.copy(x)
        if abs(w) < 1e-4:
            new_x[0] += vx * dt;
            new_x[2] += vy * dt;
            new_x[4] += vz * dt
            F = np.eye(7);
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
            F = np.eye(7);
            F[0, 1] = s / w;
            F[0, 3] = -(1 - c) / w;
            F[1, 1] = c;
            F[1, 3] = -s;
            F[2, 1] = (1 - c) / w;
            F[2, 3] = s / w;
            F[3, 1] = s;
            F[3, 3] = c;
            F[4, 5] = dt
        P_pred = F @ P @ F.T + self.Q
        return new_x, P_pred

    def update_spherical(self, x, P, z):
        # z: [r, az, el]
        px, _, py, _, pz, _, _ = x
        r = np.sqrt(px ** 2 + py ** 2 + pz ** 2);
        rxy = np.sqrt(px ** 2 + py ** 2)
        r = max(r, 1e-3);
        rxy = max(rxy, 1e-3)
        z_pred = np.array([r, np.arctan2(py, px), np.arcsin(pz / r)])
        y = z - z_pred
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi;
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
        # z: [x, y, z]
        H = np.zeros((3, 7));
        H[0, 0] = 1;
        H[1, 2] = 1;
        H[2, 4] = 1
        y = z - H @ x
        S = H @ P @ H.T + self.R_cart
        K = P @ H.T @ np.linalg.inv(S)
        det_S = np.linalg.det(S)
        like = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt((2 * np.pi) ** 3 * det_S + 1e-10)
        return x + K @ y, (np.eye(7) - K @ H) @ P, like


# ==========================================
# 2. 算法 1: 纯 GLMB (Algo_Pure_GLMB)
# ==========================================
class Algo_Pure_GLMB:
    def __init__(self):
        self.ekf = CT_EKF()
        self.tracks = []
        self.next_label = 1
        self.P_S = 0.99;
        self.P_D = 0.85;
        self.r_birth = 0.05
        self.Clutter_Intensity = 1e-5
        self.r_prune = 1e-3;
        self.r_recycle = 0.5

    def run(self, radar_obs_list, total_frames):
        results = [];
        latency = []
        print("Running Pure GLMB...")

        for t in range(total_frames):
            t0 = time.time()
            # [r, az, el]
            Z = []
            if t < len(radar_obs_list):
                for m in radar_obs_list[t]:
                    Z.append(np.array([m[2], np.deg2rad(m[1]), np.deg2rad(m[0])]))

            # Predict
            for trk in self.tracks:
                trk['r'] *= self.P_S
                trk['x'], trk['P'] = self.ekf.predict(trk['x'], trk['P'])

            # Update
            m_cnt = len(Z);
            n_cnt = len(self.tracks)
            if m_cnt > 0:
                likelihoods = np.zeros((n_cnt, m_cnt))
                for i, trk in enumerate(self.tracks):
                    for j, z in enumerate(Z):
                        px, _, py, _, pz, _, _ = trk['x']
                        # Gating on Range (Spherical domain)
                        if abs(np.sqrt(px ** 2 + py ** 2 + pz ** 2) - z[0]) < 150:
                            _, _, like = self.ekf.update_spherical(trk['x'], trk['P'], z)
                            likelihoods[i, j] = like

                cost = np.full((n_cnt, m_cnt), 100.0)
                for i in range(n_cnt):
                    for j in range(m_cnt):
                        if likelihoods[i, j] > 1e-20:
                            cost[i, j] = -np.log(likelihoods[i, j] * self.P_D / self.Clutter_Intensity)

                r_idx, c_idx = linear_sum_assignment(cost)
                assigned_meas = set();
                new_tracks = []

                for r, c in zip(r_idx, c_idx):
                    if cost[r, c] < 50.0:
                        trk = self.tracks[r]
                        trk['x'], trk['P'], like = self.ekf.update_spherical(trk['x'], trk['P'], Z[c])
                        ratio = (self.P_D * like) / self.Clutter_Intensity
                        trk['r'] = (trk['r'] * ratio) / (1 - trk['r'] + trk['r'] * ratio);
                        trk['r'] = min(trk['r'], 0.999)
                        new_tracks.append(trk);
                        assigned_meas.add(c)
                    else:
                        self.tracks[r]['r'] = (self.tracks[r]['r'] * (1 - self.P_D)) / (
                                    1 - self.tracks[r]['r'] * self.P_D)
                        new_tracks.append(self.tracks[r])
                for i in range(n_cnt):
                    if i not in r_idx:
                        self.tracks[i]['r'] = (self.tracks[i]['r'] * (1 - self.P_D)) / (
                                    1 - self.tracks[i]['r'] * self.P_D)
                        new_tracks.append(self.tracks[i])
                for j in range(m_cnt):
                    if j not in assigned_meas:
                        z = Z[j]
                        x = z[0] * np.cos(z[2]) * np.cos(z[1]);
                        y = z[0] * np.cos(z[2]) * np.sin(z[1]);
                        zc = z[0] * np.sin(z[2])
                        init_x = np.array([x, 0, y, 0, zc, 0, 0])
                        init_P = np.diag([50, 100, 50, 100, 50, 10, 0.5]) ** 2
                        new_tracks.append({'x': init_x, 'P': init_P, 'r': self.r_birth, 'label': self.next_label})
                        self.next_label += 1
                self.tracks = new_tracks
            else:
                for trk in self.tracks: trk['r'] = (trk['r'] * (1 - self.P_D)) / (1 - trk['r'] * self.P_D)

            self.tracks = [t for t in self.tracks if t['r'] > self.r_prune]

            est = []
            for trk in self.tracks:
                if trk['r'] > self.r_recycle: est.append(trk['x'][[0, 2, 4]])
            results.append(np.array(est) if len(est) > 0 else np.empty((0, 3)))
            latency.append((time.time() - t0) * 1000)

        return results, latency


# ==========================================
# 3. 算法 2: 纯 Transformer (Algo_Pure_Transformer)
# ==========================================
class Algo_Pure_Transformer:
    def __init__(self, model, preprocessor, device, max_range):
        self.model = model
        self.prep = preprocessor
        self.device = device
        self.max_range = max_range
        self.denormalizer = Denormalizer(max_range)
        self.radar_buf = [];
        self.rf_buf = []

    def run(self, radar_data, rf_data, total_frames):
        results = [np.empty((0, 3)) for _ in range(total_frames)]
        latency = []
        print("Running Pure Transformer (Stride=1)...")

        # Padding
        for _ in range(31):
            self.radar_buf.append(radar_data[0])
            self.rf_buf.append(rf_data[0])

        for t in range(total_frames):
            t0 = time.time()
            self.radar_buf.append(radar_data[t])
            self.rf_buf.append(rf_data[t])
            if len(self.radar_buf) > 32: self.radar_buf.pop(0); self.rf_buf.pop(0)

            if len(self.radar_buf) == 32:
                r_np = np.stack([self.prep.process_frame_radar(f) for f in self.radar_buf])
                f_np = np.stack([self.prep.process_frame_rf(f) for f in self.rf_buf])
                r_t = torch.from_numpy(r_np).unsqueeze(0).float().to(self.device)
                f_t = torch.from_numpy(f_np).unsqueeze(0).float().to(self.device)

                with torch.no_grad():
                    out = self.model(r_t, f_t)
                    boxes = out['boxes'][0, :, -1, :].cpu().numpy()  # Last frame
                    probs = 1 / (1 + np.exp(-out['logits'][0, :, -1].cpu().numpy()))

                valid = probs > 0.4
                if valid.sum() > 0:
                    phys = self.denormalizer.denorm(boxes[valid])
                    results[t] = phys

            latency.append((time.time() - t0) * 1000)
            if t % 100 == 0: print(f"  Trans frame {t}")

        return results, latency


# ==========================================
# 4. 算法 3: Hybrid v7 (Algo_Hybrid_v7)
# ==========================================
class Algo_Hybrid_Fused:
    def __init__(self, model, preprocessor, device, max_range):
        self.model = model;
        self.prep = preprocessor;
        self.device = device
        self.denormalizer = Denormalizer(max_range)
        self.ekf = CT_EKF()

        self.tracks = [];
        self.next_label = 1
        self.P_S = 0.99;
        self.P_D = 0.95;
        self.r_birth = 0.1
        self.Clutter = 1e-10;
        self.r_recycle = 0.5
        self.frame_buffer = defaultdict(list)

    def _fuser_add(self, start_frame, boxes_phys):
        K, T, _ = boxes_phys.shape
        for t in range(T):
            abs_f = start_frame + t
            pts = []
            for k in range(K):
                if np.linalg.norm(boxes_phys[k, t]) > 1.0: pts.append(boxes_phys[k, t])
            if pts: self.frame_buffer[abs_f].append(np.array(pts))

    def _fuser_get(self, frame_idx):
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

    def _glmb_step_cartesian(self, measurements):
        for trk in self.tracks:
            trk['r'] *= self.P_S
            trk['x'], trk['P'] = self.ekf.predict(trk['x'], trk['P'])

        m = len(measurements);
        n = len(self.tracks)
        if m > 0:
            cost = np.full((n, m), 100.0)
            likelihoods = np.zeros((n, m))
            for i, trk in enumerate(self.tracks):
                for j, z in enumerate(measurements):
                    if np.linalg.norm(trk['x'][[0, 2, 4]] - z) < 200:  # Cartesian gating
                        _, _, like = self.ekf.update_cartesian(trk['x'], trk['P'], z)
                        likelihoods[i, j] = like
                        cost[i, j] = -np.log(like * self.P_D / self.Clutter + 1e-20)

            r_idx, c_idx = linear_sum_assignment(cost)
            assigned = set();
            new_ts = []
            for r, c in zip(r_idx, c_idx):
                if cost[r, c] < 50.0:
                    trk = self.tracks[r]
                    trk['x'], trk['P'], like = self.ekf.update_cartesian(trk['x'], trk['P'], measurements[c])
                    ratio = (self.P_D * like) / self.Clutter
                    trk['r'] = (trk['r'] * ratio) / (1 - trk['r'] + trk['r'] * ratio);
                    trk['r'] = min(trk['r'], 0.999)
                    new_ts.append(trk);
                    assigned.add(c)
                else:
                    self.tracks[r]['r'] = (self.tracks[r]['r'] * (1 - self.P_D)) / (1 - self.tracks[r]['r'] * self.P_D)
                    new_ts.append(self.tracks[r])
            for i in range(n):
                if i not in r_idx:
                    self.tracks[i]['r'] = (self.tracks[i]['r'] * (1 - self.P_D)) / (1 - self.tracks[i]['r'] * self.P_D)
                    new_ts.append(self.tracks[i])
            for j in range(m):
                if j not in assigned:
                    z = measurements[j]
                    ix = np.array([z[0], 0, z[1], 0, z[2], 0, 0])
                    iP = np.diag([20, 50, 20, 50, 20, 10, 0.5]) ** 2
                    new_ts.append({'x': ix, 'P': iP, 'r': self.r_birth, 'label': self.next_label});
                    self.next_label += 1
            self.tracks = new_ts
        else:
            for trk in self.tracks: trk['r'] = (trk['r'] * (1 - self.P_D)) / (1 - trk['r'] * self.P_D)
        self.tracks = [t for t in self.tracks if t['r'] > 1e-3]

    def run(self, radar_data, rf_data, total_frames):
        results = [np.empty((0, 3)) for _ in range(total_frames)]
        latency = [0.0] * total_frames
        print("Running Hybrid v7 (Batch-Sequential)...")

        WINDOW = 32;
        STRIDE = 16;
        last_filtered = -1

        # Batch Preprocess
        r_all = [self.prep.process_frame_radar(f) for f in radar_data]
        f_all = [self.prep.process_frame_rf(f) for f in rf_data]

        for start_t in range(0, total_frames - WINDOW + 1, STRIDE):
            end_t = start_t + WINDOW
            t0 = time.time()

            # Inference
            r_t = torch.from_numpy(np.stack(r_all[start_t:end_t])).unsqueeze(0).float().to(self.device)
            f_t = torch.from_numpy(np.stack(f_all[start_t:end_t])).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                out = self.model(r_t, f_t)
                boxes = out['boxes'][0].cpu().numpy()
                probs = 1 / (1 + np.exp(-out['logits'][0].cpu().numpy()))

            valid = np.where(probs.mean(axis=1) > 0.4)[0]
            phys = self.denormalizer.denorm(boxes[valid])
            self._fuser_add(start_t, phys)

            t1 = time.time()
            latency[start_t] += (t1 - t0) * 1000

            # GLMB Step
            target_frame = start_t + STRIDE
            if start_t + STRIDE >= total_frames - WINDOW: target_frame = end_t

            for t in range(last_filtered + 1, target_frame):
                t_g0 = time.time()
                meas = self._fuser_get(t)
                self._glmb_step_cartesian(meas)

                est = []
                for trk in self.tracks:
                    if trk['r'] > self.r_recycle: est.append(trk['x'][[0, 2, 4]])
                results[t] = np.array(est) if est else np.empty((0, 3))
                latency[t] += (time.time() - t_g0) * 1000

            last_filtered = target_frame - 1

        # Tail Processing
        if last_filtered < total_frames - 1:
            for t in range(last_filtered + 1, total_frames):
                meas = self._fuser_get(t)
                self._glmb_step_cartesian(meas)
                est = []
                for trk in self.tracks:
                    if trk['r'] > self.r_recycle: est.append(trk['x'][[0, 2, 4]])
                results[t] = np.array(est) if est else np.empty((0, 3))

        return results, latency


# ==========================================
# 5. 主程序
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TOTAL_FRAMES = 1000
    print(f"Generating {TOTAL_FRAMES} frames...")
    simulator = Trajectory_auto_generator_multitarget(TargetNum=10)
    _, Observations, _, Radar_obs, RF_obs, Clutter_rad, Clutter_rf = \
        simulator.show_data(TimeStepRange=[0, TOTAL_FRAMES // 20],
                            Camera=[10000, 0.002], Radar=[1, 10000, 0.25, 3.5, 0.85],
                            RFsensor=[10000, 2.5, 0.95], clutter_num_mean=20)

    gt_states = []
    for t in range(TOTAL_FRAMES):
        cg = []
        for obs in Observations:
            path = obs[0];
            st = int(obs[1])
            if st <= t < st + len(path):
                p, a, r = path[t - st]
                p = np.deg2rad(p);
                a = np.deg2rad(a)
                x = r * np.cos(p) * np.cos(a);
                y = r * np.cos(p) * np.sin(a);
                z = r * np.sin(p)
                cg.append([x, y, z])
        gt_states.append(np.array(cg))

    radar_in, rf_in = [], []
    for t in range(TOTAL_FRAMES):
        cr = [];
        cf = []
        if t < len(Clutter_rad): cr.extend(Clutter_rad[t])
        if t < len(Clutter_rf): cf.extend(Clutter_rf[t])
        for i, obs in enumerate(Radar_obs):
            st = int(Observations[i][1])
            if st <= t < st + len(obs) and len(obs[t - st]) > 0: cr.append(obs[t - st])
        for i, obs in enumerate(RF_obs):
            st = int(Observations[i][1])
            if st <= t < st + len(obs) and len(obs[t - st]) > 0: cf.append(obs[t - st])
        radar_in.append(cr);
        rf_in.append(cf)

    # Models
    config = {'clip_length': 32, 'max_targets': 30, 'max_range': 10000,
              'max_radar_points': 100, 'max_rf_points': 50, 'd_model': 128,
              'nhead': 4, 'num_encoder_layers': 3, 'num_decoder_layers': 3,
              'dim_feedforward': 256, 'dropout': 0.1, 'temporal_window': 3}
    model = RFGuidedTransformerTracker(30, 128, 4, 3, 3, 256, 0.1, 3).to(device)
    ckpt = torch.load('v3_checkpoint_best_r.pth', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    prep = PointCloudPreprocessor(32, 100, 50, 30, 10000)

    # Algorithms
    algo1 = Algo_Pure_GLMB()
    algo2 = Algo_Pure_Transformer(model, prep, device, 10000)
    algo3 = Algo_Hybrid_Fused(model, prep, device, 10000)

    # Run
    r1, t1 = algo1.run(radar_in, TOTAL_FRAMES)
    r2, t2 = algo2.run(radar_in, rf_in, TOTAL_FRAMES)
    r3, t3 = algo3.run(radar_in, rf_in, TOTAL_FRAMES)

    # Metrics
    o1, o2, o3 = [], [], []
    for t in range(TOTAL_FRAMES):
        o1.append(ospa_dist(gt_states[t], r1[t], 100, 2))
        o2.append(ospa_dist(gt_states[t], r2[t], 100, 2))
        o3.append(ospa_dist(gt_states[t], r3[t], 100, 2))

    print(f"GLMB OSPA: {np.mean(o1):.2f}, Lat: {np.mean(t1):.2f}")
    print(f"Trans OSPA: {np.mean(o2):.2f}, Lat: {np.mean(t2):.2f}")
    print(f"Hybrid OSPA: {np.mean(o3):.2f}, Lat: {np.mean(t3):.2f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0];
    ax.plot(o1, label='GLMB');
    ax.plot(o2, label='Trans');
    ax.plot(o3, label='Hybrid');
    ax.legend();
    ax.set_title("OSPA")
    ax = axes[0, 1];
    ax.bar(['GLMB', 'Trans', 'Hybrid'], [np.mean(t1), np.mean(t2), np.mean(t3)]);
    ax.set_title("Latency")
    ax = axes[1, 0];
    for obs in Observations:
        path = obs[0];
        st = int(obs[1]);
        ts = np.arange(len(path)) + st;
        valid = ts < TOTAL_FRAMES
        p = np.deg2rad(path[:, 0]);
        a = np.deg2rad(path[:, 1]);
        r = path[:, 2]
        x = r * np.cos(p) * np.cos(a);
        y = r * np.cos(p) * np.sin(a);
        ax.plot(x, y, 'k-', alpha=0.3)
    hx, hy = [], []
    for res in r3:
        if len(res) > 0: hx.extend(res[:, 0]); hy.extend(res[:, 1])
    ax.scatter(hx, hy, s=1, c='r');
    ax.set_title("Hybrid Traj")

    plt.tight_layout()
    plt.savefig("Ultimate_Comparison_v4.png")
    plt.show()


if __name__ == "__main__":
    main()