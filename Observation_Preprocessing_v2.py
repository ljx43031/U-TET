"""
新版数据预处理 v2
核心改变：
1. Label 使用原始的无噪声 Observations，而不是从 Labels 推导
2. 输入保持点云格式，不做Grid量化
3. 输出是每个目标每帧的无噪声 (pitch, azimuth, range)
"""

import numpy as np
import scipy.io as scio
import random
import torch
from torch.utils.data import TensorDataset, random_split

from Maneuvering_target_simulation_v2_r import Trajectory_auto_generator_multitarget, Observation_Lables


class PointCloudPreprocessor:
    """
    点云预处理器

    输入格式 (来自仿真):
        Fuse_Radar_Data[t]: List of [pitch, azimuth, range] - 第t帧所有雷达点
        Fuse_RF_Data[t]: List of [pitch, azimuth] - 第t帧所有RF点
        Observations: List of [array[N,3], start_time] - 每个目标的无噪声轨迹

    输出格式:
        radar_points: [T, M_rad, 4] - (pitch, azimuth, range, is_valid)
        rf_points: [T, M_rf, 3] - (pitch, azimuth, is_valid)
        gt_observations: [K, T, 4] - (pitch, azimuth, range, is_valid) 无噪声真值
    """

    def __init__(self, time_steps=32, max_radar_points=100, max_rf_points=50,
                 max_targets=10, max_range=10000):
        self.T = time_steps
        self.M_rad = max_radar_points
        self.M_rf = max_rf_points
        self.K = max_targets
        self.max_range = max_range

        # 归一化参数
        self.pitch_range = (0, 90)
        self.azimuth_range = (-180, 180)

    def normalize_pitch(self, pitch):
        """归一化俯仰角到 [-1, 1]"""
        return 2 * (pitch - self.pitch_range[0]) / (self.pitch_range[1] - self.pitch_range[0]) - 1

    def normalize_azimuth(self, azimuth):
        """归一化方位角到 [-1, 1]"""
        return 2 * (azimuth - self.azimuth_range[0]) / (self.azimuth_range[1] - self.azimuth_range[0]) - 1

    def normalize_range(self, range_val):
        """归一化距离到 [0, 1]"""
        return range_val / self.max_range

    def denormalize_pitch(self, pitch_norm):
        """反归一化俯仰角"""
        return (pitch_norm + 1) / 2 * (self.pitch_range[1] - self.pitch_range[0]) + self.pitch_range[0]

    def denormalize_azimuth(self, azimuth_norm):
        """反归一化方位角"""
        return (azimuth_norm + 1) / 2 * (self.azimuth_range[1] - self.azimuth_range[0]) + self.azimuth_range[0]

    def denormalize_range(self, range_norm):
        """反归一化距离"""
        return range_norm * self.max_range

    def process_frame_radar(self, frame_data):
        """
        处理单帧雷达数据

        Args:
            frame_data: List of [pitch, azimuth, range] or []

        Returns:
            points: [M_rad, 4] - (pitch_norm, azimuth_norm, range_norm, is_valid)
        """
        points = np.zeros((self.M_rad, 4), dtype=np.float32)

        if frame_data is None:
            return points

        valid_idx = 0
        for obs in frame_data:
            if valid_idx >= self.M_rad:
                break
            if len(obs) == 0:
                continue

            pitch, azimuth, range_val = obs[0], obs[1], obs[2]

            points[valid_idx, 0] = self.normalize_pitch(pitch)
            points[valid_idx, 1] = self.normalize_azimuth(azimuth)
            points[valid_idx, 2] = self.normalize_range(range_val)
            points[valid_idx, 3] = 1.0  # is_valid
            valid_idx += 1

        return points

    def process_frame_rf(self, frame_data):
        """
        处理单帧RF数据

        Args:
            frame_data: List of [pitch, azimuth] or []

        Returns:
            points: [M_rf, 3] - (pitch_norm, azimuth_norm, is_valid)
        """
        points = np.zeros((self.M_rf, 3), dtype=np.float32)

        if frame_data is None:
            return points

        valid_idx = 0
        for obs in frame_data:
            if valid_idx >= self.M_rf:
                break
            if len(obs) == 0:
                continue

            pitch, azimuth = obs[0], obs[1]

            points[valid_idx, 0] = self.normalize_pitch(pitch)
            points[valid_idx, 1] = self.normalize_azimuth(azimuth)
            points[valid_idx, 2] = 1.0  # is_valid
            valid_idx += 1

        return points

    def extract_ground_truth_from_observations(self, observations, clip_start_frame, clip_length):
        """
        从原始 Observations 中提取无噪声真值

        Args:
            observations: List of [array[N,3], start_time] - 每个目标的完整无噪声轨迹
                         array[N,3] 的列是 [pitch(°), azimuth(°), distance(m)]
            clip_start_frame: 当前clip在全局时间轴上的起始帧
            clip_length: clip长度 (T)

        Returns:
            gt_obs: [K, T, 4] - 每个目标在每帧的 (pitch, azimuth, range, is_valid)
        """
        gt_obs = np.zeros((self.K, clip_length, 4), dtype=np.float32)

        # 统计实际目标数
        actual_targets = len(observations)
        if actual_targets > self.K:
            # 仅打印一次警告，避免刷屏
            pass  # 可以在外部统计

        # 记录有效填充的目标数
        filled_targets = 0

        # 遍历每个目标轨迹，最多处理 K 个
        for target_idx, obs_data in enumerate(observations):
            if target_idx >= self.K:
                break  # 截断到最大目标数

            # obs_data = [array[N,3], start_time]
            obs_array = obs_data[0]  # [N, 3]: pitch, azimuth, distance
            target_start_time = int(obs_data[1])  # 该目标轨迹的起始帧（相对时间）

            has_valid_frame = False

            # 计算该目标在当前clip中的有效范围
            for t in range(clip_length):
                global_frame = clip_start_frame + t

                # 计算在该目标轨迹数组中的索引
                obs_idx = global_frame - target_start_time

                # 检查是否在该目标的有效时间范围内
                if 0 <= obs_idx < len(obs_array):
                    pitch = obs_array[obs_idx, 0]  # 俯仰角 (°)
                    azimuth = obs_array[obs_idx, 1]  # 方位角 (°)
                    distance = obs_array[obs_idx, 2]  # 距离 (m)

                    gt_obs[target_idx, t, 0] = self.normalize_pitch(pitch)
                    gt_obs[target_idx, t, 1] = self.normalize_azimuth(azimuth)
                    gt_obs[target_idx, t, 2] = self.normalize_range(distance)
                    gt_obs[target_idx, t, 3] = 1.0  # is_valid
                    has_valid_frame = True

            if has_valid_frame:
                filled_targets += 1

        return gt_obs, actual_targets, filled_targets

    def process_sequence(self, fuse_radar_data, fuse_rf_data, observations, clip_start_frame):
        """
        处理一个时间序列片段

        Returns:
            radar_points: [T, M_rad, 4]
            rf_points: [T, M_rf, 3]
            gt_observations: [K, T, 4]
            stats: dict 包含统计信息
        """
        T = len(fuse_radar_data)

        radar_points = np.zeros((T, self.M_rad, 4), dtype=np.float32)
        rf_points = np.zeros((T, self.M_rf, 3), dtype=np.float32)

        for t in range(T):
            radar_points[t] = self.process_frame_radar(fuse_radar_data[t])
            rf_points[t] = self.process_frame_rf(fuse_rf_data[t])

        # 从无噪声 Observations 中提取真值
        gt_observations, actual_targets, filled_targets = self.extract_ground_truth_from_observations(
            observations, clip_start_frame, T
        )

        stats = {
            'actual_targets': actual_targets,
            'filled_targets': filled_targets,
            'truncated': actual_targets > self.K
        }

        return radar_points, rf_points, gt_observations, stats


class DataGenerator:
    """
    数据生成器：封装仿真和预处理流程
    """

    def __init__(self, config):
        self.config = config
        self.preprocessor = PointCloudPreprocessor(
            time_steps=config['clip_length'],
            max_radar_points=config['max_radar_points'],
            max_rf_points=config['max_rf_points'],
            max_targets=config['max_targets'],
            max_range=config['max_range']
        )

    def generate_simulation_data(self, seed=None):
        """运行一次完整的仿真"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        config = self.config

        # 初始化仿真器
        # 使用 avg_targets 作为仿真器的输入（平均目标数）
        avg_targets = config.get('avg_targets', config.get('max_targets', 10))
        simulator = Trajectory_auto_generator_multitarget(TargetNum=avg_targets)

        # 计算需要的仿真步数
        section_len = simulator.targets_list[0][0].SectionLen
        frames_needed = config['clip_length'] * config['clips_per_sim'] + config['clip_length']
        sim_steps = int(np.ceil(frames_needed / section_len))

        TSR = [0, sim_steps]
        C_ = [10000, 0.002]
        R_ = [1, config['max_range'], 0.25, 3.5, 0.8]
        RF = [10000, 2.5, 0.95]

        # 生成数据
        Trajectories, Observations, Camera_obs, Radar_obs, RF_obs, Clutter_rad, Clutter_rf = \
            simulator.show_data(TimeStepRange=TSR, Camera=C_, Radar=R_, RFsensor=RF,
                                clutter_num_mean=config['clutter_mean'])

        # 融合数据（添加杂波和标签）
        Fuse_Radar_Data, Fuse_Radar_Labels, Fuse_RF_Data, Fuse_RF_Labels = Observation_Lables(
            Observations, Radar_obs, RF_obs, Clutter_rad, Clutter_rf
        )

        return {
            'observations': Observations,  # 无噪声真值！
            'fuse_radar_data': Fuse_Radar_Data,
            'fuse_rf_data': Fuse_RF_Data,
            'fuse_radar_labels': Fuse_Radar_Labels,
            'fuse_rf_labels': Fuse_RF_Labels,
            'total_frames': len(Fuse_Radar_Data)
        }

    def extract_clips(self, sim_data, num_clips):
        """从仿真数据中提取多个clip"""
        total_frames = sim_data['total_frames']
        clip_length = self.config['clip_length']

        max_start = total_frames - clip_length
        if max_start <= 0:
            raise ValueError(f"仿真数据太短: {total_frames} frames < {clip_length} needed")

        # 选择起始点
        if max_start >= num_clips:
            start_indices = random.sample(range(max_start), num_clips)
        else:
            start_indices = [random.randint(0, max_start - 1) for _ in range(num_clips)]

        all_radar = []
        all_rf = []
        all_gt = []

        # 统计
        total_truncated = 0
        max_actual_targets = 0

        for start_t in start_indices:
            end_t = start_t + clip_length

            radar_pts, rf_pts, gt_obs, stats = self.preprocessor.process_sequence(
                fuse_radar_data=sim_data['fuse_radar_data'][start_t:end_t],
                fuse_rf_data=sim_data['fuse_rf_data'][start_t:end_t],
                observations=sim_data['observations'],
                clip_start_frame=start_t
            )

            all_radar.append(radar_pts)
            all_rf.append(rf_pts)
            all_gt.append(gt_obs)

            # 更新统计
            if stats['truncated']:
                total_truncated += 1
            max_actual_targets = max(max_actual_targets, stats['actual_targets'])

        # 如果有截断，打印警告
        if total_truncated > 0:
            print(
                f"  [警告] {total_truncated}/{num_clips} 个clip的目标数超过max_targets({self.config['max_targets']})被截断, 最大实际目标数: {max_actual_targets}")

        return {
            'radar': np.array(all_radar),
            'rf': np.array(all_rf),
            'gt': np.array(all_gt),
            'truncated_count': total_truncated,
            'max_actual_targets': max_actual_targets
        }


def compute_dataset_statistics(gt_data):
    """计算数据集统计信息"""
    # gt_data: [N, K, T, 4]
    valid_mask = gt_data[:, :, :, 3] > 0.5

    # 每个样本的有效目标数
    targets_per_sample = valid_mask.any(axis=2).sum(axis=1)  # [N]

    # 每个目标的平均有效帧数
    valid_frames_per_target = valid_mask.sum(axis=2)  # [N, K]

    print(f"  平均每样本有效目标数: {targets_per_sample.mean():.2f}")
    print(f"  目标平均有效帧数: {valid_frames_per_target[valid_frames_per_target > 0].mean():.2f}")
    print(f"  总有效观测点数: {valid_mask.sum()}")


# ============================================
# 主程序
# ============================================

if __name__ == '__main__':
    # ================= 配置 =================
    CONFIG = {
        'clip_length': 32,  # 每个样本的时间步数
        'avg_targets': 10,  # 仿真的平均目标数 (传给仿真器)
        'max_targets': 30,  # 输出的最大目标数 (截断用，建议比avg大50%)
        'max_radar_points': 100,  # 每帧最大雷达点数
        'max_rf_points': 50,  # 每帧最大RF点数
        'max_range': 10000,  # 最大距离 (m)
        'clutter_mean': 10,  # 平均杂波数
        'clips_per_sim': 5,  # 每次仿真提取的clip数
        'num_simulations': 20,  # 仿真次数
        'train_ratio': 0.7,
        'val_ratio': 0.15,
    }

    TOTAL_SAMPLES = CONFIG['clips_per_sim'] * CONFIG['num_simulations']

    print("=" * 60)
    print("Radar-RF 点云数据集生成器 v2")
    print("=" * 60)
    print(f"配置:")
    print(f"  Clip长度: {CONFIG['clip_length']} 帧")
    print(f"  最大目标数: {CONFIG['max_targets']}")
    print(f"  最大雷达点/帧: {CONFIG['max_radar_points']}")
    print(f"  最大RF点/帧: {CONFIG['max_rf_points']}")
    print(f"  总样本数: {TOTAL_SAMPLES}")
    print("=" * 60)

    # 初始化生成器
    generator = DataGenerator(CONFIG)

    all_radar_data = []
    all_rf_data = []
    all_gt_data = []

    # 生成数据
    print("\n开始生成数据...")
    total_truncated_clips = 0
    global_max_targets = 0

    for sim_idx in range(CONFIG['num_simulations']):
        print(f"  仿真 {sim_idx + 1}/{CONFIG['num_simulations']}...", end=" ")

        # 运行仿真
        sim_data = generator.generate_simulation_data(seed=sim_idx * 100)
        print(f"生成 {sim_data['total_frames']} 帧, {len(sim_data['observations'])} 条轨迹", end=" ")

        # 提取clips
        clips = generator.extract_clips(sim_data, CONFIG['clips_per_sim'])

        all_radar_data.append(clips['radar'])
        all_rf_data.append(clips['rf'])
        all_gt_data.append(clips['gt'])

        total_truncated_clips += clips['truncated_count']
        global_max_targets = max(global_max_targets, clips['max_actual_targets'])

        print(f"-> 提取 {len(clips['radar'])} 个clip")

    # 最终统计
    print(f"\n截断统计:")
    print(f"  被截断的clip数: {total_truncated_clips}/{TOTAL_SAMPLES}")
    print(f"  全局最大实际目标数: {global_max_targets}")
    if global_max_targets > CONFIG['max_targets']:
        print(f"  [建议] 考虑将 max_targets 增加到 {global_max_targets + 5}")

    # 合并数据
    print("\n合并数据...")
    radar_tensor = torch.from_numpy(np.concatenate(all_radar_data, axis=0)).float()
    rf_tensor = torch.from_numpy(np.concatenate(all_rf_data, axis=0)).float()
    gt_tensor = torch.from_numpy(np.concatenate(all_gt_data, axis=0)).float()

    print(f"\n数据形状:")
    print(f"  Radar: {radar_tensor.shape}")  # [N, T, M_rad, 4]
    print(f"  RF: {rf_tensor.shape}")  # [N, T, M_rf, 3]
    print(f"  GT: {gt_tensor.shape}")  # [N, K, T, 4]

    # 统计信息
    print("\n数据集统计:")
    compute_dataset_statistics(gt_tensor.numpy())

    # 划分数据集
    print("\n划分数据集...")
    total_cnt = len(radar_tensor)
    n_train = int(total_cnt * CONFIG['train_ratio'])
    n_val = int(total_cnt * CONFIG['val_ratio'])
    n_test = total_cnt - n_train - n_val

    # 随机打乱
    indices = torch.randperm(total_cnt, generator=torch.Generator().manual_seed(42))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # 保存 .mat 格式
    save_mat_path = 'radar_rf_pointcloud_v2.mat'
    print(f"\n保存到 {save_mat_path}...")
    scio.savemat(save_mat_path, {
        # 训练集
        'train_radar': radar_tensor[train_idx].numpy(),
        'train_rf': rf_tensor[train_idx].numpy(),
        'train_gt': gt_tensor[train_idx].numpy(),
        # 验证集
        'val_radar': radar_tensor[val_idx].numpy(),
        'val_rf': rf_tensor[val_idx].numpy(),
        'val_gt': gt_tensor[val_idx].numpy(),
        # 测试集
        'test_radar': radar_tensor[test_idx].numpy(),
        'test_rf': rf_tensor[test_idx].numpy(),
        'test_gt': gt_tensor[test_idx].numpy(),
        # 配置
        'config': {
            'T': CONFIG['clip_length'],
            'K': CONFIG['max_targets'],
            'M_rad': CONFIG['max_radar_points'],
            'M_rf': CONFIG['max_rf_points'],
            'R_max': CONFIG['max_range'],
            'pitch_range': [0, 90],
            'azimuth_range': [-180, 180]
        }
    }, do_compression=True)

    # # 保存 .pt 格式 (PyTorch)
    # save_pt_path = 'radar_rf_pointcloud_v2.pt'
    # print(f"保存到 {save_pt_path}...")
    # torch.save({
    #     'train': {
    #         'radar': radar_tensor[train_idx],
    #         'rf': rf_tensor[train_idx],
    #         'gt': gt_tensor[train_idx]
    #     },
    #     'val': {
    #         'radar': radar_tensor[val_idx],
    #         'rf': rf_tensor[val_idx],
    #         'gt': gt_tensor[val_idx]
    #     },
    #     'test': {
    #         'radar': radar_tensor[test_idx],
    #         'rf': rf_tensor[test_idx],
    #         'gt': gt_tensor[test_idx]
    #     },
    #     'config': CONFIG
    # }, save_pt_path)
    #
    # print("\n" + "=" * 60)
    # print("数据生成完成!")
    # print("=" * 60)

    # ================= 可视化验证 =================
    print("\n生成验证可视化...")

    import matplotlib.pyplot as plt

    # 随机选一个样本可视化
    sample_idx = random.randint(0, total_cnt - 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 提取数据
    sample_radar = radar_tensor[sample_idx].numpy()  # [T, M_rad, 4]
    sample_rf = rf_tensor[sample_idx].numpy()  # [T, M_rf, 3]
    sample_gt = gt_tensor[sample_idx].numpy()  # [K, T, 4]

    preprocessor = generator.preprocessor

    # 图1: 雷达观测 (方位角-俯仰角)
    ax = axes[0, 0]
    cmap = plt.cm.viridis
    for t in range(CONFIG['clip_length']):
        valid = sample_radar[t, :, 3] > 0.5
        if valid.sum() > 0:
            pitch = preprocessor.denormalize_pitch(sample_radar[t, valid, 0])
            azimuth = preprocessor.denormalize_azimuth(sample_radar[t, valid, 1])
            ax.scatter(azimuth, pitch, c=[cmap(t / CONFIG['clip_length'])],
                       s=20, alpha=0.6)
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('Pitch (°)')
    ax.set_title('Radar Observations (color = time)')
    ax.set_xlim([-180, 180])
    ax.set_ylim([0, 90])
    ax.grid(True, alpha=0.3)

    # 图2: RF观测
    ax = axes[0, 1]
    for t in range(CONFIG['clip_length']):
        valid = sample_rf[t, :, 2] > 0.5
        if valid.sum() > 0:
            pitch = preprocessor.denormalize_pitch(sample_rf[t, valid, 0])
            azimuth = preprocessor.denormalize_azimuth(sample_rf[t, valid, 1])
            ax.scatter(azimuth, pitch, c=[cmap(t / CONFIG['clip_length'])],
                       s=20, alpha=0.6, marker='^')
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('Pitch (°)')
    ax.set_title('RF Observations (color = time)')
    ax.set_xlim([-180, 180])
    ax.set_ylim([0, 90])
    ax.grid(True, alpha=0.3)

    # 图3: 无噪声真值轨迹
    ax = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, CONFIG['max_targets']))
    for k in range(CONFIG['max_targets']):
        valid = sample_gt[k, :, 3] > 0.5
        if valid.sum() > 0:
            pitch = preprocessor.denormalize_pitch(sample_gt[k, valid, 0])
            azimuth = preprocessor.denormalize_azimuth(sample_gt[k, valid, 1])
            ax.plot(azimuth, pitch, 'o-', c=colors[k], label=f'Target {k + 1}',
                    markersize=4, linewidth=1)
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('Pitch (°)')
    ax.set_title('Ground Truth Trajectories (noise-free)')
    ax.set_xlim([-180, 180])
    ax.set_ylim([0, 90])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 图4: 距离-时间图
    ax = axes[1, 1]
    for k in range(CONFIG['max_targets']):
        valid = sample_gt[k, :, 3] > 0.5
        if valid.sum() > 0:
            t_valid = np.where(valid)[0]
            distance = preprocessor.denormalize_range(sample_gt[k, valid, 2])
            ax.plot(t_valid, distance, 'o-', c=colors[k], label=f'Target {k + 1}',
                    markersize=4, linewidth=1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Target Distance over Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_visualization_v2.png', dpi=150)
    plt.show()

    print("可视化保存到 data_visualization_v2.png")
