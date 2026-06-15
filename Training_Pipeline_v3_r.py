"""
Training_Pipeline_v3.py
功能升级:
1.运动规律约束的损失设计
2.“倾向于从原来的输入中选择目标观测值”，这在深度学习中可以通过 Cross-Attention（交叉注意力） 来完美实现
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# 使用新版 API
from torch.amp import autocast, GradScaler
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings

# 抑制 Flash Attention 警告
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# 导入自定义模块 (需要确保文件名正确)
from Observation_Preprocessing_v2 import PointCloudPreprocessor, DataGenerator
from Maneuvering_target_simulation_v2_r import Trajectory_auto_generator_multitarget, Observation_Lables
# 导入模型定义 (或者你可以把类定义直接贴在下面，这里复用之前的结构)
# 假设我们把类定义都放在这个文件里，方便运行

# ==========================================
# 0. 辅助类与模型定义 (合并以便单文件运行)
# ==========================================
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t_steps):
        return self.pe[t_steps.long()]


class PointEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class RFGuidedTransformerTracker(nn.Module):
    def __init__(self, num_targets=20, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=256, dropout=0.1, temporal_window=3):
        """
        Args:
            temporal_window: 时间窗口半径，每帧可以attention到 [t-W, t+W] 范围的点
        """
        super().__init__()
        self.d_model = d_model
        self.num_targets = num_targets
        self.temporal_window = temporal_window  # [新增] 时间窗口参数

        self.radar_embed = PointEmbedding(4, d_model)
        self.rf_embed = PointEmbedding(3, d_model)
        self.time_pos_encoder = PositionalEncoding(d_model, max_len=128)

        self.rf_guide_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rf_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.st_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.st_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.target_queries = nn.Embedding(num_targets, d_model)

        # [新增] Input Selection 相关
        self.selection_proj_q = nn.Linear(d_model, d_model)
        self.selection_proj_k = nn.Linear(d_model, d_model)

        # [新增] 时间距离的可学习衰减因子
        # 距离当前帧越远，权重衰减越多
        self.time_decay = nn.Parameter(torch.tensor(0.5))  # 可学习的衰减率

        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)
        )
        self.prob_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def _create_temporal_attention_mask(self, T, M, device):
        """
        创建局部时间窗口的 Attention Mask

        对于 Query 在时刻 t，只能 attend 到 Key 在时刻 [t-W, t+W] 范围内的点

        Returns:
            mask: [T, T*M] 的 bool tensor，True 表示屏蔽
            time_dist: [T, T*M] 的 float tensor，表示时间距离（用于衰减）
        """
        W = self.temporal_window

        # Query 时刻索引: [T]
        query_times = torch.arange(T, device=device)

        # Key 时刻索引: [T*M]，每个时刻有 M 个点
        key_times = torch.arange(T, device=device).repeat_interleave(M)

        # 时间距离: [T, T*M]
        time_dist = torch.abs(query_times.unsqueeze(1) - key_times.unsqueeze(0)).float()

        # Mask: 距离 > W 的位置被屏蔽
        mask = time_dist > W

        return mask, time_dist

    def forward(self, radar_data, rf_data):
        B, T, M_rad, _ = radar_data.shape
        _, _, M_rf, _ = rf_data.shape
        device = radar_data.device

        # 1. Embedding
        radar_flat = radar_data.view(B, T * M_rad, -1)
        rf_flat = rf_data.view(B, T * M_rf, -1)

        # 保存原始坐标: [B, T*M, 3]
        raw_coords = radar_flat[:, :, :3]

        radar_padding_mask = (radar_flat[:, :, 3] < 0.5)  # [B, T*M]
        rf_padding_mask = (rf_flat[:, :, 2] < 0.5)

        radar_feat = self.radar_embed(radar_flat)
        rf_feat = self.rf_embed(rf_flat)

        # 2. 时序位置编码
        time_idx_rad = torch.arange(T, device=device).repeat_interleave(M_rad).unsqueeze(0).expand(B, -1)
        time_idx_rf = torch.arange(T, device=device).repeat_interleave(M_rf).unsqueeze(0).expand(B, -1)

        radar_feat = radar_feat + self.time_pos_encoder(time_idx_rad)
        rf_feat = rf_feat + self.time_pos_encoder(time_idx_rf)

        # 3. RF 引导与编码
        guided_feat, _ = self.rf_guide_attn(
            query=radar_feat, key=rf_feat, value=rf_feat,
            key_padding_mask=rf_padding_mask
        )
        radar_feat = self.rf_norm(radar_feat + guided_feat)
        encoded_feat = self.st_encoder(radar_feat, src_key_padding_mask=radar_padding_mask)

        # 4. 解码
        query_k = self.target_queries.weight.unsqueeze(0).expand(B, -1, -1)
        query_k = query_k.unsqueeze(2).repeat(1, 1, T, 1)
        t_pos_query = self.time_pos_encoder(
            torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        ).unsqueeze(1).repeat(1, self.num_targets, 1, 1)

        final_queries = (query_k + t_pos_query).view(B, self.num_targets * T, self.d_model)
        decoded_feat = self.st_decoder(
            tgt=final_queries, memory=encoded_feat,
            memory_key_padding_mask=radar_padding_mask
        )

        # Reshape: [B, K*T, d] -> [B, K, T, d]
        decoded_feat = decoded_feat.view(B, self.num_targets, T, self.d_model)

        # ==========================================
        # 5. [核心] 局部时间窗口的 Input Selection
        # ==========================================

        # 创建时间窗口 mask 和距离矩阵
        temporal_mask, time_dist = self._create_temporal_attention_mask(T, M_rad, device)
        # temporal_mask: [T, T*M], time_dist: [T, T*M]

        # 计算 Query 和 Key
        # decoded_feat: [B, K, T, d] -> [B*K, T, d]
        query = self.selection_proj_q(decoded_feat).view(B * self.num_targets, T, self.d_model)

        # encoded_feat: [B, T*M, d] -> [B, T*M, d]，然后扩展给每个 target
        key = self.selection_proj_k(encoded_feat)
        key = key.unsqueeze(1).expand(-1, self.num_targets, -1, -1)  # [B, K, T*M, d]
        key = key.reshape(B * self.num_targets, T * M_rad, self.d_model)  # [B*K, T*M, d]

        # Attention scores: [B*K, T, T*M]
        scores = torch.bmm(query, key.transpose(1, 2)) / (self.d_model ** 0.5)

        # ========== 应用时间窗口约束 ==========
        # (a) 硬性时间窗口 mask
        temporal_mask_expanded = temporal_mask.unsqueeze(0).expand(B * self.num_targets, -1, -1)
        scores = scores.masked_fill(temporal_mask_expanded, float('-inf'))

        # (b) 软性时间衰减：越远的点权重越低
        # decay = exp(-time_decay * distance)
        decay_factor = torch.exp(-torch.relu(self.time_decay) * time_dist)  # [T, T*M]
        decay_factor = decay_factor.unsqueeze(0).expand(B * self.num_targets, -1, -1)
        scores = scores + torch.log(decay_factor + 1e-8)  # 加在 log 空间

        # (c) Padding mask: 屏蔽无效的雷达点
        # radar_padding_mask: [B, T*M] -> [B*K, T, T*M]
        padding_mask_expanded = radar_padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T*M]
        padding_mask_expanded = padding_mask_expanded.expand(-1, self.num_targets, T, -1)
        padding_mask_expanded = padding_mask_expanded.reshape(B * self.num_targets, T, T * M_rad)
        scores = scores.masked_fill(padding_mask_expanded, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [B*K, T, T*M]
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # 加权求和得到 selected_coords
        # raw_coords: [B, T*M, 3] -> [B*K, T*M, 3]
        raw_coords_expanded = raw_coords.unsqueeze(1).expand(-1, self.num_targets, -1, -1)
        raw_coords_expanded = raw_coords_expanded.reshape(B * self.num_targets, T * M_rad, 3)

        # [B*K, T, T*M] @ [B*K, T*M, 3] -> [B*K, T, 3]
        selected_coords = torch.bmm(attn_weights, raw_coords_expanded)

        # Reshape back: [B*K, T, 3] -> [B, K, T, 3]
        selected_coords = selected_coords.view(B, self.num_targets, T, 3)

        # 6. 残差修正
        residual = self.residual_head(decoded_feat)  # [B, K, T, 3]

        # 最终预测 = 选择的基准点 + 小幅残差
        pred_coords = selected_coords + 0.1 * torch.tanh(residual)

        # 限制输出范围
        pred_coords = torch.cat([
            torch.clamp(pred_coords[..., 0:1], -1, 1),  # Pitch
            torch.clamp(pred_coords[..., 1:2], -1, 1),  # Azimuth
            torch.clamp(pred_coords[..., 2:3], 0, 1)  # Range
        ], dim=-1)

        # 7. 分类
        pred_logits = self.prob_head(decoded_feat).squeeze(-1)  # [B, K, T]

        return {'logits': pred_logits, 'boxes': pred_coords}


# ==========================================
# 1. 损失函数
# ==========================================
# ==========================================
# 新版 Loss: 支持 Focal Loss
# ==========================================
# ==========================================
# 升级版 Loss: 物理感知 + Focal Loss
# ==========================================
class PhysicsGuidedCriterion(nn.Module):
    def __init__(self, cost_class=2.0, cost_coord=5.0, cost_smooth=0.5,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        self.cost_smooth = cost_smooth
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def sigmoid_focal_loss(self, inputs, targets, num_boxes):
        prob = inputs.sigmoid()
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)
        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.sum() / num_boxes

    @torch.no_grad()
    def match(self, outputs, targets):
        bs = outputs['logits'].shape[0]
        indices = []
        for b in range(bs):
            prob = torch.sigmoid(outputs['logits'][b].float())
            prob_mean = prob.mean(dim=1)
            out_bbox = outputs['boxes'][b].float()
            tgt = targets[b]

            valid_tgt_mask = tgt[:, :, 3].sum(dim=1) > 0
            if valid_tgt_mask.sum() == 0:
                indices.append(([], []))
                continue

            tgt_bbox = tgt[valid_tgt_mask, :, :3]
            cost_class = -prob_mean.unsqueeze(1).expand(-1, tgt_bbox.shape[0])
            cost_bbox = torch.cdist(out_bbox.flatten(1), tgt_bbox.flatten(1), p=1) / out_bbox.shape[1]

            C = self.cost_class * cost_class + self.cost_coord * cost_bbox
            row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
            indices.append((row_ind.tolist(), col_ind.tolist()))
        return indices

    def forward(self, outputs, targets):
        device = outputs['logits'].device
        indices = self.match(outputs, targets)

        num_boxes = sum(len(x[0]) for x in indices)
        num_boxes = max(num_boxes, 1) * outputs['logits'].shape[2]

        # ========== AMP 类型修复 ==========
        target_classes = torch.zeros_like(outputs['logits'], dtype=torch.float32, device=device)
        loss_bbox = torch.tensor(0.0, device=device, dtype=torch.float32)
        loss_smooth = torch.tensor(0.0, device=device, dtype=torch.float32)

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                valid_tgt_mask = targets[b][:, :, 3].sum(dim=1) > 0
                matched_tgt = targets[b][valid_tgt_mask][tgt_idx]
                target_classes[b, src_idx] = matched_tgt[:, :, 3]

                src_boxes = outputs['boxes'][b][src_idx].float()  # [N, T, 3]
                tgt_pos = matched_tgt[:, :, :3]
                tgt_valid = matched_tgt[:, :, 3].unsqueeze(-1)  # [N, T, 1]

                # 1. 回归 Loss
                diff = torch.abs(src_boxes - tgt_pos) * tgt_valid
                loss_bbox = loss_bbox + diff.sum() / (tgt_valid.sum() + 1e-6)

                # ==========================================
                # 2. [修复] 速度平滑性 Loss（Jerk 约束）
                # ==========================================
                T = src_boxes.shape[1]
                if T > 3:  # 至少需要4帧才能计算 jerk
                    # 速度: v_t = p_{t+1} - p_t
                    vel = src_boxes[:, 1:] - src_boxes[:, :-1]  # [N, T-1, 3]

                    # 加速度: a_t = v_{t+1} - v_t
                    acc = vel[:, 1:] - vel[:, :-1]  # [N, T-2, 3]

                    # Jerk (加速度变化): j_t = a_{t+1} - a_t
                    jerk = acc[:, 1:] - acc[:, :-1]  # [N, T-3, 3]

                    # ========== [关键修复] 正确的连续4帧有效 mask ==========
                    # 对于 jerk[t]，需要 valid[t], valid[t+1], valid[t+2], valid[t+3] 都为1
                    # jerk 有 T-3 个时间点，对应原始时间的 t=0 到 t=T-4
                    jerk_mask = (tgt_valid[:, :-3] *  # [N, T-3, 1] 对应 t
                                 tgt_valid[:, 1:-2] *  # [N, T-3, 1] 对应 t+1
                                 tgt_valid[:, 2:-1] *  # [N, T-3, 1] 对应 t+2
                                 tgt_valid[:, 3:])  # [N, T-3, 1] 对应 t+3
                    # jerk_mask: [N, T-3, 1]

                    # 使用 Smooth L1 更鲁棒
                    jerk_loss = nn.functional.smooth_l1_loss(
                        jerk, torch.zeros_like(jerk), reduction='none'
                    )
                    # jerk_loss: [N, T-3, 3]

                    # 应用 mask 并归一化
                    masked_jerk_loss = jerk_loss * jerk_mask  # [N, T-3, 3]
                    loss_smooth = loss_smooth + masked_jerk_loss.sum() / (jerk_mask.sum() * 3 + 1e-6)

        # 分类 Loss
        loss_cls = self.sigmoid_focal_loss(
            outputs['logits'].float(),
            target_classes,
            num_boxes
        )

        loss_bbox_avg = loss_bbox / max(len(indices), 1)
        loss_smooth_avg = loss_smooth / max(len(indices), 1)

        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox_avg,
            'loss_smooth': loss_smooth_avg,
            'loss_total': 2.0 * loss_cls + 5.0 * loss_bbox_avg + self.cost_smooth * loss_smooth_avg
        }


# ==========================================
# 2. 数据生成器 (保持不变)
# ==========================================
class OnlineDataGenerator:
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.preprocessor = PointCloudPreprocessor(
            time_steps=config['clip_length'], max_radar_points=config['max_radar_points'],
            max_rf_points=config['max_rf_points'], max_targets=config['max_targets'], max_range=config['max_range']
        )

    def generate_batch(self, batch_size, seed=None):
        if seed is not None:
            np.random.seed(seed);
            random.seed(seed)
        config = self.config
        min_frames = batch_size * config['clip_length'] // 2 + config['clip_length']

        # Simulation
        avg_targets = config.get('avg_targets', 10)
        simulator = Trajectory_auto_generator_multitarget(TargetNum=avg_targets)
        sim_steps = int(np.ceil(min_frames / simulator.targets_list[0][0].SectionLen)) + 1
        _, Observations, _, Radar_obs, RF_obs, Clutter_rad, Clutter_rf = simulator.show_data(
            TimeStepRange=[0, sim_steps], Camera=[10000, 0.002], Radar=[1, config['max_range'], 0.25, 3.5, 0.8],
            RFsensor=[10000, 2.5, 0.95], clutter_num_mean=config.get('clutter_mean', 10)
        )
        Fuse_Radar, _, Fuse_RF, _ = Observation_Lables(Observations, Radar_obs, RF_obs, Clutter_rad, Clutter_rf)

        start_indices = [random.randint(0, len(Fuse_Radar) - config['clip_length']) for _ in range(batch_size)]
        all_radar, all_rf, all_gt = [], [], []
        for start_t in start_indices:
            r, rf, gt, _ = self.preprocessor.process_sequence(
                Fuse_Radar[start_t:start_t + config['clip_length']], Fuse_RF[start_t:start_t + config['clip_length']],
                Observations, start_t
            )
            all_radar.append(r);
            all_rf.append(rf);
            all_gt.append(gt)

        return torch.from_numpy(np.stack(all_radar)).float().to(self.device), \
            torch.from_numpy(np.stack(all_rf)).float().to(self.device), \
            torch.from_numpy(np.stack(all_gt)).float().to(self.device)


# ==========================================
# 3. 增强的训练器 (支持 Checkpoint)
# ==========================================

class OnlineTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, config, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.train_generator = OnlineDataGenerator(config, device)
        self.scaler = GradScaler('cuda') if config.get('use_amp', True) else None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        # [新增] 分阶段训练参数
        self.stage1_epochs = config.get('stage1_epochs', 100)
        self.stage2_epochs = config.get('stage2_epochs', 200)
        self.smooth_weight_stage1 = config.get('smooth_weight_stage1', 0.0)
        self.smooth_weight_stage2 = config.get('smooth_weight_stage2', 0.5)

    def _get_current_smooth_weight(self, epoch):
        """根据当前 epoch 返回 smooth loss 的权重"""
        if epoch < self.stage1_epochs:
            # Stage 1: 不使用或少量使用 smooth loss
            return self.smooth_weight_stage1
        else:
            # Stage 2: 逐渐增加 smooth loss 权重
            # 可以做线性插值，让过渡更平滑
            progress = min((epoch - self.stage1_epochs) / 50, 1.0)  # 50个epoch过渡
            return self.smooth_weight_stage1 + progress * (self.smooth_weight_stage2 - self.smooth_weight_stage1)

    def save_checkpoint(self, epoch, is_best=False, filename='v3_checkpoint_latest_r.pth'):
        """保存检查点"""
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        torch.save(state, filename)
        if is_best:
            torch.save(state, 'v3_checkpoint_best_r.pth')
            print(f"  [Save] 最佳模型已保存 -> v3_checkpoint_best_r.pth")

    def load_checkpoint(self, filename):
        """加载检查点"""
        if os.path.isfile(filename):
            print(f"  [Load] 正在加载断点: {filename}")
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler and checkpoint['scheduler']:
                self.scheduler.load_state_dict(checkpoint['scheduler'])

            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            print(f"  [Load] 成功! 从 Epoch {self.start_epoch} 继续训练")
        else:
            print(f"  [Load] 未找到断点文件 {filename}，将开始重新训练")

    def train_one_epoch(self, epoch, num_batches=50):
        self.model.train()
        epoch_losses = defaultdict(float)
        batch_size = self.config['batch_size']

        # [关键] 动态调整 smooth loss 权重
        current_smooth_weight = self._get_current_smooth_weight(epoch)
        self.criterion.cost_smooth = current_smooth_weight
        for batch_idx in range(num_batches):
            seed = epoch * 10000 + batch_idx * 100 + random.randint(0, 99)
            radar, rf, gt = self.train_generator.generate_batch(batch_size, seed=seed)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(radar, rf)
                    losses = self.criterion(outputs, gt)
                    loss = losses['loss_total']
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(radar, rf)
                losses = self.criterion(outputs, gt)
                loss = losses['loss_total']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                self.optimizer.step()
            for k, v in losses.items():
                epoch_losses[k] += v.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches} | "
                      f"Total: {loss.item():.4f} | Box: {losses['loss_bbox'].item():.4f} | "
                      f"Smooth: {losses['loss_smooth'].item():.4f}")
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        self.train_losses.append(epoch_losses['loss_total'])

        # 返回时附带当前的 smooth weight
        epoch_losses['smooth_weight'] = current_smooth_weight
        return dict(epoch_losses)

    @torch.no_grad()
    def validate(self, num_batches=10):
        self.model.eval()
        val_losses = defaultdict(float)
        for batch_idx in range(num_batches):
            seed = 99999 + batch_idx
            radar, rf, gt = self.train_generator.generate_batch(self.config['batch_size'], seed=seed)
            outputs = self.model(radar, rf)
            losses = self.criterion(outputs, gt)
            for k, v in losses.items(): val_losses[k] += v.item()

        for k in val_losses: val_losses[k] /= num_batches
        self.val_losses.append(val_losses['loss_total'])
        return dict(val_losses)

    def train(self, num_epochs, batches_per_epoch=50, val_interval=5):
        print(f"\n{'=' * 20} 训练开始 (从 Epoch {self.start_epoch} 到 {num_epochs}) {'=' * 20}")
        print(f"分阶段训练: Stage1 (0-{self.stage1_epochs}) smooth={self.smooth_weight_stage1}, "
              f"Stage2 ({self.stage1_epochs}+) smooth={self.smooth_weight_stage2}")
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            # 打印当前阶段
            stage = 1 if epoch < self.stage1_epochs else 2
            print(f"\nEpoch {epoch + 1}/{num_epochs} [Stage {stage}]")
            # 训练
            train_loss = self.train_one_epoch(epoch, batches_per_epoch)

            print(f"  Smooth Weight: {train_loss['smooth_weight']:.3f}")
            # 学习率更新
            if self.scheduler:
                self.scheduler.step()
                curr_lr = self.optimizer.param_groups[0]['lr']
                print(f"  LR: {curr_lr:.6f}")
            # 验证与保存
            if (epoch + 1) % val_interval == 0:
                val_loss = self.validate()
                print(f"  Train Loss: {train_loss['loss_total']:.4f} | Val Loss: {val_loss['loss_total']:.4f}")
                is_best = val_loss['loss_total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss['loss_total']
                self.save_checkpoint(epoch, is_best=is_best)
            else:
                self.save_checkpoint(epoch, is_best=False)
                print(f"  Train Loss: {train_loss['loss_total']:.4f}")
            print(f"  Time: {time.time() - epoch_start:.1f}s")
        self.plot_losses()
        print("\n训练完成!")

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train')
        if len(self.val_losses) > 0:
            # 只有部分 epoch 有 val loss，这里简单对齐显示
            val_x = np.linspace(0, len(self.train_losses) - 1, len(self.val_losses))
            plt.plot(val_x, self.val_losses, 'o-', label='Validation')
        plt.title(f'Best Val Loss: {self.best_val_loss:.4f}')
        plt.xlabel('Epoch');
        plt.ylabel('Loss')
        plt.legend();
        plt.grid(True)
        plt.savefig('training_curve.png')
        plt.close()


# ==========================================
# 4. 主程序 & 可视化
# ==========================================
def visualize_predictions(model, generator, preprocessor, device):
    model.eval()
    radar, rf, gt = generator.generate_batch(1, seed=12345)
    with torch.no_grad():
        outputs = model(radar, rf)

    pred_boxes = outputs['boxes'][0].cpu().numpy()
    pred_probs = torch.sigmoid(outputs['logits'][0]).cpu().numpy()
    gt_boxes = gt[0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 俯仰角-方位角 轨迹
    ax = axes[0]
    for k in range(min(10, pred_boxes.shape[0])):
        if pred_probs[k].mean() > 0.3:
            ax.plot(pred_boxes[k, :, 1], pred_boxes[k, :, 0], 'r-', alpha=0.5)
    for k in range(gt_boxes.shape[0]):
        if gt_boxes[k, :, 3].sum() > 0:
            ax.plot(gt_boxes[k, :, 1], gt_boxes[k, :, 0], 'b--', linewidth=1)
    ax.set_title('Traj (Red: Pred, Blue: GT)')
    ax.set_xlim([-1, 1]);
    ax.set_ylim([-1, 1])

    # 距离-时间
    ax = axes[1]
    T = pred_boxes.shape[1]
    for k in range(min(10, pred_boxes.shape[0])):
        if pred_probs[k].mean() > 0.3:
            ax.plot(range(T), pred_boxes[k, :, 2], 'r-', alpha=0.5)
    for k in range(gt_boxes.shape[0]):
        valid = gt_boxes[k, :, 3] > 0.5
        if valid.sum():
            ax.plot(np.where(valid)[0], gt_boxes[k, valid, 2], 'b.', markersize=2)
    ax.set_title('Range vs Time')

    plt.savefig('final_prediction.png')
    print("  [Viz] 可视化保存至 final_prediction.png")


if __name__ == '__main__':
    CONFIG = {
        # ========== 数据参数 ==========
        'clip_length': 32,
        'max_targets': 30,
        'avg_targets': 10,
        'max_radar_points': 100,
        'max_rf_points': 50,
        'max_range': 10000,
        'clutter_mean': 20,

        # ========== 模型参数 ==========
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'temporal_window': 3,  # [新增] 时间窗口半径，可以attention到前后3帧

        # ========== 训练参数 ==========
        'batch_size': 32,
        'num_epochs': 300,
        'batches_per_epoch': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,
        'use_amp': True,

        # ========== 分阶段训练参数 ==========
        'stage1_epochs': 100,  # Stage 1: 0-100 epoch
        'stage2_epochs': 200,  # Stage 2: 100-300 epoch (这个值其实只用于标记)
        'smooth_weight_stage1': 0.0,  # Stage 1 不加 smooth loss
        'smooth_weight_stage2': 0.5,  # Stage 2 逐渐加入 smooth loss

        # ========== Loss 参数 ==========
        'cost_class': 2.0,
        'cost_coord': 5.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,

        # ========== 其他 ==========
        'resume_path': None
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 创建模型（使用 CONFIG 中的模型参数）
    model = RFGuidedTransformerTracker(
        num_targets=CONFIG['max_targets'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_encoder_layers=CONFIG['num_encoder_layers'],
        num_decoder_layers=CONFIG['num_decoder_layers'],
        dim_feedforward=CONFIG['dim_feedforward'],
        dropout=CONFIG['dropout'],
        temporal_window=CONFIG['temporal_window']  # [新增]
    ).to(device)

    # 创建 Loss（初始 smooth weight 为 0）
    criterion = PhysicsGuidedCriterion(
        cost_class=CONFIG['cost_class'],
        cost_coord=CONFIG['cost_coord'],
        cost_smooth=CONFIG['smooth_weight_stage1'],  # 初始值，训练中会动态调整
        focal_alpha=CONFIG['focal_alpha'],
        focal_gamma=CONFIG['focal_gamma']
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs'],
        eta_min=1e-6
    )

    trainer = OnlineTrainer(model, criterion, optimizer, scheduler, CONFIG, device)

    if CONFIG['resume_path']:
        trainer.load_checkpoint(CONFIG['resume_path'])

    # # ================= 关键修改：手动加载权重实现微调 =================
    # if os.path.isfile(CONFIG['resume_path']):
    #     print(f"[*] 正在加载微调模型权重: {CONFIG['resume_path']}")
    #     checkpoint = torch.load(CONFIG['resume_path'])
    #
    #     # 仅加载模型权重 (State Dict)
    #     model.load_state_dict(checkpoint['state_dict'])
    #
    #     # 不要加载 optimizer 和 scheduler，也不要覆盖 start_epoch
    #     # trainer.start_epoch = checkpoint['epoch']  <-- 注释掉这句，让 epoch 从 0 开始计数
    #
    #     print("[*] 权重加载成功！优化器已重置，开始 Stage 2 微调训练...")
    # else:
    #     print("[!] 未找到模型文件，请检查路径")

    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        batches_per_epoch=CONFIG['batches_per_epoch']
    )

    # 可视化
    visualize_predictions(model, trainer.train_generator, trainer.train_generator.preprocessor, device)
