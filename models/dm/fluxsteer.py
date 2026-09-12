from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from omegaconf import DictConfig
from models.base import DIFFUSER
from models.optimizer.optimizer import Optimizer
from pytorch3d.ops import knn_points
from utils.handmodel import get_handmodel, ERF_loss, SPF_loss, SRF_loss
from einops import rearrange, repeat
import numpy as np
from loguru import logger

import trimesh
import os

def load_object_mesh_from_data(data, batch_idx=0):
    """
    从数据中加载物体的完整网格（而不仅仅是点云）
    用于可选的轨迹网格导出
    
    Args:
        data: 数据字典，包含 'scene_id', 'scale', 'scene_rot_mat' 等信息
        batch_idx: batch 中的索引
    
    Returns:
        trimesh.Trimesh: 变换后的物体网格，如果加载失败则返回 None
    """
    try:
        # 尝试从数据中获取物体名称（scene_id）
        if 'scene_id' in data:
            # scene_id 可能是 list/tuple（batch）或 string（单个）
            scene_id_raw = data['scene_id']
            if isinstance(scene_id_raw, (list, tuple)):
                obj_name = scene_id_raw[batch_idx]
            else:
                obj_name = scene_id_raw
        elif 'obj_name' in data:
            obj_name = data['obj_name'][batch_idx] if isinstance(data['obj_name'], (list, tuple)) else data['obj_name']
        else:
            logger.warning("No 'scene_id' or 'obj_name' in data, cannot load mesh")
            return None
        
        # 数据集根目录
        dataset_root = os.environ['FLUXSTEER_DATASET_DIR']
        
        # 尝试加载网格文件（多种路径）
        # 格式：contactdb+apple -> object/contactdb/apple/apple.stl
        possible_paths = [
            os.path.join(dataset_root, 'meshdata', obj_name, 'coacd', 'decomposed.obj'),
            os.path.join(dataset_root, 'meshdata', obj_name, 'model.obj'),
            os.path.join(dataset_root, 'object', obj_name.replace('+', '/'), f"{obj_name.split('+')[-1]}.stl"),
        ]
        
        mesh_path = None
        for path in possible_paths:
            if os.path.exists(path):
                mesh_path = path
                break
        
        if not mesh_path:
            logger.debug(f"Mesh file not found for {obj_name}")
            return None
        
        # 加载网格
        mesh = trimesh.load(mesh_path, process=False)
        
        # 应用变换（scale 和 rotation）
        # 注意：MultiDex数据集中，物体已经通过scene_rot_mat旋转过了，点云是旋转后的
        # 但网格文件是原始的，所以需要应用相同的旋转
        if 'scene_rot_mat' in data:
            rot = data['scene_rot_mat']
            if isinstance(rot, torch.Tensor):
                if rot.dim() == 3:  # (B, 3, 3)
                    rot = rot[batch_idx].cpu().numpy()
                else:  # (3, 3)
                    rot = rot.cpu().numpy()
            # 应用旋转
            T = np.eye(4)
            T[:3, :3] = rot
            mesh.apply_transform(T)
        elif 'rot' in data:
            rot = data['rot']
            if isinstance(rot, torch.Tensor):
                rot = rot[batch_idx].cpu().numpy() if rot.dim() > 2 else rot.cpu().numpy()
            T = np.eye(4)
            T[:3, :3] = rot
            mesh.apply_transform(T)
        
        # 应用缩放（如果存在）
        if 'scale' in data:
            scale = data['scale']
            if isinstance(scale, torch.Tensor):
                scale = scale[batch_idx].item() if scale.dim() > 0 else scale.item()
            mesh.apply_scale(scale)
        
        logger.debug(f"✅ Loaded mesh for {obj_name} from {mesh_path}")
        return mesh
        
    except Exception as e:
        logger.warning(f"Failed to load object mesh: {e}")
        return None

@DIFFUSER.register()
class FluxSteer(nn.Module):
    def __init__(self, eps_model: nn.Module, cfg: DictConfig, has_obser: bool, *args, **kwargs) -> None:
        super(FluxSteer, self).__init__()
        
        # 手部关节角度的归一化边界（用于反归一化）
        self.register_buffer('_joint_angle_lower', torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.]), persistent=False)
        self.register_buffer('_joint_angle_upper', torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.]), persistent=False)
        self.register_buffer('_global_trans_lower', torch.tensor([-0.13128923, -0.10665303, -0.45753425]), persistent=False)
        self.register_buffer('_global_trans_upper', torch.tensor([0.12772022, 0.22954416, -0.21764427]), persistent=False)
        self._NORMALIZE_LOWER = -1.
        self._NORMALIZE_UPPER = 1.
        
        self.cfg = cfg
        self.eps_model = eps_model  # U-Net模型
        self.timesteps = cfg.steps
        self.has_observation = has_obser
        self.last_posthoc_stats = None
        
        # 损失函数
        if cfg.loss_type == 'l1':
            self.criterion = F.l1_loss
        elif cfg.loss_type == 'l2':
            self.criterion = F.mse_loss
        else:
            raise Exception('不支持的损失函数类型。')
        
        self.optimizer = None

    @property
    def device(self):
        return next(self.eps_model.parameters()).device
    
    def apply_observation(self, x_t: torch.Tensor, data: Dict) -> torch.Tensor:
        """如果有观测数据（起始状态），则固定住"""
        if self.has_observation and 'start' in data:
            start = data['start']
            T = start.shape[1]
            x_t[:, 0:T, :] = start[:, 0:T, :].clone()
        return x_t
    
    def _sigma_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Stochastic Flow Matching的噪声schedule σ(t)
        
        参考论文：Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
        官方实现：sigma = 0.01 (常数) 或 sigma = sqrt(t*(1-t))
        
        Args:
            t: 时间步 (B,), 范围[0, 1]
        
        Return:
            噪声强度 σ(t) (B,)
        
        常见schedule:
        - Constant: σ(t) = σ_max（论文推荐，简单有效）
        - Time-dependent: σ(t) = σ_max * sqrt(t*(1-t))（论文也提到）
        - Linear: σ(t) = σ_max * t
        - Cosine: σ(t) = σ_max * (1 - cos(π*t/2))
        """
        sigma_max = self.cfg.get('sigma_max', 0.01)  # ✅ 修正：默认0.01（参考官方实现）
        schedule_type = self.cfg.get('sigma_schedule', 'constant')  # ✅ 修正：默认constant
        
        if schedule_type == 'constant':
            # 恒定噪声：在所有时刻加入相同强度的噪声（论文推荐）
            return sigma_max * torch.ones_like(t)
        elif schedule_type == 'time-dependent':
            # 时间依赖：在插值中间（t≈0.5）时噪声最强（论文也提到的选择）
            # 在端点（t≈0或t≈1）时噪声弱，保持端点清晰
            return sigma_max * torch.sqrt(t * (1 - t) + 1e-8)
        elif schedule_type == 'linear':
            # 线性增长：t=0无噪声，t=1最大噪声
            return sigma_max * t
        elif schedule_type == 'cosine':
            # 余弦增长：平滑过渡
            return sigma_max * (1 - torch.cos(np.pi * t / 2))
        else:
            raise ValueError(f"Unknown sigma schedule: {schedule_type}")
    
    def forward(self, data: Dict) -> torch.Tensor:
        """
        Flow Matching的训练损失计算
        核心思想：学习从噪声到数据的瞬时速度场
        
        🆕 支持Stochastic Flow Matching (SFM):
        - 原版FM（确定性）: z_t = (1-t)*x0 + t*noise
        - SFM（随机性）: z_t = (1-t)*x0 + t*noise + σ(t)*ε
        
        SFM优点：
        1. 训练信号更丰富 → 避免overfitting
        2. 模型学习"模糊路径" → 泛化能力更强
        3. 兼容Diffusion模型参数初始化（可迁移）
        
        配置：
        - use_stochastic_fm: true/false (是否使用SFM)
        - sigma_max: 0.05-0.2 (噪声强度)
        - sigma_schedule: 'time-dependent'/'linear'/'cosine'/'constant'
        """
        B, *x_shape = data['x'].shape
        x0 = data['x']  # 真实数据（目标抓取姿态）
        
        # 1. 采样噪声和时间
        noise = torch.randn_like(x0, device=self.device)
        t = torch.rand(B, device=self.device)  # 时间 t ~ Uniform[0, 1]
        
        # 2. 🆕 判断是否使用Stochastic Flow Matching
        use_sfm = self.cfg.get('use_stochastic_fm', False)
        
        if use_sfm:
            # === Stochastic Flow Matching (SFM) ===
            # 构造随机插值路径: z_t = (1-t)*x0 + t*noise + σ(t)*ε
            
            # 2.1 计算噪声强度 σ(t)
            sigma_t = self._sigma_schedule(t)  # (B,)
            sigma_t_shape = sigma_t.reshape(B, *((1, ) * len(x_shape)))
            
            # 2.2 采样额外的随机噪声 ε ~ N(0, I)
            epsilon = torch.randn_like(x0, device=self.device)
            
            # 2.3 构造随机路径
            t_shape = t.reshape(B, *((1, ) * len(x_shape)))
            z_t = (1 - t_shape) * x0 + t_shape * noise + sigma_t_shape * epsilon
            
            # 2.4 计算目标速度（SFM的速度场）
            # 对于SDE形式: dx_t = u_t dt + σ(t) dW_t
            # 对应的确定性速度场（期望）: u_t = (noise - x0)
            # 注意：虽然路径是随机的，但目标速度仍然指向期望方向
            v_target = noise - x0
            
        else:
            # === 原版Flow Matching（确定性） ===
            # 构造确定性插值路径: z_t = (1-t)*x0 + t*noise
            
            t_shape = t.reshape(B, *((1, ) * len(x_shape)))
            z_t = (1 - t_shape) * x0 + t_shape * noise
            
            # 计算目标速度
            v_target = noise - x0  # Flow Matching的核心：速度就是从数据指向噪声
        
        # 3. 模型预测速度
        # 注意：这里t是一维张量(B,)，U-Net会自动识别为单时间参数
        condition = self.eps_model.condition(data)
        v_pred = self.eps_model(z_t, t, condition)
        
        # 4. 计算Flow Matching损失
        loss_fm = self.criterion(v_pred, v_target)
        
        return {'loss': loss_fm}
    
    def p_sample_loop(self, data: Dict, num_steps: int = 100, method: str = 'heun') -> torch.Tensor:
        """
        Flow Matching的标准采样过程
        使用Euler或Heun方法进行多步ODE积分
        
        Args:
            data: 输入数据
            num_steps: 积分步数，默认100步
            method: ODE求解器 ('euler' 或 'heun')，默认heun（更精确）
        """
        # 1. 从标准正态分布采样初始噪声（t=1的状态）
        z_t = torch.randn_like(data['x'], device=self.device)
        z_t = self.apply_observation(z_t, data)
        
        # 2. 准备时间步长
        dt = 1.0 / num_steps  # 从t=1到t=0，每步的时间间隔
        
        with torch.no_grad():
            # 获取条件（只需要获取一次）
            condition = self.eps_model.condition(data)
            
            # 3. ODE积分：从t=1逐步走到t=0
            for step in range(num_steps):
                t_current = 1.0 - step * dt  # 当前时间点
                t_next = max(0.0, t_current - dt)  # 下一个时间点
                
                # 每步都重新apply observation（和师兄的实现一致）
                z_t = self.apply_observation(z_t, data)
                
                t_current_tensor = torch.full((z_t.shape[0],), t_current, device=self.device)
                
                if method == 'heun':
                    # Heun方法（二阶Runge-Kutta，更精确）
                    # 预测当前时刻的速度
                    v_current = self.eps_model(z_t, t_current_tensor, condition)
                    
                    # 用Euler预测下一步
                    z_pred = z_t - v_current * dt
                    
                    # 在预测点评估速度
                    t_next_tensor = torch.full((z_t.shape[0],), t_next, device=self.device)
                    v_next = self.eps_model(z_pred, t_next_tensor, condition)
                    
                    # Heun步进：用两个速度的平均
                    z_t = z_t - 0.5 * (v_current + v_next) * dt
                else:
                    # Euler方法（一阶，更快）
                    v_t = self.eps_model(z_t, t_current_tensor, condition)
                    z_t = z_t - v_t * dt
        
        # z_t现在应该接近t=0（真实数据）
        x_0 = z_t
        return x_0.unsqueeze(1)

    def p_sample_loop_with_posthoc(self, data: Dict, num_steps: int,
                                   method: str, learning_rate: float,
                                   iterations: int, w_erf: float,
                                   w_spf: float, w_srf: float) -> torch.Tensor:
        """Run Vanilla FM and refine only its normalized terminal state."""
        from models.dm.refinement import refine_terminal_state

        terminal_state = self.p_sample_loop(data, num_steps=num_steps, method=method).squeeze(1)

        def energy_fn(state: torch.Tensor) -> torch.Tensor:
            pose = self._denormalize_pose(state, data)
            return self._compute_physics_energy(
                pose,
                data,
                w_erf=w_erf,
                w_spf=w_spf,
                w_srf=w_srf,
                use_baseline_potential=False,
            )

        refined_state, self.last_posthoc_stats = refine_terminal_state(
            terminal_state,
            energy_fn,
            learning_rate=learning_rate,
            iterations=iterations,
        )
        return refined_state.unsqueeze(1)

    def _compute_physics_energy(self, pose: torch.Tensor, data: Dict,
                               w_erf: float = 1.0, w_spf: float = 1.0, w_srf: float = 1.0,
                               use_baseline_potential: bool = False) -> torch.Tensor:
        """
        计算每个样本的物理能量 J(x) (Energy)，用于MC引导
        
        改进版物理约束：
        1. SPF: 包含全局引导（手心距离）和局部引导（接触点距离），防止远处梯度消失。
        2. ERF: 对穿透深度施加更强的惩罚（非线性/放大）。
        3. SRF: 保持自穿透检查。
        
        Args:
            pose: 完整的手姿态 (B, 33)
            data: 数据字典
            w_erf: 手-物体穿透能量权重
            w_spf: 手指自穿透能量权重
            w_srf: 姿态自然度能量权重
            use_baseline_potential: 是否使用baseline原文定义的物理势能（Mean ERF, Hard Threshold SPF）
            
        Return:
            energy: (B,) 每个样本的总能量值
        """
        B = pose.shape[0]
        
        # 0. 鲁棒性检查
        if torch.isnan(pose).any():
            return torch.full((B,), 1e6, device=self.device) # 返回高能量而非NaN
        
        try:
            hand_model = get_handmodel(batch_size=B, device=self.device)
            hand_model.update_kinematics(q=pose)
        except Exception as e:
            logger.error(f"❌ Hand model error: {e}")
            return torch.full((B,), 1e6, device=self.device)
        
        # 获取点云
        hand_pcd = hand_model.get_surface_points(q=pose).to(dtype=torch.float32)
        obj_pcd = data['pos'].to(self.device).to(dtype=torch.float32)
        if isinstance(data['normal'], torch.Tensor):
            normal = data['normal'].to(self.device).to(dtype=torch.float32)
        else:
            normal = torch.tensor(np.array(data['normal']), device=self.device, dtype=torch.float32)
        
        # === 修复 Scale 问题 ===
        # 如果 data 中包含 'scale'，说明物体点云被缩放过（通常是放大回 CAD 尺寸）。
        # 我们需要把它缩放回真实物理尺寸，以匹配手模型。
        if 'scale' in data:
            scale = data['scale']
            if isinstance(scale, torch.Tensor):
                scale = scale.to(self.device).to(dtype=torch.float32)
            else:
                scale = torch.tensor(scale, device=self.device, dtype=torch.float32)
            
            # Reshape scale for broadcasting: (B, 1, 1)
            if scale.ndim == 1:
                scale = scale.view(-1, 1, 1)
            
            # Broadcast scale if necessary
            if scale.shape[0] != B:
                 if scale.shape[0] == 1:
                     # Single scale for all
                     scale = scale.repeat(B, 1, 1)
                 elif B % scale.shape[0] == 0:
                     # Repeat scale for MC samples
                     K = B // scale.shape[0]
                     scale = repeat(scale, 'b 1 1 -> (b k) 1 1', k=K)
                 else:
                     logger.warning(f"⚠️ Scale shape {scale.shape} mismatch with B={B}, using raw scale.")

            # Apply scale: pos_real = pos_cad * scale (because dataset did pos_cad = pos_real / scale)
            # Check if obj_pcd needs broadcasting first to match scale's batch dim if scale was broadcasted differently?
            # Actually, logic below handles obj_pcd broadcasting. Let's do scaling AFTER broadcasting obj_pcd.
            pass # We will apply it after broadcasting obj_pcd
        else:
            scale = None

        # 广播检查：如果pose是(MC*B)，而obj只有(B)，则重复obj
        if obj_pcd.shape[0] != B:
            K = B // obj_pcd.shape[0]
            obj_pcd = repeat(obj_pcd, 'b n c -> (b k) n c', k=K)
            normal = repeat(normal, 'b n c -> (b k) n c', k=K)
            
        # 现在 obj_pcd 是 (B, N, 3)。如果 scale 存在，确保它也是 (B, 1, 1) 并应用
        if scale is not None:
             # 如果 scale 还没被 broadcast (因为上面只是准备逻辑)，在这里确保它匹配
             if scale.shape[0] != B:
                 if B % scale.shape[0] == 0:
                     K = B // scale.shape[0]
                     scale = repeat(scale, 'b 1 1 -> (b k) 1 1', k=K)
             
             # Apply scale correction
             obj_pcd = obj_pcd * scale
            
        obj_pcd_nor = torch.cat((obj_pcd, normal), dim=-1)
        obj_points = obj_pcd_nor[:, :, :3]
        obj_normals = obj_pcd_nor[:, :, 3:6]
        
        # KNN计算 (Hand -> Object)
        # 确保类型一致
        hand_pcd = hand_pcd.float()
        obj_points = obj_points.float()
        
        # K=1, 找到每个手部点最近的物体点
        knn_result = knn_points(hand_pcd, obj_points, K=1, return_nn=True)
        dists_sq = knn_result.dists # (B, N_hand, 1)
        dists = dists_sq.sqrt()
        indices = knn_result.idx
        
        closest_obj_points = torch.gather(obj_points, 1, indices.expand(-1, -1, 3))
        closest_obj_normals = torch.gather(obj_normals, 1, indices.expand(-1, -1, 3))
        
        # === 1. ERF Energy (External Repulsion / Anti-Penetration) ===
        # 向量：手 -> 物体表面点
        vec_hand_to_obj = closest_obj_points - hand_pcd
        proj_dist = (vec_hand_to_obj * closest_obj_normals).sum(dim=2, keepdim=True)
        
        # 穿透深度：只有当 proj_dist > 0 时才算穿透
        penetration_depth = F.relu(proj_dist) # (B, N_hand, 1)
        
        if use_baseline_potential:
            # === baseline Formula: Mean Penetration ===
            # 计算平均穿透深度（对穿透点求平均）
            # 注意：如果没有任何点穿透，则为0
            penetration_mask = penetration_depth > 1e-6
            count = penetration_mask.sum(dim=1).squeeze(-1) # (B,)
            total_depth = penetration_depth.sum(dim=1).squeeze(-1) # (B,)
            # 避免除以0
            energy_erf = total_depth / (count + 1e-6)
        else:
            # === Ours Formula: Max Penetration (One-Strike-Out) ===
            # 使用 max 深度，而不是 mean，因为只要有一处深穿透就是失败
            max_penetration = penetration_depth.max(dim=1).values.squeeze(-1) # (B,)
            energy_erf = max_penetration
        
        # === 2. SPF Energy (Surface Pulling / Attraction) ===
        if use_baseline_potential:
            # === baseline Formula: Hard Threshold ===
            # 只对距离小于阈值（3cm）的点施加惩罚
            # L_SPF = sum(sqrt(d_i)) / (|S| + eta)
            thres_contact = 0.03
            dist_mask = dists < thres_contact # (B, N_hand, 1)
            masked_dists = dists * dist_mask # 只保留 < thres 的距离
            
            # Sum of distances (or sqrt of squared distances, since dists is already sqrt)
            total_dist = masked_dists.sum(dim=1).squeeze(-1)
            count_dist = dist_mask.sum(dim=1).squeeze(-1)
            
            energy_spf = total_dist / (count_dist + 1e-6)
            
            # 如果所有点都离得远，baseline的Loss是0（导致梯度消失），这里我们保持原样以复现其缺陷
        else:
            # === Ours Formula: Global + Local Top-K ===
            hand_center = hand_pcd.mean(dim=1) # (B, 3)
            obj_center = obj_points.mean(dim=1) # (B, 3)
            global_dist = (hand_center - obj_center).norm(dim=-1) # (B,)
            
            # Local Contact
            thres_contact = 0.03 # 3cm
            # 使用 Softmin 类型的逻辑或者简单的 masked mean
            # 如果距离小于阈值，希望它更小（贴合）
            # 如果所有点都大于阈值，则依赖全局距离

            # 计算所有点到物体的平均距离（Robust Mean）
            # 为了避免离群点（比如手背）干扰，我们可以取最近的k个点的平均距离
            # Top-k nearest distances
            k_nearest = 50 # 假设我们希望至少50个点接触
            topk_dists, _ = torch.topk(dists.squeeze(-1), k=k_nearest, dim=1, largest=False)
            local_dist = topk_dists.mean(dim=1) # (B,)
            
            # 组合SPF
            # 如果离得远，global_dist 主导。
            # 如果离得近，local_dist 主导。
            energy_spf = 1.0 * global_dist + 5.0 * local_dist
        
        # === 3. SRF Energy (Self-Penetration) ===
        hand_keypoints = hand_model.get_keypoints(q=pose) # (B, N_k, 3)
        diff = hand_keypoints.unsqueeze(2) - hand_keypoints.unsqueeze(1)
        dist_mat = diff.norm(dim=-1) # (B, N, N)
        
        N_k = hand_keypoints.shape[1]
        eye_mask = torch.eye(N_k, device=self.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        dist_mat = dist_mat.masked_fill(eye_mask, 1e6)
        
        # 阈值：2cm
        # 惩罚所有小于阈值的点对
        self_collision = F.relu(0.02 - dist_mat)
        energy_srf = self_collision.sum(dim=(1, 2)) # (B,)
        
        # === Total Energy ===
        total_energy = w_erf * energy_erf + w_spf * energy_spf + w_srf * energy_srf
        
        return total_energy

    def p_sample_loop_with_mc_guidance(self, data: Dict, num_steps: int = 20, 
                                     mc_samples: int = 100, 
                                     w_erf: float = 1.0, w_spf: float = 1.0, w_srf: float = 1.0,
                                     method: str = 'heun', guidance_strategy: str = 'local_sim_mc',
                                     temperature: float = 0.05, guidance_scale: float = 30.0,
                                     use_baseline_potential: bool = False,
                                     save_intermediate: bool = False,
                                     save_dir: str = None) -> torch.Tensor:
        """
        使用Monte Carlo引导的Flow Matching采样
        
        Args:
            guidance_strategy: 引导策略
                - 'legacy': 原来的反解方式 (Algorithm 2 in paper)
                - 'global_support': (Strategy A) 先采样一批支持集，全程引导
                - 'local_sim_mc': (Strategy B) 每一步基于预测的x0进行局部采样引导
            use_baseline_potential: 是否使用baseline原文定义的物理势能公式（用于消融实验）
            save_intermediate: 是否保存中间时刻的手部姿态（用于可视化生成过程）
            save_dir: 保存中间状态的目录路径
        """
        B = data['x'].shape[0]
        dt = 1.0 / num_steps
        condition = self.eps_model.condition(data)
        
        # 创建保存目录（如果需要）
        if save_intermediate and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"📁 Intermediate states will be saved to: {save_dir}")
        
        # === Strategy A: Global Support Set Preparation ===
        support_set_x0 = None
        support_set_energy = None
        
        if guidance_strategy == 'global_support':
            logger.info(f"🚀 [Guidance] Generating Global Support Set (Size={mc_samples})...")
            # 1. Run unguided sampling to get candidate x0s
            # We can use a faster/fewer steps solver for this, or just one step prediction?
            # For quality, let's run a fast Heun solver with fewer steps (e.g. 10 steps)
            support_steps = 10
            # Create a larger batch for support set: (B * mc_samples)
            # Note: This might be memory intensive!
            # We assume B is small (e.g. 1). If B is large, we need to be careful.
            
            # Repeat data for support set
            data_support = {k: v.repeat(mc_samples, *([1]*(v.ndim-1))) if isinstance(v, torch.Tensor) else v 
                           for k, v in data.items()}
            if 'pos' in data: # obj pcd
                data_support['pos'] = data['pos'].repeat(mc_samples, 1, 1)
            if 'normal' in data:
                data_support['normal'] = data['normal'].repeat(mc_samples, 1, 1)
                
            # Unguided sampling
            with torch.no_grad():
                x0_support_norm = self.p_sample_loop(data_support, num_steps=support_steps, method='euler').squeeze(1)
            
            # 2. Calculate Energy for Support Set
            x0_support_real = self._denormalize_pose(x0_support_norm, data_support)
            # Energy is (B*N, )
            energy_support = self._compute_physics_energy(x0_support_real, data_support, w_erf, w_spf, w_srf, use_baseline_potential=use_baseline_potential)
            
            # Reshape to (B, N, D) and (B, N)
            support_set_x0 = rearrange(x0_support_norm, '(n b) d -> b n d', n=mc_samples, b=B)
            support_set_energy = rearrange(energy_support, '(n b) -> b n', n=mc_samples, b=B)
            
            # Pre-calculate weights (Boltzmann)
            min_E = support_set_energy.min(dim=1, keepdim=True).values
            support_weights = torch.exp(-(support_set_energy - min_E)) # (B, N)
            # Self-normalize weights?
            # support_weights = support_weights / support_weights.sum(dim=1, keepdim=True)
            
            logger.info(f"✅ Support Set Ready. Min Energy: {min_E.mean():.4f}")

        # 1. Initialize t=1 state
        z_t = torch.randn_like(data['x'], device=self.device)
        z_t = self.apply_observation(z_t, data)
        
        # 2. ODE Loop t=1 -> t=0
        for step in range(num_steps):
            t_current = 1.0 - step * dt
            z_t = self.apply_observation(z_t, data)
            
            # === Visualization: Save Intermediate States ===
            if save_intermediate and save_dir is not None:
                # 只保存关键时间点: t=1.0, 0.75, 0.5, 0.25, 0.0
                target_times = [1.0, 0.75, 0.5, 0.25, 0.0]
                # 判断当前时刻是否接近目标时间点（容差为半步长）
                is_target_time = any(abs(t_current - target_t) < 0.5 * dt for target_t in target_times)
                
                if is_target_time:
                    try:
                        # 1. Save Object (Only once per batch)
                        if step == 0:
                            for batch_idx in range(B):
                                # Create subdirectory for each sample
                                sample_dir = os.path.join(save_dir, str(batch_idx + 1))
                                os.makedirs(sample_dir, exist_ok=True)
                                
                                # 尝试加载完整的物体网格（包含面信息）
                                obj_mesh = load_object_mesh_from_data(data, batch_idx)
                                
                                if obj_mesh is not None:
                                    # 保存为OBJ格式（完整网格，可在Blender中渲染）
                                    obj_path_obj = os.path.join(sample_dir, 'object.obj')
                                    obj_mesh.export(obj_path_obj)
                                    logger.debug(f"💾 Saved object mesh to {obj_path_obj}")
                                else:
                                    # 回退：如果无法加载网格，保存点云
                                    if 'pos' in data:
                                        obj_pcd = data['pos'][batch_idx].cpu().numpy()
                                        if 'scale' in data:
                                            s = data['scale'][batch_idx].item() if isinstance(data['scale'], torch.Tensor) else data['scale']
                                            obj_pcd = obj_pcd * s
                                        
                                        # 保存为PLY格式（点云）
                                        obj_pointcloud = trimesh.PointCloud(vertices=obj_pcd)
                                        obj_path_ply = os.path.join(sample_dir, 'object.ply')
                                        obj_pointcloud.export(obj_path_ply)
                                        
                                        # 同时保存OBJ格式（仅顶点）
                                        obj_path_obj = os.path.join(sample_dir, 'object.obj')
                                        trimesh.Trimesh(vertices=obj_pcd).export(obj_path_obj)
                                        logger.debug(f"💾 Saved object pointcloud to {obj_path_obj}")
                            
                            logger.info(f"💾 Saved {B} object(s) to {save_dir}/{{1,2,...}}")

                        # 2. Save Current Hand State (z_t) for all batch samples
                        z_t_real = self._denormalize_pose(z_t, data)
                        hand_model_vis = get_handmodel(batch_size=B, device=self.device)
                        
                        for batch_idx in range(B):
                            sample_dir = os.path.join(save_dir, str(batch_idx + 1))
                            os.makedirs(sample_dir, exist_ok=True)
                            
                            meshes = hand_model_vis.get_meshes_from_q(q=z_t_real, i=batch_idx)
                            full_mesh = trimesh.util.concatenate(meshes)
                            
                            # 更清晰的文件命名（t从1到0递减）
                            save_name = f"hand_t_{t_current:.2f}.obj"
                            save_path = os.path.join(sample_dir, save_name)
                            full_mesh.export(save_path)
                        
                        logger.info(f"💾 Saved {B} hand state(s) at t={t_current:.2f}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to save intermediate mesh: {e}")

            g_t = torch.zeros_like(z_t)
            
            # ==========================================
            # ===       Guidance Calculation         ===
            # ==========================================
            
            # Guidance hyperparams are now passed as parameters
            # Smaller temperature -> sharper distribution (more focus on low energy)
            # Scale factor to make g_t comparable to v_model (~6.0) 
            
            # --- Strategy A: Global Support (Official gMC Algorithm 2) ---
            if guidance_strategy == 'global_support':
                # Constants for numerical stability
                MC_EP = 1e-6
                
                # 1. Calculate log probability of z_t given each support sample x0_i
                # Assumption: z_t = (1-t)x0 + t*noise
                # So z_t | x0 ~ N((1-t)x0, t^2 * I)
                # We calculate log p(z_t | x0_i)
                
                # Flatten dimensions for calculation
                # z_t: (B, D) -> (B, 1, D)
                # support_set_x0: (B, N, D)
                
                t_safe = max(t_current, 1e-3) # Avoid division by zero
                std_t = t_safe # Standard deviation is proportional to t
                
                # Calculate Gaussian Log Prob: -0.5 * ||(z - mu)/std||^2 - log(std) - const
                # Mean mu = (1-t) * x0
                mu = (1 - t_current) * support_set_x0 # (B, N, D)
                
                # Difference
                diff = z_t.unsqueeze(1) - mu # (B, N, D)
                
                # Squared norm
                # We sum over feature dimensions D
                norm_sq = diff.square().sum(dim=-1) # (B, N)
                
                # Log Prob (ignoring constant factors like 2pi)
                # log_p = -0.5 * norm_sq / std^2 - D * log(std)
                D_dim = z_t.shape[-1]
                log_p_t_given_x0 = -0.5 * norm_sq / (std_t**2 + MC_EP) - D_dim * math.log(std_t + MC_EP) # (B, N)
                
                # 2. Calculate Energy Term (J) with Self-Normalization
                # support_set_energy: (B, N)
                # Invert energy to get "Value" (Higher is better): v = -energy
                v_support = -support_set_energy # (B, N)
                
                # Self-Normalization (Trick C)
                # Normalize v across the support set batch
                v_mean = v_support.mean(dim=1, keepdim=True)
                v_std = v_support.std(dim=1, keepdim=True)
                v_norm = (v_support - v_mean) / (v_std + 1e-8)
                
                # Clamp v_norm to avoid extreme outliers causing numerical instability
                v_norm = v_norm.clamp(-5.0, 5.0)
                
                # J_term = exp(Scale * v_norm)
                # We keep everything in Log Space to avoid overflow
                scale_factor = 1.0 / temperature # e.g. 1/0.05 = 20
                log_J = scale_factor * v_norm # (B, N)
                # J_ = torch.exp(log_J) # <--- DO NOT COMPUTE THIS DIRECTLY
                
                # 3. Calculate Marginal Log Prob log_p(z_t) using LogSumExp (Trick B)
                log_p_t = torch.logsumexp(log_p_t_given_x0, dim=1, keepdim=True) - math.log(mc_samples) # (B, 1)
                
                # 4. Calculate Log Normalization Z
                # Z = E [ e^{-E} p(z|x0) / p(z) ]
                # log_Z term inside expectation: log(J) + log_p(z|x0) - log_p(z)
                
                log_integrand = log_J + log_p_t_given_x0 # (B, N)
                log_Z = torch.logsumexp(log_integrand, dim=1, keepdim=True) - math.log(mc_samples) - log_p_t # (B, 1)
                
                # 5. Calculate Velocity u
                u = (z_t.unsqueeze(1) - support_set_x0) / t_safe # (B, N, D)
                
                # 6. Assemble Guidance g_t (Log-Space Trick A)
                # weight_i = exp(log_p(z|x0_i) - log_p(z))
                log_weight = log_p_t_given_x0 - log_p_t # (B, N)
                weight = torch.exp(log_weight)
                
                # Term = (e^{-E}/Z - 1) = (J/Z - 1)
                # Compute J/Z in log space: exp(log_J - log_Z)
                log_ratio = log_J - log_Z # (B, N)
                term = torch.exp(log_ratio) - 1.0 # (B, N)
                
                # Final Sum
                # g_t = mean( weight * term * u )
                # Note: weight and term are scalars per sample, u is vector
                
                g_t_components = weight.unsqueeze(-1) * term.unsqueeze(-1) * u # (B, N, D)
                g_t = g_t_components.mean(dim=1) * guidance_scale
                
                # Logging
                if step % 10 == 0:
                    z_t_real = self._denormalize_pose(z_t, data)
                    energy_current = self._compute_physics_energy(z_t_real, data, w_erf, w_spf, w_srf, use_baseline_potential=use_baseline_potential)
                    avg_energy = energy_current.mean().item()
                    min_e = energy_current.min().item()
                    
                    g_norm = g_t.norm(dim=-1).mean().item()
                    logger.debug(f"[Step {step}/{num_steps}] t={t_current:.3f} | Energy(Curr): {avg_energy:.4f} (min {min_e:.4f}) | |g_t|: {g_norm:.4f}")

            # --- Strategy B: Local SimMC (Recommended) ---
            elif guidance_strategy == 'local_sim_mc':
                # 1. Predict x0 from current state
                t_tensor = torch.full((B,), t_current, device=self.device)
                with torch.no_grad():
                    v_pred = self.eps_model(z_t, t_tensor, condition)
                
                # x0 = z_t - t * v
                x0_pred = z_t - t_current * v_pred
                
                # 2. Sample around predicted x0
                # (B, D) -> (B, N, D)
                noise_std = 0.1 # Hyperparameter for local exploration
                x0_candidates = x0_pred.unsqueeze(1) + torch.randn(B, mc_samples, z_t.shape[1], device=self.device) * noise_std
                
                # 3. Calculate Energy
                x0_flat = rearrange(x0_candidates, 'b n d -> (b n) d')
                
                energy = self._compute_physics_energy(self._denormalize_pose(x0_flat, data), data, w_erf, w_spf, w_srf, use_baseline_potential=use_baseline_potential)
                energy = rearrange(energy, '(b n) -> b n', b=B)
                
                # 4. Compute Guidance with Temperature
                min_E = energy.min(dim=1, keepdim=True).values
                weights = torch.exp(-(energy - min_E) / temperature)
                Z = weights.mean(dim=1, keepdim=True)
                
                # === 调试信息：每10步输出一次能量统计 ===
                if step % 10 == 0:
                    avg_energy = energy.mean().item()
                    min_e = energy.min().item()
                    max_e = energy.max().item()
                    energy_std = energy.std().item()
                    logger.debug(f"[Step {step}/{num_steps}] t={t_current:.3f} | Energy: min={min_e:.4f}, avg={avg_energy:.4f}, max={max_e:.4f}, std={energy_std:.6f}")
                
                # Conditional velocity: v = (z_t - x0) / t
                t_safe = max(t_current, 1e-3)
                v_cond = (z_t.unsqueeze(1) - x0_candidates) / t_safe
                
                term = (weights.unsqueeze(-1) / (Z.unsqueeze(-1) + 1e-8)) - 1.0
                
                # Scale Guidance
                g_t = (term * v_cond).mean(dim=1) * guidance_scale
                
                if step % 10 == 0:
                    g_norm = g_t.norm(dim=-1).mean().item()
                    logger.debug(f"    Guidance |g_t|: {g_norm:.6f} (scale={guidance_scale})")
            
            # --- Strategy C: SimMC-B (Improved Local SimMC with Spatial Weights) ---
            elif guidance_strategy == 'sim_mc_b':
                # 1. Predict x0 from current state
                t_tensor = torch.full((B,), t_current, device=self.device)
                with torch.no_grad():
                    v_pred = self.eps_model(z_t, t_tensor, condition)
                
                # x0 = z_t - t * v
                x0_pred = z_t - t_current * v_pred
                
                # 2. Sample around predicted x0
                noise_std = 0.1
                x0_candidates = x0_pred.unsqueeze(1) + torch.randn(B, mc_samples, z_t.shape[1], device=self.device) * noise_std
                
                # 3. Calculate Energy (Same as before)
                x0_flat = rearrange(x0_candidates, 'b n d -> (b n) d')
                energy = self._compute_physics_energy(self._denormalize_pose(x0_flat, data), data, w_erf, w_spf, w_srf, use_baseline_potential=use_baseline_potential)
                energy = rearrange(energy, '(b n) -> b n', b=B)
                
                # 4. Calculate Spatial Weights (p(z_t | x0))
                # z_t ~ N((1-t)x0, t^2 I)
                t_safe = max(t_current, 1e-3)
                std_t = t_safe
                
                mu = (1 - t_current) * x0_candidates # (B, N, D)
                diff = z_t.unsqueeze(1) - mu
                norm_sq = diff.square().sum(dim=-1) # (B, N)
                D_dim = z_t.shape[-1]
                log_p_t_given_x0 = -0.5 * norm_sq / (std_t**2 + 1e-6) - D_dim * math.log(std_t + 1e-6)
                
                # 5. Calculate Energy Weights in Log Space
                # Value v = -Energy
                v_support = -energy
                # Self-Normalize Energy
                v_mean = v_support.mean(dim=1, keepdim=True)
                v_std = v_support.std(dim=1, keepdim=True)
                v_norm = (v_support - v_mean) / (v_std + 1e-8)
                v_norm = v_norm.clamp(-5.0, 5.0)
                log_J = (1.0 / temperature) * v_norm
                
                # 6. Combine Weights (LogSumExp Trick)
                # log_p(z_t) = LogSumExp(log_p(z_t|x0)) - log(N)
                log_p_t = torch.logsumexp(log_p_t_given_x0, dim=1, keepdim=True) - math.log(mc_samples)
                
                # log_Z = LogSumExp(log_J + log_p(z_t|x0)) - log(N) - log_p(z_t)
                # Note: This is Z_combined
                log_integrand = log_J + log_p_t_given_x0
                log_Z = torch.logsumexp(log_integrand, dim=1, keepdim=True) - math.log(mc_samples) - log_p_t
                
                # 7. Compute Term: (J/Z - 1)
                log_ratio = log_J - log_Z
                term = torch.exp(log_ratio) - 1.0
                
                # 8. Spatial Importance Weights
                # w_spatial = p(z_t|x0) / p(z_t)
                log_w_spatial = log_p_t_given_x0 - log_p_t
                w_spatial = torch.exp(log_w_spatial)
                # Clip spatial weights to avoid single sample dominance
                # Stricter clamping to prevent weight explosion (ESS collapse)
                w_spatial = w_spatial.clamp(max=2.0)
                
                # 9. Conditional Velocity
                v_cond = (z_t.unsqueeze(1) - x0_candidates) / t_safe
                
                # 10. Final Guidance
                # g_t = E [ w_spatial * (J/Z - 1) * v_cond ]
                
                # Clip the term (J/Z - 1) to avoid explosion
                term = term.clamp(-5.0, 5.0)
                
                g_t_components = w_spatial.unsqueeze(-1) * term.unsqueeze(-1) * v_cond
                g_t = g_t_components.mean(dim=1) * guidance_scale
                
                # === Safety: Adaptive Gradient Rescaling ===
                # Scale guidance relative to the base flow velocity magnitude.
                # Strategy: g_t should not overpower v_model completely.
                # Formula: |g_t| <= beta * |v_model|
                
                # We already computed v_pred (v_model) at the beginning of this strategy
                v_base_norm = v_pred.norm(dim=-1, keepdim=True)
                
                # Beta factor: 0.5 means guidance is half as strong as the base flow
                # Min threshold: 0.5 ensures we still have some guidance even if v_model is tiny
                beta = 0.3
                min_guidance_norm = 0.5
                target_max_norm = torch.maximum(
                    torch.full_like(v_base_norm, min_guidance_norm),
                    beta * v_base_norm
                )
                
                g_norm_val = g_t.norm(dim=-1, keepdim=True)
                
                # If g_norm > target_max_norm, scale it down
                scale_factor = torch.minimum(
                    torch.ones_like(g_norm_val),
                    target_max_norm / (g_norm_val + 1e-6)
                )
                g_t = g_t * scale_factor

                
                if step % 10 == 0:

                    avg_energy = energy.mean().item()
                    min_e = energy.min().item()
                    g_norm = g_t.norm(dim=-1).mean().item()
                    logger.debug(f"[SimMC-B] t={t_current:.3f} | E_min: {min_e:.3f} | g_norm: {g_norm:.3f}")

            # --- Legacy: Search z (Algorithm 2) ---
            elif guidance_strategy == 'legacy':
                 # ... (Keep original logic) ...
                 # Copy paste the previous logic here or refactor
                 # For brevity, I will put the previous logic here
                 
                 z_t_expanded = repeat(z_t, 'b d -> (b n) d', n=mc_samples)
                 x_1_noise = torch.randn_like(z_t_expanded)
                 
                 if t_current < 0.99:
                     x_0_candidates = (z_t_expanded - t_current * x_1_noise) / (1.0 - t_current + 1e-8)
                 else:
                     x_0_candidates = z_t_expanded - x_1_noise

                 x_0_real = self._denormalize_pose(x_0_candidates, data)
                 energy = self._compute_physics_energy(x_0_real, data, w_erf, w_spf, w_srf)
                 
                 energy_reshaped = rearrange(energy, '(b n) -> b n', b=B, n=mc_samples)
                 min_energy = energy_reshaped.min(dim=1, keepdim=True).values
                 weights = torch.exp(-(energy_reshaped - min_energy))
                 Z_t = weights.mean(dim=1, keepdim=True)
                 
                 term = (weights / (Z_t + 1e-8)) - 1.0
                 v_conditional = x_1_noise - x_0_candidates
                 v_cond_reshaped = rearrange(v_conditional, '(b n) d -> b n d', b=B, n=mc_samples)
                 g_t = (term.unsqueeze(-1) * v_cond_reshaped).mean(dim=1)
                 
            # ==========================================
            # ===          Step Update               ===
            # ==========================================
            
            with torch.no_grad():
                t_current_tensor = torch.full((B,), t_current, device=self.device)
                v_model = self.eps_model(z_t, t_current_tensor, condition)
                
                # Add guidance
                # Note: We might want to scale g_t?
                # GFlower uses a scale factor. Here we assume scale=1.0 or controlled by w_weights?
                # The term (e^-J/Z - 1) naturally scales it.
                
                v_final = v_model + g_t
                
                # Logging energy stats occasionally
                if step % 10 == 0 and guidance_strategy != 'legacy':
                     # Calculate energy of the current "best guess" x0
                     # x0_curr = z_t - t_current * v_final # Not used for now
                     
                     v_model_norm = v_model.norm(dim=-1).mean().item()
                     logger.debug(f"    Base Flow |v_model|: {v_model_norm:.6f}")
                
                z_t = z_t - v_final * dt
        
        # === Save Final State at t=0 ===
        if save_intermediate and save_dir is not None:
            try:
                z_t_real = self._denormalize_pose(z_t, data)
                hand_model_vis = get_handmodel(batch_size=B, device=self.device)
                
                for batch_idx in range(B):
                    sample_dir = os.path.join(save_dir, str(batch_idx + 1))
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    meshes = hand_model_vis.get_meshes_from_q(q=z_t_real, i=batch_idx)
                    full_mesh = trimesh.util.concatenate(meshes)
                    
                    save_path = os.path.join(sample_dir, "hand_t_0.00.obj")
                    full_mesh.export(save_path)
                
                logger.info(f"✅ Final state saved at t=0.00 for {B} sample(s)")
            except Exception as e:
                logger.error(f"Failed to save final state: {e}")
                
        return z_t.unsqueeze(1)

    def _denormalize_pose(self, x_norm: torch.Tensor, data: Dict) -> torch.Tensor:
        """
        将归一化的姿态反归一化到真实空间，用于计算物理损失
        
        Args:
            x_norm: 归一化的姿态 (B, 27)
            data: 数据字典
        
        Return:
            完整的手姿态 (B, 33) = (trans:3, rot6d:6, joints:24)
        """
        B = x_norm.shape[0]
        
        # 调试：检查输入
        if torch.isnan(x_norm).any():
            logger.error(f"❌ [Denorm] NaN in x_norm input! Shape: {x_norm.shape}")
            logger.error(f"   NaN count: {torch.isnan(x_norm).sum()}, range: [{x_norm[~torch.isnan(x_norm)].min():.3f}, {x_norm[~torch.isnan(x_norm)].max():.3f}]")
        
        x_denorm = x_norm.clone()
        
        # 反归一化平移和关节角度（使用模型自己的硬编码参数）
        trans_denorm = self.trans_denormalize(x_norm[:, :3])
        angle_denorm = self.angle_denormalize(x_norm[:, 3:])
        
        # 调试：检查反归一化结果
        if torch.isnan(trans_denorm).any():
            logger.error(f"❌ [Denorm] NaN in trans_denormalize!")
        if torch.isnan(angle_denorm).any():
            logger.error(f"❌ [Denorm] NaN in angle_denormalize!")
        
        x_denorm[:, :3] = trans_denorm
        x_denorm[:, 3:] = angle_denorm
        
        # 构造完整姿态（加上6D旋转，这里用单位旋转）
        id_6d_rot = torch.tensor([1., 0., 0., 0., 1., 0.], 
                                 device=self.device).view(1, 6).repeat(B, 1)
        full_pose = torch.cat([x_denorm[:, :3], id_6d_rot, x_denorm[:, 3:]], dim=-1)
        
        if torch.isnan(full_pose).any():
            logger.error(f"❌ [Denorm] NaN in full_pose output!")
            logger.error(f"   Shape: {full_pose.shape}, NaN count: {torch.isnan(full_pose).sum()}")
        
        return full_pose
    
    def sample(self, data: Dict, k: int=1, num_steps: int=None, method: str='heun', use_guidance: bool=False,
              use_posthoc: bool=False, posthoc_learning_rate: float=0.003,
              posthoc_iterations: int=10,
              w_erf: float=1.0, w_spf: float=1.0, w_srf: float=1.0,
              mc_samples: int=100, guidance_strategy: str = 'local_sim_mc', **kwargs) -> torch.Tensor:
        """
        反向过程，通过给定的条件数据进行采样
        
        Args:
            data: 测试数据, data['x'] 给出目标数据的形状
            k: 采样的数量
            num_steps: ODE积分步数，如果为None则使用默认值
            method: ODE求解器 ('euler' 或 'heun')，默认heun
            use_guidance: 是否使用物理引导采样 (这里实际上使用的是MC引导)
            w_erf: ERF损失权重（手-物体穿透）
            w_spf: SPF损失权重（手指自碰撞）
            w_srf: SRF损失权重（手姿态自然性）
            mc_samples: MC引导的采样数 (N)
            guidance_strategy: 'legacy', 'global_support', 'local_sim_mc'
        
        Return:
            采样结果, 形状为 <B, k, T, ...>
        """
        # 使用配置中的步数，如果没有则使用默认值
        if use_guidance and use_posthoc:
            raise ValueError('trajectory guidance and terminal post-hoc refinement are mutually exclusive')
        if num_steps is None:
            if use_guidance:
                num_steps = self.cfg.get('num_sampling_steps_guided', 20)  # MC引导步数可以少一点
            else:
                num_steps = self.cfg.get('num_sampling_steps', 100)
        
        ksamples = []
        self.last_posthoc_stats = None
        for _ in range(k):
            if use_guidance:
                # 切换为使用 MC Guidance
                # 从 kwargs 中提取中间状态保存参数（如果存在）
                save_intermediate = kwargs.get('save_intermediate', False)
                save_dir = kwargs.get('save_dir', None)
                temperature = kwargs.get('temperature', 0.05)
                guidance_scale = kwargs.get('guidance_scale', 30.0)
                use_baseline_potential = kwargs.get('use_baseline_potential', False)
                
                final_sample = self.p_sample_loop_with_mc_guidance(
                    data, num_steps=num_steps, mc_samples=mc_samples,
                    w_erf=w_erf, w_spf=w_spf, w_srf=w_srf,
                    method=method, guidance_strategy=guidance_strategy,
                    temperature=temperature, guidance_scale=guidance_scale,
                    use_baseline_potential=use_baseline_potential,
                    save_intermediate=save_intermediate, save_dir=save_dir
                )
            elif use_posthoc:
                final_sample = self.p_sample_loop_with_posthoc(
                    data,
                    num_steps=num_steps,
                    method=method,
                    learning_rate=posthoc_learning_rate,
                    iterations=posthoc_iterations,
                    w_erf=w_erf,
                    w_spf=w_spf,
                    w_srf=w_srf,
                )
            else:
                final_sample = self.p_sample_loop(data, num_steps=num_steps, method=method)
            ksamples.append(final_sample)
        
        ksamples = torch.stack(ksamples, dim=1)
        
        # 反归一化
        if 'normalizer' in data and data['normalizer'] is not None:
            O = 0
            if self.has_observation and 'start' in data:
                _, O, _ = data['start'].shape
            ksamples[..., -1, :] = data['normalizer'].unnormalize(ksamples[..., -1, :])
        
        # 处理相对/绝对表示
        if 'repr_type' in data:
            if data['repr_type'] == 'absolute':
                pass
            elif data['repr_type'] == 'relative':
                O = 1
                if self.has_observation and 'start' in data:
                    _, O, _ = data['start'].shape
                ksamples[..., O-1:, :] = torch.cumsum(ksamples[..., O-1:, :], dim=-2)
            else:
                raise Exception('不支持的 repr 类型。')
        
        return ksamples
    
    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
    
    def angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_upper = self._joint_angle_upper
        joint_angle_lower = self._joint_angle_lower
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (joint_angle_upper - joint_angle_lower) + joint_angle_lower
        return joint_angle_denorm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm
