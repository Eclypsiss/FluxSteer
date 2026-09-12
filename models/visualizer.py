import os
import json
import csv
from time import perf_counter
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import trimesh
import pickle
from omegaconf import DictConfig
from plotly import graph_objects as go
from typing import Any
import random
from utils.registry import Registry
from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh
from utils.rot6d import rot_to_orthod6d, robust_compute_rotation_matrix_from_ortho6d, random_rot  
from tqdm import tqdm

VISUALIZER = Registry('Visualizer')
@VISUALIZER.register()
@torch.no_grad()
class GraspGenURVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.
        Args:
            cfg: visuzalizer configuration
        """
        self.cfg = cfg  # 保存完整配置以访问temperature等参数
        self.ksample = cfg.ksample
        self.hand_model = get_handmodel(batch_size=1, device='cuda')
        self.use_llm = cfg.use_llm
        self.visualize_html = cfg.visualize_html
        self.datasetname = cfg.datasetname
        self.object_list_file = cfg.get('object_list_file', None)
        self.object_name = cfg.get('object_name', None)
        self.record_inference_times = cfg.get('record_inference_times', False)
        self.runtime_warmup = cfg.get('runtime_warmup', False)
        self.experiment_method = cfg.get('experiment_method', None)
        self.experiment_config_id = cfg.get('experiment_config_id', None)
        ##############################################################################################################################
        ##### Since other datasets have object scale=1, for testing convenience we use average scales from DexGraspNet and UniDexGrasp test sets #####
        ##### To test with different mesh sizes, please adjust the mesh scale accordingly for proper visualization #####
        ##############################################################################################################################
        # ✅ 修复：动态加载scales.pkl以正确处理点云缩放
        if self.datasetname == 'DexGraspNet':
            scales_path = os.path.join(cfg.asset_dir, 'scales.pkl')
            if os.path.exists(scales_path):
                self.average_scales = self.load_average_scales(scales_path)
                print(f"✅ Loaded scales for DexGraspNet from {scales_path}")
            else:
                self.average_scales = {}
                print(f"⚠️  scales.pkl not found for DexGraspNet, using empty scales")
        elif self.datasetname == 'Unidexgrasp':
            scales_path = os.path.join(cfg.asset_dir, 'scales.pkl')
            if os.path.exists(scales_path):
                self.average_scales = self.load_average_scales(scales_path)
                print(f"✅ Loaded scales for UniDexGrasp from {scales_path}")
            else:
                self.average_scales = {}
                print(f"⚠️  scales.pkl not found for UniDexGrasp, using empty scales")
        else:
            self.average_scales = {} # Other datasets don't need scales
        ##############################################################################################################################
        ##############################################################################################################################
    def load_average_scales(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    def visualize(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            save_dir: str
    ) -> None:
        """ Visualize method
        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        model.eval()
        device = model.device

        os.makedirs(save_dir, exist_ok=True)
        if self.visualize_html:
            os.makedirs(os.path.join(save_dir, 'html'), exist_ok=True)
        # Getting descriptions from LLM
        if self.use_llm:
            self.scene_text = dataloader.dataset.scene_text
        objects = list(dataloader.dataset._test_split)
        if self.object_list_file:
            object_list_path = os.path.abspath(self.object_list_file)
            with open(object_list_path, encoding='utf-8') as handle:
                objects = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith('#')]
            missing_objects = [name for name in objects if name not in dataloader.dataset.scene_pcds]
            if missing_objects:
                raise ValueError(f'Object list contains missing point clouds: {missing_objects[:5]}')

        if self.object_name:
            if self.object_name not in dataloader.dataset.scene_pcds:
                raise ValueError(f'对象点云不存在：{self.object_name}')
            objects = [self.object_name]
        if not objects or len(objects) != len(set(objects)):
            raise ValueError('对象列表不能为空或包含重复 ID')

        pbar = tqdm(total=len(objects) * self.ksample)
        object_pcds_dict = dataloader.dataset.scene_pcds
        timing_path = os.path.join(save_dir, 'inference_times.csv')
        timing_fields = [
            'object_id', 'method', 'config_id', 'candidate_count',
            'generation_time_sec', 'refinement_time_sec', 'total_inference_time_sec',
            'initial_energy_mean', 'final_energy_mean', 'invalid_candidate_count',
            'clamped_initial_candidate_count', 'energy_evaluations',
        ]
        if self.record_inference_times:
            with open(timing_path, 'w', newline='', encoding='utf-8') as handle:
                csv.DictWriter(handle, fieldnames=timing_fields).writeheader()
        runtime_warmed_up = False
        res = {'method': self.experiment_method or 'FluxSteer',
               'desc': 'ShadowHand grasp candidates: translation 3 + rotation 6 + joints 24',
               'sample_qpos': {}}
        # Define dataset configuration dictionary
        DATASET_CONFIG = {
            "DexGraspNet": {
                "scale_op": lambda x, s: x * s,
                "objects": objects
            },
            "Unidexgrasp": {
                "scale_op": lambda x, s: x / s,
                "objects": objects
            },
            "default": {
                "scale_op": lambda x, s: x,
                "objects": objects
            }
        }

        # Get current dataset configuration
        cfg = DATASET_CONFIG.get(
            dataloader.dataset.datasetname, 
            DATASET_CONFIG["default"]
        )

        # Unified processing flow
        for object_name in cfg["objects"]:
            # Point cloud scaling processing
            scale = self.average_scales.get(object_name, 1.0)  # Get average scale for object, default to 1.0
            obj_pcd_can = cfg["scale_op"](
                torch.tensor(object_pcds_dict[object_name], device=device).unsqueeze(0).repeat(self.ksample, 1, 1),
                scale
            )
            obj_pcd_nor = obj_pcd_can[:, :dataloader.dataset.num_points, 3:]
            obj_pcd_can = obj_pcd_can[:, :dataloader.dataset.num_points, :3]
            i_rot_list = []
            for k_rot in range(self.ksample):
                i_rot_list.append(random_rot(device))
            i_rot = torch.stack(i_rot_list).to(torch.float64)
            obj_pcd_rot = torch.matmul(i_rot, obj_pcd_can.transpose(1, 2)).transpose(1, 2)
            obj_pcd_nor_rot = torch.matmul(i_rot, obj_pcd_nor.transpose(1, 2)).transpose(1, 2)
            
            all_sentence = []
            for n in range(self.ksample):
                if self.use_llm:
                    all_sentence.extend(self.scene_text[object_name])               
                
            # construct data
            data = {'x': torch.randn(self.ksample, 27, device=device),
                    'pos': obj_pcd_rot.to(device),
                    'normal':obj_pcd_nor_rot.to(device),
                    'feat':obj_pcd_nor_rot.to(device),
                    'scene_rot_mat': i_rot,
                    'scene_id': [object_name for i in range(self.ksample)],
                    'cam_trans': [None for i in range(self.ksample)],
                    'text': (all_sentence if self.use_llm else None),
                    'sentence_cnt': ([len(self.scene_text[object_name])] * self.ksample if self.use_llm else None)}
            scene_model_name = getattr(model, 'scene_model_name', None)
            if scene_model_name is None and hasattr(model, 'eps_model'):
                scene_model_name = getattr(model.eps_model, 'scene_model_name', None)
            if scene_model_name == 'PointTransformer':
                offset, count = [], 0
                for item in data['pos']:
                    count += item.shape[0]
                    offset.append(count)
                offset = torch.IntTensor(offset)
                data['offset'] = offset.to(device)
                data['pos'] = rearrange(data['pos'], 'b n c -> (b n) c').to(device)
                data['feat'] = rearrange(data['feat'], 'b n c -> (b n) c').to(device)
            
            # Temperature Scaling支持（仅Flow Matching）
            model_class_name = model.__class__.__name__

            def sample_once():
                if model_class_name == 'FlowMatching' or 'Flow' in model_class_name:
                    temperature = self.cfg.get('temperature', 1.0)
                    return model.sample(data, k=1, temperature=temperature)
                return model.sample(data, k=1)

            if self.record_inference_times and self.runtime_warmup and not runtime_warmed_up:
                cpu_rng_state = torch.random.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state(device)
                python_rng_state = random.getstate()
                numpy_rng_state = np.random.get_state()
                warmup_output = sample_once()
                torch.cuda.synchronize(device)
                del warmup_output
                torch.random.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state(cuda_rng_state, device)
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                runtime_warmed_up = True

            if self.record_inference_times:
                torch.cuda.synchronize(device)
                sample_started = perf_counter()
                sampled = sample_once()
                torch.cuda.synchronize(device)
                total_inference_time = perf_counter() - sample_started
            else:
                sampled = sample_once()
                total_inference_time = None

            outputs = sampled.squeeze(1)[:, -1, :].to(torch.float64)

            if self.record_inference_times:
                posthoc_stats = getattr(model, 'last_posthoc_stats', None) or {}
                refinement_time = float(posthoc_stats.get('refinement_time_sec', 0.0))
                timing_row = {
                    'object_id': object_name,
                    'method': self.experiment_method or model_class_name,
                    'config_id': self.experiment_config_id or '',
                    'candidate_count': outputs.shape[0],
                    'generation_time_sec': max(total_inference_time - refinement_time, 0.0),
                    'refinement_time_sec': refinement_time,
                    'total_inference_time_sec': total_inference_time,
                    'initial_energy_mean': posthoc_stats.get('initial_energy_mean', ''),
                    'final_energy_mean': posthoc_stats.get('final_energy_mean', ''),
                    'invalid_candidate_count': posthoc_stats.get('invalid_candidate_count', 0),
                    'clamped_initial_candidate_count': posthoc_stats.get('clamped_initial_candidate_count', 0),
                    'energy_evaluations': posthoc_stats.get('energy_evaluations', 0),
                }
                with open(timing_path, 'a', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=timing_fields)
                    writer.writerow(timing_row)
            
            ## denormalization
            if dataloader.dataset.normalize_x:
                outputs[:, 3:] = dataloader.dataset.angle_denormalize(joint_angle=outputs[:, 3:].cpu()).to(device)
            if dataloader.dataset.normalize_x_trans:
                outputs[:, :3] = dataloader.dataset.trans_denormalize(global_trans=outputs[:, :3].cpu()).to(device)
            
            id_6d_rot = torch.tensor([1., 0., 0., 0., 1., 0.], device=device).view(1, 6).repeat(self.ksample, 1).to(torch.float64)
            outputs_3d_rot = rot_to_orthod6d(torch.bmm(i_rot.transpose(1, 2), robust_compute_rotation_matrix_from_ortho6d(id_6d_rot)))
            outputs[:, :3] = torch.bmm(i_rot.transpose(1, 2), outputs[:, :3].unsqueeze(-1)).squeeze(-1)
            
            outputs = torch.cat([outputs[:, :3], outputs_3d_rot, outputs[:, 3:]], dim=-1)
            if outputs.shape != (self.ksample, 33) or not torch.isfinite(outputs).all():
                raise FloatingPointError(f'对象 {object_name} 的生成结果不是有限的 {self.ksample}×33 张量')

            # visualization for checking
            scene_id = data['scene_id'][0]
            if dataloader.dataset.datasetname == 'MultiDexShadowHandUR' and self.visualize_html:
                scene_dataset, scene_object = scene_id.split('+')
                mesh_path = os.path.join(dataloader.dataset.asset_dir,'object', scene_dataset, scene_object, f'{scene_object}.stl')
                obj_mesh = trimesh.load(mesh_path)

            elif dataloader.dataset.datasetname == 'real_dex'and self.visualize_html:
                scene_object = scene_id
                mesh_path = os.path.join(dataloader.dataset.asset_dir,'meshdata', f'{scene_object}.obj')
                obj_mesh = trimesh.load(mesh_path)

            elif dataloader.dataset.datasetname == 'Unidexgrasp'and self.visualize_html:
                scene_object = scene_id
                mesh_path = os.path.join(dataloader.dataset.asset_dir,'obj_scale_urdf', f'{scene_object}.obj')
                obj_mesh = trimesh.load(mesh_path)

            elif dataloader.dataset.datasetname == 'DexGraspNet'and self.visualize_html:
                scene_object = scene_id
                mesh_path = os.path.join(dataloader.dataset.asset_dir,'obj_scale_urdf', f'{scene_object}.obj')
                obj_mesh = trimesh.load(mesh_path)

            elif dataloader.dataset.datasetname == 'DexGRAB'and self.visualize_html:
                scene_object = scene_id
                mesh_path = os.path.join(dataloader.dataset.asset_dir,'contact_meshes', f'{scene_object}.ply')
                obj_mesh = trimesh.load(mesh_path)

            for i in range(outputs.shape[0]):
                if self.visualize_html:
                    self.hand_model.update_kinematics(q=outputs[i:i+1, :])
                    vis_data = [plot_mesh(obj_mesh, color='lightpink')]
                    
                    vis_data += self.hand_model.get_plotly_data(opacity=1.0, color='#8799C6')
                    # Save as HTML file
                    save_path = os.path.join(save_dir, 'html', f'{object_name}+sample-{i}.html')
                    fig = go.Figure(data=vis_data)
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),
                            zaxis=dict(visible=False),
                            bgcolor="white"
                        )
                    )
                    fig.write_html(save_path)
                pbar.update(1)
            res['sample_qpos'][object_name] = np.array(outputs.cpu().detach())
        pickle.dump(res, open(os.path.join(save_dir, 'samples.pkl'), 'wb'))
        pbar.close()

def create_visualizer(cfg: DictConfig) -> nn.Module:
    """ Create a visualizer for visual evaluation
    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A visualizer
    """
    return VISUALIZER.get(cfg.name)(cfg)
