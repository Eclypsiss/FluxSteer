import os
import sys
import warnings
import csv
import json
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.getcwd())

import gc
import yaml
import pickle
import argparse
from loguru import logger

from isaacgym import gymapi, gymutil, gymtorch
import torch
import random
import numpy as np

import trimesh as tm
# from pyvirtualdisplay import Display
from utils.handmodel import get_handmodel, compute_collision
from envs.tasks.grasp_test_force_shadowhand import IsaacGraspTestForce_shadowhand as IsaacGraspTestForce
from engine.assets import find_object_assets, normals_cache

# 屏蔽trimesh和其他库的烦人警告
warnings.filterwarnings('ignore', message='.*concatenating texture.*')
warnings.filterwarnings('ignore', category=UserWarning, module='trimesh')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def set_global_seed(seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='FluxSteer MAIN stability、diversity 和 collision 评估')
    parser.add_argument('--stability-config', dest='stability_config', type=str,
                        default='envs/tasks/grasp_test_force.yaml',
                        help='stability config file path')
    parser.add_argument('--eval-dir', dest='eval_dir', type=str, required=True,
                        help='evaluation directory path (e.g.,\
                             "outputs/2022-11-15_18-07-50_GPUR_l1_pn2_T100/eval/final/2023-04-20_13-06-44")')
    parser.add_argument('--dataset', choices=['dexgraspnet', 'unidexgrasp', 'dexgrab', 'multidex', 'realdex'], required=True)
    parser.add_argument('--data-root', type=str, default=os.environ.get('FLUXSTEER_DATA_ROOT', 'data'))
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu device id for simulation')
    parser.add_argument('--cpu', action='store_true', default=False, 
                        help='run all on cpu')
    parser.add_argument('--onscreen', action='store_true', default=False,
                        help='run simulator onscreen')
    parser.add_argument('--object', dest='object_name', type=str, default=None,
                        help='specific object name to test (optional)')
    
    return parser.parse_args()


def get_sim_param():
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = 0
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.physx.num_subscenes = 0
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.num_threads = 0
    return sim_params


def stability_tester(args: argparse.Namespace) -> dict:
    stability_config_path = args.stability_config
    if not os.path.isabs(stability_config_path):
        stability_config_path = os.path.join(project_root, stability_config_path)

    with open(stability_config_path) as f:
        stability_config = yaml.safe_load(f)
    
    # 💡 传入dataset_dir，让Isaac Gym环境能找到mesh
    stability_config['dataset_dir'] = args.dataset_dir
    
    sim_params = get_sim_param()
    sim_headless = not args.onscreen

    # load generated grasp results here
    grasps = pickle.load(open(args.sample_file, 'rb'))
    isaac_env = None
    results = {}
    across_all_cases = 0
    across_all_succ = 0
    results1 = {}
    across_all_cases1 = 0
    across_all_succ1 = 0

    # Filter objects if object_name is specified
    object_list = list(grasps['sample_qpos'].keys())
    if args.object_name:
        if args.object_name in object_list:
            object_list = [args.object_name]
        else:
            logger.warning(f"Object {args.object_name} not found in generated grasps.")
            return {}, {}, {'suc6_count': 0, 'suc6_total': 0, 'suc1_count': 0, 'suc1_total': 0}

    for object_name in object_list:
        # Skip problematic objects that cause SegFault
        if object_name == 'body_lotion':
             logger.warning(f"Skipping {object_name} to avoid SegFault.")
             args.evaluation_errors[object_name] = 'known simulator asset failure'
             continue

        logger.info(f'Stability test for [{object_name}]')
        q_generated = grasps['sample_qpos'][object_name]
        q_generated = torch.tensor(q_generated, device=args.device).to(torch.float32)

        try:
            _, object_mesh_path = find_object_assets(args.dataset_dir, object_name)
        except FileNotFoundError as error:
            args.evaluation_errors[object_name] = str(error)
            logger.error(str(error))
            continue

        object_mesh = tm.load(object_mesh_path)
        object_volume = object_mesh.volume
        
        # ✅ 添加异常捕获：跳过加载失败的物体，继续测试其他物体
        try:
            isaac_env = IsaacGraspTestForce(stability_config, sim_params, gymapi.SIM_PHYSX, 
                                            args.device, args.gpu, headless=sim_headless, init_opt_q=q_generated,
                                            object_name=object_name, object_volume=object_volume, fix_object=False)
            succ_grasp_object ,succ_grasp_object1 = isaac_env.push_object()
            results[object_name] = {'total': int(succ_grasp_object.shape[0]),
                                    'succ': int(succ_grasp_object.sum()),
                                    'case_list': succ_grasp_object.tolist()}
            results1[object_name] = {'total': int(succ_grasp_object1.shape[0]),
                            'succ': int(succ_grasp_object1.sum()),
                            'case_list': succ_grasp_object1.tolist()}
            logger.info(f'all 6dir Success rate of [{object_name}]: {int(succ_grasp_object.sum())} / {int(succ_grasp_object.shape[0])} ({(succ_grasp_object.sum() / succ_grasp_object.shape[0]) * 100:.2f}%)')
            logger.info(f'one 6dir Success rate of [{object_name}]: {int(succ_grasp_object1.sum())} / {int(succ_grasp_object1.shape[0])} ({(succ_grasp_object1.sum() / succ_grasp_object1.shape[0]) * 100:.2f}%)')
        except Exception as e:
            logger.error(f'❌ Failed to test object "{object_name}": {str(e)}')
            logger.warning(f'⏭️  Skipping this object and continuing with remaining tests...')
            args.evaluation_errors[object_name] = str(e)
            results.pop(object_name, None)
            results1.pop(object_name, None)
            if isaac_env is not None:
                del isaac_env
                isaac_env = None
                gc.collect()
            continue
        across_all_succ += int(succ_grasp_object.sum())
        across_all_cases += int(succ_grasp_object.shape[0])
        across_all_succ1 += int(succ_grasp_object1.sum())
        across_all_cases1 += int(succ_grasp_object1.shape[0])
        if isaac_env is not None:
            del isaac_env
            isaac_env = None
            gc.collect()
    if across_all_cases == 0 or across_all_cases1 == 0:
        raise RuntimeError('stability evaluation produced no valid cases')
    logger.info(f'**all 6dir Success Rate** across all objects: {across_all_succ} / {across_all_cases} ({(across_all_succ / across_all_cases) * 100:.2f}%)')
    logger.info(f'**one 6dir Success Rate** across all objects: {across_all_succ1} / {across_all_cases1} ({(across_all_succ1 / across_all_cases1) * 100:.2f}%')
    
    return results, results1, {
        'suc6_count': across_all_succ,
        'suc6_total': across_all_cases,
        'suc1_count': across_all_succ1,
        'suc1_total': across_all_cases1,
    }


def diversity_tester(args: argparse.Namespace, stability_results: dict) -> None:    
    grasps = pickle.load(open(args.sample_file, 'rb'))

    qpos_std = []
    object_diversity = {}
    skipped_objects = []
    
    for object_name in grasps['sample_qpos'].keys():
        # Check if object was tested in stability_tester
        if object_name not in stability_results:
            skipped_objects.append(object_name)
            continue
            
        i_qpos = grasps['sample_qpos'][object_name][:, 9:]
        i_qpos = i_qpos[stability_results[object_name]['case_list'], :]
        if i_qpos.shape[0]:
            i_qpos = np.sqrt(i_qpos.var(axis=0))
            qpos_std.append(i_qpos)
            object_diversity[object_name] = float(i_qpos.mean())
        else:
            object_diversity[object_name] = None

    if skipped_objects:
        logger.warning(f"Skipped diversity test for {len(skipped_objects)} objects (missing in stability results).")

    if not qpos_std:
        logger.warning("No successful grasps found for diversity calculation.")
        return object_diversity, None

    qpos_std = np.stack(qpos_std, axis=0)
    aggregate_diversity = float(qpos_std.mean(axis=0).mean())
    logger.info(f'**Diversity** (std: rad.) across all success grasps: {aggregate_diversity}')
    return object_diversity, aggregate_diversity


def collision_tester(args: argparse.Namespace, stability_results: dict) -> None:
    _BATCHSIZE = 8 #NOTE: adjust this batchsize to fit your GPU memory && need to be divided by generated grasps per object
    _NPOINTS = 4096 #NOTE: number of surface points sampled from a object

    grasps = pickle.load(open(args.sample_file, 'rb'))
    pkl_path = normals_cache(args.dataset_dir)
    logger.info(f'Loading point cloud normals from: {pkl_path}')
    obj_pcds_nors_dict = pickle.load(open(pkl_path, 'rb'))
    
    hand_model = get_handmodel(batch_size=_BATCHSIZE, device=args.device)

    collisions_dict = {obj: [] for obj in grasps['sample_qpos'].keys()}
    collisions_dict2 = {obj: [] for obj in grasps['sample_qpos'].keys()}
    object_collision_success = {}
    object_collision_all = {}
    
    skipped_objects = []
    missing_pcd_objects = []

    for object_name in grasps['sample_qpos'].keys():
        if object_name not in stability_results:
            skipped_objects.append(object_name)
            continue

        qpos = grasps['sample_qpos'][object_name]
        
        # 尝试多种方式查找物体点云（兼容不同命名格式）
        obj_pcd_nor = None
        
        # 方式1: 直接匹配
        if object_name in obj_pcds_nors_dict:
            obj_pcd_nor = obj_pcds_nors_dict[object_name]
        else:
            # 方式2: 尝试添加常见前缀
            for prefix in ['contactdb+', 'ycb+', 'dexycb+', '']:
                key = f'{prefix}{object_name}'
                if key in obj_pcds_nors_dict:
                    obj_pcd_nor = obj_pcds_nors_dict[key]
                    # logger.info(f'Found point cloud for "{object_name}" as "{key}"')
                    break
            
            # 方式3: 模糊匹配（后缀匹配）
            if obj_pcd_nor is None:
                for key in obj_pcds_nors_dict.keys():
                    if key.endswith(object_name) or object_name in key:
                        obj_pcd_nor = obj_pcds_nors_dict[key]
                        # logger.info(f'Found point cloud for "{object_name}" via fuzzy match: "{key}"')
                        break
        
        # 如果还是找不到，跳过
        if obj_pcd_nor is None:
            missing_pcd_objects.append(object_name)
            # 将空列表转为空数组，避免后续.size报错
            collisions_dict[object_name] = np.array([])
            collisions_dict2[object_name] = np.array([])
            continue
        
        obj_pcd_nor = obj_pcd_nor[:_NPOINTS, :]
        for i in range(qpos.shape[0] // _BATCHSIZE):
            i_qpos = qpos[i * _BATCHSIZE: (i + 1) * _BATCHSIZE, :]
            hand_model.update_kinematics(q=torch.tensor(i_qpos, device=args.device).float())
            hand_surface_points = hand_model.get_surface_points()
            depth_collision = compute_collision(torch.tensor(obj_pcd_nor, device=args.device), hand_surface_points)
            collisions_dict[object_name].append(np.array(depth_collision.cpu()[stability_results[object_name]['case_list'][i * _BATCHSIZE : (i + 1) * _BATCHSIZE]]))
            collisions_dict2[object_name].append(np.array(depth_collision.cpu()))
        
        # 只有当列表非空时才进行concatenate
        if collisions_dict[object_name]:
            collisions_dict[object_name] = np.concatenate(collisions_dict[object_name], axis=0)
        else:
            collisions_dict[object_name] = np.array([])
        
        if collisions_dict2[object_name]:
            collisions_dict2[object_name] = np.concatenate(collisions_dict2[object_name], axis=0)
        else:
            collisions_dict2[object_name] = np.array([])

        object_collision_success[object_name] = (
            float(collisions_dict[object_name].mean() * 1e3)
            if collisions_dict[object_name].size else None
        )
        object_collision_all[object_name] = (
            float(collisions_dict2[object_name].mean() * 1e3)
            if collisions_dict2[object_name].size else None
        )
    
    if skipped_objects:
        logger.warning(f"Skipped collision test for {len(skipped_objects)} objects (missing in stability results).")
    
    if missing_pcd_objects:
        logger.warning(f"Skipped collision test for {len(missing_pcd_objects)} objects (missing point clouds).")

    # 过滤掉空的collision结果
    valid_collisions = [collisions_dict[obj] for obj in grasps['sample_qpos'].keys() 
                        if isinstance(collisions_dict[obj], np.ndarray) and collisions_dict[obj].size > 0]
    valid_collisions2 = [collisions_dict2[obj] for obj in grasps['sample_qpos'].keys() 
                         if isinstance(collisions_dict2[obj], np.ndarray) and collisions_dict2[obj].size > 0]

    if valid_collisions:
        collision_values = np.concatenate(valid_collisions, axis=0)
        aggregate_collision_success = float(collision_values.mean() * 1e3)
        logger.info(f'**Collision** (depth: mm.) across succ grasps: {aggregate_collision_success}')
    else:
        aggregate_collision_success = None
        logger.warning("No valid collision data found for successful grasps.")

    if valid_collisions2:
        collision_values2 = np.concatenate(valid_collisions2, axis=0)
        aggregate_collision_all = float(collision_values2.mean() * 1e3)
        logger.info(f'**Collision** (depth: mm.) across all grasps: {aggregate_collision_all}')
    else:
        aggregate_collision_all = None
        logger.warning("No valid collision data found for any grasps.")

    return object_collision_success, object_collision_all, aggregate_collision_success, aggregate_collision_all


def _read_inference_times(eval_dir: str) -> dict:
    timing_path = os.path.join(eval_dir, 'inference_times.csv')
    if not os.path.isfile(timing_path):
        return {}
    with open(timing_path, newline='', encoding='utf-8') as handle:
        return {row['object_id']: row for row in csv.DictReader(handle)}


def _optional_float(value):
    if value is None or value == '':
        return None
    return float(value)


def write_structured_results(args, stability_results, stability_results1,
                             stability_summary, object_diversity, aggregate_diversity,
                             object_collision_success, object_collision_all,
                             aggregate_collision_success, aggregate_collision_all):
    grasps = pickle.load(open(args.sample_file, 'rb'))
    timing = _read_inference_times(args.eval_dir)
    rows = []
    for object_name, candidates in grasps['sample_qpos'].items():
        suc6 = stability_results.get(object_name)
        suc1 = stability_results1.get(object_name)
        timing_row = timing.get(object_name, {})
        rows.append({
            'object_id': object_name,
            'evaluation_status': 'evaluated' if suc6 else 'skipped',
            'evaluation_error': args.evaluation_errors.get(object_name, ''),
            'method': timing_row.get('method') or grasps.get('method', ''),
            'config_id': timing_row.get('config_id', ''),
            'candidate_count': int(len(candidates)),
            'suc6_count': suc6['succ'] if suc6 else None,
            'suc1_count': suc1['succ'] if suc1 else None,
            'suc6_rate': (suc6['succ'] / suc6['total']) if suc6 and suc6['total'] else None,
            'suc1_rate': (suc1['succ'] / suc1['total']) if suc1 and suc1['total'] else None,
            'penetration_success_mm': object_collision_success.get(object_name),
            'penetration_all_mm': object_collision_all.get(object_name),
            'diversity_rad': object_diversity.get(object_name),
            'generation_time_sec': _optional_float(timing_row.get('generation_time_sec')),
            'refinement_time_sec': _optional_float(timing_row.get('refinement_time_sec')),
            'total_inference_time_sec': _optional_float(timing_row.get('total_inference_time_sec')),
            'initial_energy_mean': _optional_float(timing_row.get('initial_energy_mean')),
            'final_energy_mean': _optional_float(timing_row.get('final_energy_mean')),
            'invalid_candidate_count': int(timing_row.get('invalid_candidate_count') or 0),
            'clamped_initial_candidate_count': int(timing_row.get('clamped_initial_candidate_count') or 0),
            'energy_evaluations': int(timing_row.get('energy_evaluations') or 0),
        })

    per_object_path = os.path.join(args.eval_dir, 'per_object_metrics.csv')
    with open(per_object_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def mean_timing(field):
        values = [row[field] for row in rows if row[field] is not None]
        return float(np.mean(values)) if values else None

    summary = {
        'method': rows[0]['method'],
        'config_id': rows[0]['config_id'],
        'objects': len(rows),
        'objects_evaluated': len(stability_results),
        'objects_skipped': len(rows) - len(stability_results),
        'skipped_objects': args.evaluation_errors,
        'candidates': int(sum(row['candidate_count'] for row in rows)),
        'Suc6_count': stability_summary['suc6_count'],
        'Suc6_total': stability_summary['suc6_total'],
        'Suc6_percent': (
            100.0 * stability_summary['suc6_count'] / stability_summary['suc6_total']
            if stability_summary['suc6_total'] else None
        ),
        'Suc1_count': stability_summary['suc1_count'],
        'Suc1_total': stability_summary['suc1_total'],
        'Suc1_percent': (
            100.0 * stability_summary['suc1_count'] / stability_summary['suc1_total']
            if stability_summary['suc1_total'] else None
        ),
        'Pen_success_mm': aggregate_collision_success,
        'Pen_all_mm': aggregate_collision_all,
        'Div_rad': aggregate_diversity,
        'generation_time_sec_per_object': mean_timing('generation_time_sec'),
        'refinement_time_sec_per_object': mean_timing('refinement_time_sec'),
        'total_inference_time_sec_per_object': mean_timing('total_inference_time_sec'),
        'evaluation_seed': args.seed,
        'evaluator': 'engine/evaluate.py',
    }
    with open(os.path.join(args.eval_dir, 'summary_metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    logger.info(f'Structured per-object metrics: {per_object_path}')


def main() -> None:
    args = parse_args()

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    os.environ['FLUXSTEER_DATA_ROOT'] = str(Path(args.data_root).resolve())
    with initialize_config_dir(version_base=None, config_dir=str(Path(project_root) / 'configs')):
        cfg = compose(config_name='default', overrides=[f'dataset={args.dataset}'])
        args.dataset_dir = str(Path(OmegaConf.to_container(cfg.dataset, resolve=True)['asset_dir']).resolve())
    sample_file = Path(args.eval_dir) / 'samples.pkl'
    if not sample_file.is_file():
        sample_file = Path(args.eval_dir) / 'res_diffuser.pkl'
    if not sample_file.is_file():
        raise FileNotFoundError(f'未找到 samples.pkl：{args.eval_dir}')
    args.sample_file = str(sample_file)
    with sample_file.open('rb') as handle:
        candidates = pickle.load(handle)['sample_qpos']
    if not candidates or any(len(poses) == 0 or len(poses) % 8 for poses in candidates.values()):
        raise ValueError('MAIN collision 要求每个对象的候选数为正的 8 的倍数')
    args.evaluation_errors = {}

    set_global_seed(args.seed)
    args.device = f'cuda:{args.gpu}' if not args.cpu else 'cpu'

    logger.add(args.eval_dir + '/evaluation.log')
    logger.info(f'Evaluation directory: {args.eval_dir}')

    logger.info('Start evaluating..')

    stability_results, stability_results1, stability_summary = stability_tester(args)
    object_diversity, aggregate_diversity = diversity_tester(args, stability_results)
    (object_collision_success, object_collision_all,
     aggregate_collision_success, aggregate_collision_all) = collision_tester(args, stability_results)
    write_structured_results(
        args,
        stability_results,
        stability_results1,
        stability_summary,
        object_diversity,
        aggregate_diversity,
        object_collision_success,
        object_collision_all,
        aggregate_collision_success,
        aggregate_collision_all,
    )
    
    logger.info('End evaluating..')


if __name__ == '__main__':
    main()
