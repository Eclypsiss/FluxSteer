"""使用已选数据集配置生成 FluxSteer 抓取，并记录实际推理参数。"""

from functools import partial
import os
from pathlib import Path
import random

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from engine.checkpoint import load_weights
from models.base import create_model
from models.visualizer import create_visualizer
from utils.misc import compute_model_dim


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(int(cfg.gpu))
    device = torch.device("cuda", int(cfg.gpu))
    cfg.model.d_x = compute_model_dim(cfg.task)
    run_dir = Path(cfg.exp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "samples.pkl").exists():
        raise FileExistsError(f"采样结果已存在：{run_dir / 'samples.pkl'}")
    logger.add(str(run_dir / "sample.log"))
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml", resolve=True)
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    os.environ["FLUXSTEER_DATASET_DIR"] = str(Path(cfg.task.dataset.asset_dir).resolve())
    dataset = create_dataset(cfg.task.dataset, "test", cfg.slurm, case_only=True)
    collate = (collate_fn_squeeze_pcd_batch if cfg.model.scene_model.name == "PointTransformer"
               else collate_fn_general)
    loader = DataLoader(dataset, batch_size=cfg.task.test.batch_size, num_workers=0,
                        collate_fn=partial(collate, use_llm=cfg.model.use_llm), shuffle=False)
    model = create_model(cfg, slurm=cfg.slurm, device=device).to(device)
    load_weights(model, cfg.checkpoint)
    logger.info(f"完整加载 checkpoint：{cfg.checkpoint}")
    settings = OmegaConf.to_container(cfg.sampling, resolve=True)
    mode = settings.pop("mode")
    settings.update(use_guidance=mode == "fluxsteer", use_posthoc=mode == "posthoc")
    cfg.task.visualizer.experiment_method = {
        "fluxsteer": "FluxSteer", "vanilla": "Vanilla FM", "posthoc": "Vanilla + Post-hoc E"
    }[mode]
    params = {"method": cfg.task.visualizer.experiment_method,
              "dataset": cfg.dataset.id, "checkpoint": str(Path(cfg.checkpoint).resolve()),
              "seed": seed, "candidates_per_object": int(cfg.task.test.batch_size),
              "effective_solver": "euler" if mode == "fluxsteer" else settings["method"],
              "flow_forwards_per_step": 2 if mode == "fluxsteer" or settings["method"] == "heun" else 1,
              "settings": settings}
    OmegaConf.save(OmegaConf.create(params), run_dir / "inference_parameters.yaml")
    sample = model.sample
    model.sample = partial(sample, **settings)
    try:
        create_visualizer(cfg.task.visualizer).visualize(model, loader, str(run_dir))
    finally:
        model.sample = sample
    logger.info(f"生成完成：{run_dir / 'samples.pkl'}")


if __name__ == "__main__":
    main()
