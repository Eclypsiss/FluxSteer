"""FluxSteer 的 flow-matching 训练，支持单卡、多卡与断点继续。"""

from functools import partial
import math
import os
from pathlib import Path
import random

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from engine.checkpoint import load_weights, save_training_state
from models.base import create_model
from utils.misc import compute_model_dim


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(cfg.gpu)))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
    seed = int(cfg.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg.model.d_x = compute_model_dim(cfg.task)
    run_dir = Path(cfg.exp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if rank == 0:
        logger.add(str(run_dir / "train.log"))
        OmegaConf.save(cfg, run_dir / "resolved_config.yaml", resolve=True)
        writer = SummaryWriter(str(run_dir / "tensorboard"))
        logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    dataset = create_dataset(cfg.task.dataset, "train", cfg.slurm)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    collate = (collate_fn_squeeze_pcd_batch if cfg.model.scene_model.name == "PointTransformer"
               else collate_fn_general)
    loader = dataset.get_dataloader(batch_size=cfg.task.train.batch_size, sampler=sampler,
                                   shuffle=sampler is None, num_workers=cfg.task.train.num_workers,
                                   collate_fn=partial(collate, use_llm=cfg.model.use_llm), pin_memory=True)
    model = create_model(cfg, slurm=cfg.slurm, device=device).to(device)
    checkpoint = load_weights(model, cfg.resume) if cfg.resume else {}
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    step = int(checkpoint.get("step", 0))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.task.lr)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler = None
    if cfg.task.use_scheduler:
        def lr_scale(epoch):
            warmup = int(cfg.task.warmup_epochs)
            peak = float(cfg.task.peak_lr / cfg.task.lr)
            minimum = float(cfg.task.min_lr / cfg.task.lr)
            if epoch < warmup:
                return 1.0 + (peak - 1.0) * epoch / max(warmup, 1)
            progress = min(1.0, (epoch - warmup) / max(1, cfg.task.train.num_epochs - warmup))
            return minimum + (peak - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", cfg.task.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale, last_epoch=start_epoch - 1)
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
    if cfg.resume and "optimizer" not in checkpoint:
        logger.info("历史 checkpoint 只有模型权重，优化器将从当前配置初始化。")
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                        find_unused_parameters=True)
    try:
        for epoch in range(start_epoch, cfg.task.train.num_epochs):
            model.train()
            if sampler is not None:
                sampler.set_epoch(epoch)
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.task.train.num_epochs}", disable=rank != 0)
            for data in progress:
                data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
                data["epoch"] = epoch
                optimizer.zero_grad()
                loss = model(data)["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"非有限训练 loss：epoch={epoch}, step={step}")
                loss.backward()
                optimizer.step()
                step += 1
                if rank == 0:
                    progress.set_postfix(loss=loss.item())
                    if step % cfg.task.train.log_step == 0:
                        writer.add_scalar("train/loss", loss.item(), step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                        logger.info(f"epoch={epoch + 1} step={step} loss={loss.item():.6f}")
            if scheduler is not None:
                scheduler.step()
            if rank == 0 and (epoch + 1) % cfg.save_model_interval == 0:
                path = save_training_state(model, optimizer, scheduler, epoch, step,
                                           run_dir / "checkpoints", cfg.checkpoint_keep_latest,
                                           cfg.checkpoint_milestone_interval)
                logger.info(f"保存 checkpoint：{path}")
    finally:
        if writer is not None:
            writer.close()
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
