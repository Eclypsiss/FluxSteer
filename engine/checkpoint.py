"""读取历史模型权重和保存完整训练状态。"""

from pathlib import Path
import torch


def load_weights(model, path):
    checkpoint = torch.load(str(path), map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def save_training_state(model, optimizer, scheduler, epoch, step, directory, keep_latest=3,
                        milestone_interval=100):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if keep_latest < 1 or milestone_interval < 1:
        raise ValueError("checkpoint 保留数量与里程碑间隔必须大于 0")
    module = model.module if hasattr(model, "module") else model
    path = directory / f"epoch_{epoch:04d}.pth"
    torch.save({"model": module.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch, "step": step}, str(path))
    candidates = sorted(directory.glob("epoch_*.pth"))
    ordinary = [p for p in candidates if (int(p.stem.split("_")[1]) + 1) % milestone_interval]
    for old in ordinary[:-keep_latest]:
        old.unlink()
    return path
