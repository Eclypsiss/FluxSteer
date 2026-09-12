#!/usr/bin/env python3
"""Train, sample, and evaluate FluxSteer on five Shadow Hand datasets."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASETS = ("dexgraspnet", "unidexgrasp", "dexgrab", "multidex", "realdex")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "sample", "evaluate", "sample-evaluate"):
        command = actions.add_parser(action)
        command.add_argument("dataset", choices=DATASETS)
        if action == "evaluate":
            command.add_argument("run_dir", type=Path, help="Sampling directory containing samples.pkl")
        else:
            command.add_argument("--output", type=Path, help="New output directory; timestamped by default")
        command.add_argument("--gpus", default="0", help="Physical GPU IDs, e.g. 0 or 0,1; multi-GPU is training only")
        command.add_argument("--data-root", type=Path, help="Dataset parent directory; also FLUXSTEER_DATA_ROOT")
        command.add_argument("--dry-run", action="store_true", help="Print commands without running them")
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Repeatable Hydra override")
        if action == "train":
            command.add_argument("--resume", type=Path, help="Checkpoint to resume from")
            command.add_argument("--epochs", type=int, help="Total training epochs, including completed epochs")
            command.add_argument("--batch-size", type=int, help="Training batch size per GPU")
        elif action != "evaluate":
            command.add_argument("--checkpoint", type=Path, help="Override the dataset's released FluxSteer weights")
            command.add_argument("--steps", type=int, help="Integration steps")
            command.add_argument("--mc-samples", type=int, help="Monte Carlo proposals per grasp per step")
            command.add_argument("--candidates", type=int, default=32, help="Grasp candidates per object")
            command.add_argument("--object", help="Generate only this object ID")
            command.add_argument("--objects-file", type=Path, help="Object IDs, one per line")
            command.add_argument("--mode", choices=("fluxsteer", "vanilla", "posthoc"), default="fluxsteer")
        if action in ("evaluate", "sample-evaluate"):
            command.add_argument("--evaluation-seed", type=int, default=42)
    return parser


def config_value(value):
    # Hydra 字符串需要保留空格，并让 OmegaConf 处理转义。
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_commands(args):
    for override in args.set:
        key = override.split("=", 1)[0].lstrip("+")
        if key in ("dataset", "exp_dir", "gpu", "task.dataset.asset_dir", "dataset.asset_dir"):
            raise ValueError(f"{key} 请通过数据集位置参数、--output、--gpus 或 --data-root 设置")
    gpu_ids = [part.strip() for part in args.gpus.split(",")]
    if not gpu_ids or any(not part.isdigit() for part in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus 必须是不同的非负整数编号，例如 0,1")
    if args.action != "train" and len(gpu_ids) != 1:
        raise ValueError("单次采样/评估使用一张卡；可用不同输出目录分别启动")
    if getattr(args, "object", None) and getattr(args, "objects_file", None):
        raise ValueError("--object 与 --objects-file 只能选择一个")
    for field in ("epochs", "batch_size", "steps", "mc_samples", "candidates"):
        value = getattr(args, field, None)
        if value is not None and value <= 0:
            raise ValueError(f"{field} 必须大于 0")
    if args.action == "sample-evaluate" and args.candidates % 8:
        raise ValueError("MAIN collision 评估要求 --candidates 是 8 的倍数")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    if args.data_root:
        environment["FLUXSTEER_DATA_ROOT"] = str(args.data_root.expanduser().resolve())
    environment.setdefault("FLUXSTEER_DATA_ROOT", str(ROOT / "data"))
    environment.setdefault("OMP_NUM_THREADS", "1")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = getattr(args, "output", None)
    if args.action == "evaluate":
        run_dir = args.run_dir.expanduser().resolve()
    else:
        run_dir = (output.expanduser().resolve() if output else
                   ROOT / "outputs" / f"fluxsteer_{args.dataset}_{args.action}_{stamp}")
    common = [f"dataset={args.dataset}", f"exp_dir={config_value(run_dir)}"]
    commands = []
    if args.action == "train":
        prefix = [sys.executable, "-m", "engine.train"]
        if len(gpu_ids) > 1:
            prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                      f"--nproc_per_node={len(gpu_ids)}", "--module", "engine.train"]
        options = list(common)
        if args.resume:
            options.append(f"resume={config_value(args.resume.expanduser().resolve())}")
        if args.epochs:
            options.append(f"task.train.num_epochs={args.epochs}")
        if args.batch_size:
            options.append(f"task.train.batch_size={args.batch_size}")
        commands.append(prefix + options + args.set)
    if args.action in ("sample", "sample-evaluate"):
        options = list(common) + [f"sampling.mode={args.mode}", f"task.test.batch_size={args.candidates}"]
        if args.checkpoint:
            options.append(f"checkpoint={config_value(args.checkpoint.expanduser().resolve())}")
        if args.steps:
            options.append(f"sampling.num_steps={args.steps}")
        if args.mc_samples:
            options.append(f"sampling.mc_samples={args.mc_samples}")
        if args.object:
            options.append(f"task.visualizer.object_name={config_value(args.object)}")
        if args.objects_file:
            options.append(f"task.visualizer.object_list_file={config_value(args.objects_file.expanduser().resolve())}")
        commands.append([sys.executable, "-m", "engine.sample"] + options + args.set)
    if args.action in ("evaluate", "sample-evaluate"):
        commands.append([sys.executable, "-m", "engine.evaluate", "--dataset", args.dataset,
                         "--eval-dir", str(run_dir), "--data-root", environment["FLUXSTEER_DATA_ROOT"],
                         "--seed", str(args.evaluation_seed), "--gpu", "0"])
    return commands, environment, run_dir


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        commands, environment, run_dir = build_commands(args)
    except ValueError as error:
        parser.error(str(error))
    if args.action == "evaluate" and args.set:
        parser.error("evaluate 使用采样目录和 --data-root；不接收 --set")
    if not args.dry_run and args.action != "evaluate" and run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"输出目录非空，请选择新目录：{run_dir}")
    print(f"FLUXSTEER_DATA_ROOT={environment['FLUXSTEER_DATA_ROOT']}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']}", flush=True)
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    if not args.dry_run:
        print(f"结果目录：{run_dir}")


if __name__ == "__main__":
    main()
