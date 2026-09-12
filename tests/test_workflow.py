"""迁移后的 CPU 回归：入口、五数据集、checkpoint 和输出结构。"""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from datasets.base import DATASET, create_dataset
from datasets.misc import collate_fn_general
from fluxsteer import DATASETS, build_commands, make_parser
from engine.assets import find_object_assets, normals_cache
from engine.checkpoint import load_weights, save_training_state
from models.base import create_model
from models.dm.refinement import refine_terminal_state
from scripts.download_checkpoints import main as download_checkpoints


ROOT = Path(__file__).resolve().parents[1]


def configuration(dataset):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="default", overrides=[f"dataset={dataset}"])


class WorkflowTests(unittest.TestCase):
    def test_public_checkpoint_downloads_match_configs(self):
        with (ROOT / "checkpoints" / "manifest.json").open() as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["repository"], "Eclypsiss/FluxSteer")
        self.assertEqual(manifest["release_tag"], "checkpoints-v1")
        self.assertEqual(set(manifest["models"]), set(DATASETS))
        output = io.StringIO()
        with patch("sys.argv", ["download_checkpoints.py", "all", "--dry-run"]), redirect_stdout(output):
            download_checkpoints()
        for name, entry in manifest["models"].items():
            with self.subTest(dataset=name):
                self.assertEqual(entry["filename"], f"fluxsteer_{name}.pth")
                self.assertEqual(configuration(name).checkpoint,
                                 "checkpoints/fluxsteer/" + entry["filename"])
                url = ("https://github.com/Eclypsiss/FluxSteer/releases/download/checkpoints-v1/"
                       + entry["filename"])
                self.assertIn(url, output.getvalue())

    def test_cpu_training_gradient_and_sampling_shape(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        cfg = configuration("dexgraspnet")
        model = create_model(cfg, slurm=False, device="cpu")
        data = {"x": torch.randn(2, 27), "pos": torch.randn(2, 64, 3)}
        model.train()
        loss = model(data)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        model.eval()
        with torch.no_grad():
            sampled = model.sample(data, num_steps=2, method="euler", use_guidance=False)
        self.assertEqual(sampled.shape, (2, 1, 1, 27))
        self.assertTrue(torch.isfinite(sampled).all())
        state, stats = refine_terminal_state(torch.full((2, 27), .5),
                                            lambda x: x.square().mean(dim=1), .01, 5)
        self.assertEqual(state.shape, (2, 27))
        self.assertLess(stats["final_energy_mean"], stats["initial_energy_mean"])

    def test_five_dataset_presets(self):
        expected = {
            "dexgraspnet": (100, 200, .4, .4, .4),
            "unidexgrasp": (75, 200, .4, .4, .4),
            "dexgrab": (200, 400, .3, .5, .4),
            "multidex": (75, 400, .6, .11, .4),
            "realdex": (75, 400, .95, .3, .4),
        }
        cfg = configuration("dexgraspnet")
        model = create_model(cfg, slurm=False, device="cpu")
        self.assertEqual(len(model.state_dict()), 190)
        for name in DATASETS:
            with self.subTest(dataset=name):
                cfg = configuration(name)
                settings = cfg.sampling
                self.assertEqual((settings.num_steps, settings.mc_samples,
                                  settings.w_erf, settings.w_spf, settings.w_srf), expected[name])
                self.assertEqual(model._global_trans_lower.device.type, "cpu")
                with patch.dict(os.environ, {"FLUXSTEER_DATA_ROOT": "/tmp/fluxsteer external data"}):
                    self.assertTrue(cfg.task.dataset.asset_dir.startswith("/tmp/fluxsteer external data/"))
                    self.assertEqual(cfg.task.dataset.asset_dir, cfg.task.visualizer.asset_dir)

    def test_released_checkpoints_strict_load(self):
        directory = Path(os.environ.get("FLUXSTEER_CHECKPOINT_DIR", ROOT / "checkpoints" / "fluxsteer"))
        paths = [directory / Path(configuration(name).checkpoint).name for name in DATASETS]
        if not all(path.is_file() for path in paths):
            self.skipTest("Download all five release checkpoints to run the weight-loading test")
        model = create_model(configuration("dexgraspnet"), slurm=False, device="cpu")
        self.assertEqual(len(model.state_dict()), 190)
        for path in paths:
            with self.subTest(checkpoint=path.name):
                load_weights(model, path)

    def test_launcher_gpu_and_pipeline_paths(self):
        args = make_parser().parse_args([
            "sample-evaluate", "dexgraspnet", "--gpus", "2", "--output", "/tmp/fluxsteer output",
            "--data-root", "/tmp/fluxsteer data", "--object", "bottle", "--candidates", "8", "--dry-run"])
        commands, env, run_dir = build_commands(args)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(len(commands), 2)
        self.assertIn('exp_dir="/tmp/fluxsteer output"', commands[0])
        self.assertIn(str(run_dir), commands[1])
        self.assertIn("/tmp/fluxsteer data", commands[1])
        self.assertEqual(commands[1][-1], "0")
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="default", overrides=commands[0][3:])
        self.assertEqual(cfg.exp_dir, "/tmp/fluxsteer output")
        self.assertEqual(cfg.task.visualizer.object_name, "bottle")
        self.assertEqual(cfg.task.test.batch_size, 8)
        args = make_parser().parse_args(["train", "multidex", "--gpus", "0,3", "--dry-run"])
        commands, _, _ = build_commands(args)
        self.assertIn("torch.distributed.run", commands[0])
        self.assertIn("--nproc_per_node=2", commands[0])

    def test_strict_weights_and_training_resume_state(self):
        with tempfile.TemporaryDirectory(prefix="fluxsteer-checkpoint-") as directory:
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            loss = model(torch.ones(1, 3)).square().mean()
            loss.backward()
            optimizer.step()
            for epoch in range(5):
                save_training_state(model, optimizer, None, epoch, epoch + 1, directory, 2, 3)
            files = sorted(Path(directory).glob("*.pth"))
            self.assertEqual([p.name for p in files], ["epoch_0002.pth", "epoch_0003.pth", "epoch_0004.pth"])
            restored = torch.nn.Linear(3, 2)
            state = load_weights(restored, files[-1])
            self.assertEqual(state["epoch"], 4)
            self.assertTrue(state["optimizer"]["state"])
            self.assertTrue(torch.equal(restored.weight, model.weight))
            invalid = Path(directory) / "incomplete.pth"
            torch.save({"model": {"weight": model.weight.detach()}}, invalid)
            with self.assertRaises(RuntimeError):
                load_weights(restored, invalid)

    def test_five_dataset_loaders_with_minimal_fixtures(self):
        filenames = {"dexgraspnet": "dexgraspnet_shadowhand_downsample.pt",
                     "unidexgrasp": "unidexgrasp_shadowhand_downsample.pt",
                     "realdex": "realdex_shadowhand_downsample.pt",
                     "dexgrab": "DexGRAB_shadowhand_downsample.pt",
                     "multidex": "shadowhand/shadowhand_downsample.pt"}
        with tempfile.TemporaryDirectory(prefix="fluxsteer-datasets-") as directory:
            for name in DATASETS:
                with self.subTest(dataset=name):
                    cfg = configuration(name)
                    asset_dir = Path(directory) / name
                    asset_dir.mkdir()
                    cls = DATASET.get(cfg.dataset.name)
                    if name in ("dexgrab", "multidex"):
                        train_object, test_object = cls._train_split[0], cls._test_split[0]
                    else:
                        train_object, test_object = "train_object", "test_object"
                        (asset_dir / "grasp.json").write_text(json.dumps({
                            "_train_split": [train_object], "_test_split": [test_object],
                            "_all_split": [train_object, test_object]}))
                    objects = [train_object, test_object]
                    pointclouds = {obj: np.random.default_rng(1).normal(size=(64, 6)) for obj in objects}
                    with (asset_dir / "object_pcds_nors.pkl").open("wb") as handle:
                        pickle.dump(pointclouds, handle)
                    with (asset_dir / "scales.pkl").open("wb") as handle:
                        pickle.dump({obj: 1.0 for obj in objects}, handle)
                    metadata = []
                    for obj in objects:
                        if name == "multidex":
                            metadata.append((torch.zeros(33), torch.eye(3), obj))
                        else:
                            metadata.append({"rotations": torch.eye(3), "joint_positions": torch.zeros(24),
                                             "translations": torch.zeros(3), "scale": 1.0, "object_name": obj})
                    tensor_file = asset_dir / filenames[name]
                    tensor_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"info": {"num_per_object": {obj: 1 for obj in objects}},
                                "metadata": metadata}, tensor_file)
                    cfg.task.dataset.asset_dir = str(asset_dir)
                    cfg.task.dataset.asset_dir_slurm = str(asset_dir)
                    for phase in ("train", "test"):
                        dataset = create_dataset(cfg.task.dataset, phase, False, case_only=phase == "test")
                        self.assertGreater(len(dataset), 0)
                        batch = next(iter(dataset.get_dataloader(batch_size=1, collate_fn=collate_fn_general)))
                        self.assertEqual(batch["x"].shape, (1, 27))
                        self.assertTrue(torch.isfinite(batch["x"]).all())

    def test_assets_resolve_only_inside_selected_dataset(self):
        with tempfile.TemporaryDirectory(prefix="fluxsteer-assets-") as directory:
            root = Path(directory)
            flat = root / "obj_scale_urdf"
            flat.mkdir()
            for extension in ("obj", "urdf"):
                (flat / f"bottle.{extension}").touch()
            urdf, mesh = find_object_assets(root, "bottle")
            self.assertEqual(Path(mesh), flat / "bottle.obj")
            self.assertEqual(Path(urdf), flat / "bottle.urdf")
            nested = root / "object" / "contactdb" / "apple"
            nested.mkdir(parents=True)
            for extension in ("stl", "urdf"):
                (nested / f"apple.{extension}").touch()
            self.assertEqual(Path(find_object_assets(root, "contactdb+apple")[1]), nested / "apple.stl")
            for filename in ("scaled_object_pcds_nors.pkl", "object_pcds_nors.pkl"):
                (root / filename).touch()
            self.assertEqual(normals_cache(root).name, "scaled_object_pcds_nors.pkl")
            with self.assertRaises(FileNotFoundError):
                find_object_assets(root, "missing")


if __name__ == "__main__":
    unittest.main()
