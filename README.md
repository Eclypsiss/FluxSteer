# FluxSteer

Energy-guided flow matching for physics-aware dexterous grasp generation.

FluxSteer steers a pretrained flow-matching generator using object-penetration, surface-contact, and hand self-penetration energies, without additional guidance-stage training. This implementation supports **Shadow Hand** on **DexGraspNet, UniDexGrasp, DexGRAB, MultiDex, and RealDex**.

[Installation](#installation) · [Checkpoints](#checkpoints) · [Datasets](#datasets) · [Quick start](#quick-start) · [Training](#training) · [Sampling and evaluation](#sampling-and-evaluation) · [Limitations](#limitations)

## Installation

Use **Linux x86_64, Python 3.8, PyTorch 1.11.0 + CUDA 11.3, and Isaac Gym Preview 4**. Ubuntu 20.04 is a suitable starting point. A working NVIDIA driver and a CUDA toolkit containing `nvcc` are required; the PyTorch wheel alone does not provide the compiler. Isaac Sim and Isaac Lab are not drop-in replacements for this evaluator.

### Python environment

```bash
git clone https://github.com/Eclypsiss/FluxSteer.git
cd FluxSteer
conda create -n fluxsteer python=3.8 -y
conda activate fluxsteer
python -m pip install pip==23.3.2 setuptools==59.5.0 wheel==0.42.0
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements.txt
```

All subsequent commands run from the repository root. NumPy is pinned to 1.23.5 for the unmodified simulator's legacy aliases. The released checkpoints use PointNet; optional inherited PointNet2, PointTransformer, and LLM branches are not required by this installation.

Typical system packages on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install build-essential gcc-9 g++-9 curl unzip \
  libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 libvulkan1
```

### Geometry extensions

Set `CUDA_HOME` to the actual CUDA 11.3 toolkit location and select your GPU architecture. Examples: V100 `7.0`, RTX 2080 Ti `7.5`, A100 `8.0`, RTX 3090 `8.6`. For RTX 4090, this legacy toolkit needs a PTX-compatible build such as `8.6+PTX`, followed by a local smoke test; it cannot compile native `8.9` code.

```bash
export CUDA_HOME=/usr/local/cuda-11.3
export PATH="$CUDA_HOME/bin:$PATH"
export CC=gcc-9
export CXX=g++-9
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="8.6+PTX"  # Change for your GPU.
nvcc --version

FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps \
  https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.7.2.tar.gz
FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps \
  https://github.com/wrc042/CSDF/archive/41ce6052786d9a93f446d71531eddf6f4a4f1ece.tar.gz
```

See the upstream [PyTorch3D installation instructions](https://github.com/facebookresearch/pytorch3d/blob/v0.7.2/INSTALL.md) for compiler/build details. Build extensions on the target machine instead of copying another machine's `.so` files.

### Simulator and hand assets

Download `IsaacGym_Preview_4_Package.tar.gz` from [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym), following its license/login requirements, then install it:

```bash
mkdir -p src
tar -xzf /absolute/path/to/IsaacGym_Preview_4_Package.tar.gz -C src
python -m pip install --no-deps -e ./src/isaacgym/python
python scripts/fetch_assets.py
```

The asset helper fetches `assets/urdf/`, `envs/assets/`, and `envs/tasks/base_task.py` from the fixed upstream [DGA revision](https://github.com/4DVLab/DexGrasp-Anything/tree/86353d710ca71d0f547f0d8b15cc8ca959c78163). These third-party components retain their own terms and are excluded from this source repository. The helper does not overwrite existing assets. Offline use: download the archive URL printed by `python scripts/fetch_assets.py --dry-run`, transfer it, and run `python scripts/fetch_assets.py --archive /path/to/upstream.tar.gz`.

Import Isaac Gym **before** PyTorch in a process using the simulator:

```bash
python -c "from isaacgym import gymapi; import torch; from pytorch3d.ops import knn_points; from csdf import compute_sdf; import pytorch_kinematics; print(torch.__version__, torch.cuda.is_available())"
```

## Checkpoints

Download the weights from [checkpoints-v1](https://github.com/Eclypsiss/FluxSteer/releases/tag/checkpoints-v1). The helper uses this repository and release by default:

```bash
python scripts/download_checkpoints.py dexgraspnet
# Or download all five models:
python scripts/download_checkpoints.py all
```

| Dataset | Direct download |
|---|---|
| DexGraspNet | [fluxsteer_dexgraspnet.pth](https://github.com/Eclypsiss/FluxSteer/releases/download/checkpoints-v1/fluxsteer_dexgraspnet.pth) |
| UniDexGrasp | [fluxsteer_unidexgrasp.pth](https://github.com/Eclypsiss/FluxSteer/releases/download/checkpoints-v1/fluxsteer_unidexgrasp.pth) |
| DexGRAB | [fluxsteer_dexgrab.pth](https://github.com/Eclypsiss/FluxSteer/releases/download/checkpoints-v1/fluxsteer_dexgrab.pth) |
| MultiDex | [fluxsteer_multidex.pth](https://github.com/Eclypsiss/FluxSteer/releases/download/checkpoints-v1/fluxsteer_multidex.pth) |
| RealDex | [fluxsteer_realdex.pth](https://github.com/Eclypsiss/FluxSteer/releases/download/checkpoints-v1/fluxsteer_realdex.pth) |

Files are saved under `checkpoints/fluxsteer/` and selected automatically by the dataset configuration. For manual downloads, use the same directory and filenames. Each file is approximately 123 MiB. The source-code ZIP does not include weights. See [Limitations](#limitations) for historical checkpoint provenance.

## Datasets

Use the **DGA-preprocessed Shadow Hand packages** below, not the original raw dataset formats. The packages include the pose annotations, splits, point clouds/normals, scaling information, and physics assets expected by these loaders. Only the selected dataset is needed for a single-dataset experiment.

| CLI argument | Processed package | Google Drive mirror |
|---|---|---|
| `dexgraspnet` | [DexGraspNet.zip](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/DexGraspNet.zip?download=true) | [Drive](https://drive.google.com/file/d/1FHJxEDl2jegOpq-g4KZ4eEVvM3gqDQCh/view) |
| `unidexgrasp` | [UniDexGrasp.zip](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/UniDexGrasp.zip?download=true) | [Drive](https://drive.google.com/file/d/1-nPUP14x0VOfIqQwYU-hc-WhUaPBxEQ7/view) |
| `dexgrab` | [DexGRAB.zip](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/DexGRAB.zip?download=true) | [Drive](https://drive.google.com/file/d/1Xmgw-c3lrkab2NIs_1i0Hq95I0Y4Sp8n/view) |
| `multidex` | [MultiDex_UR.zip](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/MultiDex_UR.zip?download=true) | [Drive](https://drive.google.com/file/d/1wHdWLfvxWjpFBV_Ld-j4DwNXAr1UMERf/view) |
| `realdex` | [Realdex.zip](https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/Realdex.zip?download=true) | [Drive](https://drive.google.com/file/d/12rgyyKg07PmY6jzl7pMocA4o5ikLFuOA/view) |

These links are provided by [upstream DGA](https://github.com/4DVLab/DexGrasp-Anything#-datasets). DexGRAB is the Shadow Hand-retargeted GRAB dataset. Approximate archive sizes are 3.79 GB, 47.1 GB, 63 MB, 70 MB, and 76 MB, respectively; allow extra space for extraction. The separate `Dexgraspanyting.tar.gz` archive is not one of the five required datasets.

Example download and extraction:

```bash
export FLUXSTEER_DATA_ROOT=/absolute/path/to/grasp_data
mkdir -p "$FLUXSTEER_DATA_ROOT" downloads
curl -fL --retry 3 -C - \
  'https://huggingface.co/datasets/GaussionZhong/DexGrasp-Anything/resolve/main/DexGraspNet.zip?download=true' \
  -o downloads/DexGraspNet.zip
unzip -l downloads/DexGraspNet.zip
unzip -n downloads/DexGraspNet.zip -d "$FLUXSTEER_DATA_ROOT"
```

Repeat for the required datasets. Inspect each archive's top-level directory and arrange the files as below; avoid extra nesting such as `DexGraspNet/DexGraspNet/`. The expected directory name is `Realdex` (lowercase `d`). If `MultiDex_UR.zip` contains a top-level `MultiDex_UR/`, extract it into `$FLUXSTEER_DATA_ROOT/MultiDex/`.

```text
$FLUXSTEER_DATA_ROOT/
├── DexGraspNet/
│   ├── grasp.json
│   ├── dexgraspnet_shadowhand_downsample.pt
│   ├── object_pcds_nors.pkl
│   ├── scales.pkl
│   └── obj_scale_urdf/                 # Matching <object_id>.obj and .urdf
├── UniDexGrasp/
│   ├── grasp.json
│   ├── unidexgrasp_shadowhand_downsample.pt
│   ├── object_pcds_nors.pkl
│   ├── scales.pkl
│   └── obj_scale_urdf/                 # Matching flat .obj and .urdf
├── DexGRAB/
│   ├── DexGRAB_shadowhand_downsample.pt
│   ├── object_pcds_nors.pkl
│   └── contact_meshes/                 # Matching .ply and .urdf
├── MultiDex/MultiDex_UR/
│   ├── shadowhand/shadowhand_downsample.pt
│   ├── object_pcds_nors.pkl
│   └── object/<category>/<name>/       # Matching <name>.stl and .urdf
└── Realdex/
    ├── grasp.json
    ├── realdex_shadowhand_downsample.pt
    ├── object_pcds_nors.pkl
    └── meshdata/                       # Matching .obj and .urdf
```

Keep `scaled_object_pcds_nors.pkl` when supplied: evaluation prefers it over `object_pcds_nors.pkl`. The point clouds must contain XYZ and surface normals even when encoder normal features are disabled. Preserve `scales.pkl`, object IDs, and supplied splits. Several sampling loaders also read the `.pt` annotations, so meshes alone are insufficient. `grasp.json` contains `_train_split`, `_test_split`, and `_all_split`; DexGRAB and MultiDex use explicit splits in their loader source. Physics assets are resolved only inside the chosen dataset directory.

```bash
python scripts/check_data.py dexgraspnet
# After preparing all five:
python scripts/check_data.py all
```

`datasets/` in this repository contains loader code, not dataset contents. The checker validates the required layout and split files; per-object mesh/URDF availability is checked by evaluation. Use `--data-root PATH` to override the environment variable. If a required file is missing, check extraction nesting and the processed-data mirror.

## Quick start

After installing dependencies and downloading DexGraspNet and its checkpoint, run a low-budget, one-object smoke test:

```bash
python fluxsteer.py sample-evaluate dexgraspnet --gpus 0 \
  --object core-bottle-114509277e76e413c8724d5673a063a6 \
  --candidates 8 --steps 2 --mc-samples 4 --output outputs/fluxsteer_smoke
```

This checks installation, not paper-result performance. If the object is absent from your archive, choose an ID from its test split with a point cloud, mesh, and URDF. Expect finite `8 x 33` exported grasps plus evaluation CSV/JSON files. The model internally uses 27 state coordinates; export constructs translation (3), rotation (6), and joint coordinates (24).

All `fluxsteer.py` commands support `--dry-run`, which prints the invocation without launching an experiment. CPU regression tests use synthetic fixtures and no datasets:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

The strict weight-loading test requires all five checkpoints and is otherwise skipped.

## Training

```bash
# Single GPU; replace the dataset argument with any name in the dataset table.
python fluxsteer.py train dexgraspnet --gpus 0 --epochs 3000 --batch-size 128 \
  --output outputs/train_dexgraspnet

# Multi-GPU training; batch size is per GPU.
python fluxsteer.py train multidex --gpus 0,1 --epochs 3000 --batch-size 64 \
  --output outputs/train_multidex

# Resume; epochs is the total target count, not additional epochs.
python fluxsteer.py train dexgraspnet \
  --resume outputs/train_dexgraspnet/checkpoints/epoch_0099.pth \
  --epochs 3000 --output outputs/train_resumed
```

The default backbone training uses the implemented L1 flow-matching loss, Adam, and 3,000 target epochs. New checkpoints include model, optimizer, scheduler, epoch, and step. To sample a newly trained model, pass `--checkpoint outputs/TRAIN_RUN/checkpoints/epoch_XXXX.pth`.

## Sampling and evaluation

Sample and evaluate separately:

```bash
python fluxsteer.py sample dexgraspnet --gpus 0 --output outputs/fluxsteer_dexgraspnet
python fluxsteer.py evaluate dexgraspnet outputs/fluxsteer_dexgraspnet --gpus 0
```

Or run both stages for an entire test split. These are full-dataset experiments:

```bash
python fluxsteer.py sample-evaluate dexgraspnet --gpus 0 --output outputs/fluxsteer_dexgraspnet_full
python fluxsteer.py sample-evaluate unidexgrasp --gpus 0 --output outputs/fluxsteer_unidexgrasp
python fluxsteer.py sample-evaluate dexgrab --gpus 0 --output outputs/fluxsteer_dexgrab
python fluxsteer.py sample-evaluate multidex --gpus 0 --output outputs/fluxsteer_multidex
python fluxsteer.py sample-evaluate realdex --gpus 0 --output outputs/fluxsteer_realdex
```

Dataset-specific inference presets:

| Dataset argument | Integration steps | MC proposals per step | ERF / SPF / SRF |
|---|---:|---:|---|
| `dexgraspnet` | 100 | 200 | .4 / .4 / .4 |
| `unidexgrasp` | 75 | 200 | .4 / .4 / .4 |
| `dexgrab` | 200 | 400 | .3 / .5 / .4 |
| `multidex` | 75 | 400 | .4 / .4 / .4 |
| `realdex` | 75 | 400 | 1.0 / .3 / .4 |

Defaults: 32 candidates/object, `local_sim_mc`, guided Euler, guidance scale 30, temperature .05, sampling seed 0, and evaluation seed 42. Proposal standard deviation is .1 in normalized model-state coordinates. Guided sampling uses two flow-network forwards per step; vanilla Euler uses one. MC energy evaluations add further computation. Evaluation requires a positive multiple of 8 candidates/object. Sampling and evaluation each use one physical GPU; multiple `--gpus` values are training-only.

```bash
# Selected objects, one object ID per line; optionally choose another checkpoint.
python fluxsteer.py sample-evaluate multidex --objects-file objects.txt \
  --checkpoint checkpoints/fluxsteer/fluxsteer_multidex.pth --output outputs/selected_objects

# Controls using the same flow model: no guidance, or terminal energy refinement.
python fluxsteer.py sample dexgraspnet --mode vanilla --output outputs/vanilla
python fluxsteer.py sample dexgraspnet --mode posthoc --output outputs/posthoc

# Explicit seed and configuration overrides.
python fluxsteer.py sample-evaluate dexgraspnet --set seed=0 \
  --evaluation-seed 42 --set sampling.w_spf=0.4 --output outputs/explicit_settings
```

Configuration entry points: `configs/default.yaml`, `configs/dataset/`, and `configs/task/grasp.yaml`. The supplied post-hoc control defaults to 25 iterations with learning rate .01.

### Outputs and evaluation protocol

Sampling writes `samples.pkl`, `sample.log`, `resolved_config.yaml`, `inference_parameters.yaml`, and `inference_times.csv`. Evaluation adds `evaluation.log`, `per_object_metrics.csv`, and `summary_metrics.json`.

The retained MAIN physics protocol uses six sequential force directions, 50 simulation steps per direction, volume-proportional force, a .02 m displacement threshold, friction 2.0, and pre-force closure learning rate .5. Directions are not independently reset trials. `Suc6` means passing all six directions; `Suc1` means passing at least one. CSV success rates are fractions; JSON success values are percentages. `Pen_all_mm` and `Pen_success_mm` use all and successful candidates, respectively; `Div_rad` reports diversity.

Inspect `objects`, `objects_evaluated`, `objects_skipped`, and `skipped_objects` together with success counts/totals. Missing assets and failed objects are recorded rather than silently included as successful evaluations. UniDexGrasp historically generated 2,268 objects but evaluated 2,224 because 44 lacked usable assets under this protocol; use the actual denominator of each new run.

## Evaluation records

Five-benchmark evaluation records, generated grasps,
and verification tools are available in the
[results release](https://github.com/Eclypsiss/FluxSteer/releases/tag/results-v1).

See the archive README for reporting conventions, recorded configurations,
and limitations. The archive covers the historical main comparison only.

## Troubleshooting

- **Missing `libpython3.8.so.1.0`:** activate the Conda environment, then try `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"` before importing Isaac Gym.
- **PyTorch imported before Isaac Gym:** start a fresh process and import `isaacgym` first. `engine.evaluate` already follows this order.
- **Missing `nvcc`, `csdf._C`, or `pytorch3d._C`:** check the CUDA toolkit, compiler, and architecture, then rebuild the geometry extensions with CUDA support.
- **Missing hand mesh or `base_task.py`:** run `scripts/fetch_assets.py`. Object meshes and URDFs come from the separate dataset packages.
- **CUDA out of memory:** use the 8-candidate / 2-step / K=4 smoke first. Formal budgets and energy weights must be reported when changed.

## Limitations

- **Checkpoint provenance:** DexGraspNet and UniDexGrasp have verified historical main-run model identity. DexGRAB's compatible checkpoint has a later timestamp than the historical run; MultiDex supplies a complete compatible model whereas the historical run partially loaded another checkpoint; RealDex supplies a compatible model because the original historical checkpoint path was unavailable. All five match the architecture with strict loading, but the latter three do not establish exact reproduction of earlier paper-table values. Original filenames and provenance are retained in [checkpoints/manifest.json](checkpoints/manifest.json).
- **Training and inference history:** released weights contain `model`, `epoch`, and `step`, not optimizer/scheduler state. Resuming from them initializes a new optimizer. The 3,000-epoch recipe is a supported training workflow, not a reconstructed schedule for every historical model. DexGRAB's historical temperature was not recorded; .05 is the current implementation default. Preserve dataset-specific presets when comparing results.
- **Validation:** CPU regression checks cover model gradients, synthetic dataset loaders, CLI/configuration wiring, checkpoint loading/resume, and asset resolution. They do not validate guided CUDA sampling or physics on a new machine. A clean installation and the one-object GPU smoke remain required; cross-driver/GPU numerical equivalence is not guaranteed. The complete public dataset archives have not been freshly downloaded and revalidated against the local datasets.
- **Scope:** this is a Shadow Hand grasp-pose generator, not a robot-arm controller or real-robot execution package. Vanilla/post-hoc controls use the same flow backbone; original DGA diffusion checkpoints are different models and cannot replace these weights.

## License and acknowledgements

This implementation builds on [DexGrasp-Anything](https://github.com/4DVLab/DexGrasp-Anything), Scene-Diffuser, and the retained PointNet/PointNet2 components. Original copyright notices and the upstream MIT [LICENSE](LICENSE) are preserved. DGA names refer to the upstream work, not to FluxSteer.

Separately downloaded components retain their own terms: [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) and the upstream `base_task.py` carry NVIDIA notices; the Shadow Hand asset package labels its license `Private`. These assets and simulator files are not bundled or relicensed here. [PyTorch3D](https://github.com/facebookresearch/pytorch3d/tree/v0.7.2), [CSDF](https://github.com/wrc042/CSDF), and [PyTorch Kinematics](https://pypi.org/project/pytorch-kinematics/0.7.5/) are obtained from their providers under their respective terms.

Data originate from [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet), [UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp), [GRAB](https://grab.is.tue.mpg.de/), [MultiDex / GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp), and [RealDex](https://github.com/4DVLab/RealDex). Original and processed-data terms, including applicable research-use restrictions, remain in force. Dataset availability does not itself grant unrestricted redistribution rights for derived artifacts.
