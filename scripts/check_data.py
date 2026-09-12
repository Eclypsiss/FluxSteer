#!/usr/bin/env python3
"""Check the files needed by the five dataset loaders, without loading tensors."""

import argparse
import json
import os
from pathlib import Path
import sys


LAYOUT = {
    "dexgraspnet": ("DexGraspNet", "dexgraspnet_shadowhand_downsample.pt", "obj_scale_urdf"),
    "unidexgrasp": ("UniDexGrasp", "unidexgrasp_shadowhand_downsample.pt", "obj_scale_urdf"),
    "dexgrab": ("DexGRAB", "DexGRAB_shadowhand_downsample.pt", "contact_meshes"),
    "multidex": ("MultiDex/MultiDex_UR", "shadowhand/shadowhand_downsample.pt", "object"),
    "realdex": ("Realdex", "realdex_shadowhand_downsample.pt", "meshdata"),
}


def check_dataset(root, name):
    directory, annotation, meshes = LAYOUT[name]
    path = root / directory
    required = [annotation, "object_pcds_nors.pkl", meshes]
    if name in ("dexgraspnet", "unidexgrasp", "realdex"):
        required.append("grasp.json")
    if name in ("dexgraspnet", "unidexgrasp"):
        required.append("scales.pkl")
    missing = [str(path / item) for item in required if not (path / item).exists()]
    if missing:
        print(f"{name}: MISSING\n  " + "\n  ".join(missing))
        return False
    split = path / "grasp.json"
    if name in ("dexgraspnet", "unidexgrasp", "realdex"):
        with split.open() as handle:
            contents = json.load(handle)
        for key in ("_train_split", "_test_split", "_all_split"):
            if not isinstance(contents.get(key), list) or not contents[key]:
                raise ValueError(f"{split}: missing/empty list {key}")
        print(f"{name}: files present; train={len(contents['_train_split'])}, test={len(contents['_test_split'])}")
    else:
        print(f"{name}: files present; split is defined in datasets/{name}.py")
    if not (path / "scaled_object_pcds_nors.pkl").exists():
        print("  Evaluation uses object_pcds_nors.pkl (no scaled cache present).")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["all"] + list(LAYOUT))
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("FLUXSTEER_DATA_ROOT", "data")))
    args = parser.parse_args()
    names = list(LAYOUT) if args.dataset == "all" else [args.dataset]
    results = [check_dataset(args.data_root, name) for name in names]
    print("This checks layout only, not per-object mesh completeness or checkpoint identity.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
