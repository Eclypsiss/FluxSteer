#!/usr/bin/env python3
"""Fetch the original Shadow Hand assets and simulator base task separately."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
REVISION = "86353d710ca71d0f547f0d8b15cc8ca959c78163"
PREFIX = "DexGrasp-Anything-" + REVISION
URL = "https://codeload.github.com/4DVLab/DexGrasp-Anything/tar.gz/" + REVISION
PARTS = ("assets/urdf", "envs/assets", "envs/tasks/base_task.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a locally downloaded copy of the pinned source tar.gz")
    parser.add_argument("--source-dir", type=Path, help="Use an already extracted copy of that upstream revision")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(f"Source: {URL}\nComponents: {', '.join(PARTS)}")
    print("Third-party files retain their original terms; see README.md (License and acknowledgements).")
    if args.dry_run:
        return
    if args.archive and args.source_dir:
        parser.error("Choose --archive or --source-dir, not both")
    for part in PARTS:
        if (ROOT / part).exists():
            raise FileExistsError(f"Already exists: {ROOT / part}. Existing assets are not overwritten.")
    with tempfile.TemporaryDirectory(prefix="fluxsteer-assets-") as temp:
        if args.source_dir:
            source = args.source_dir.resolve()
        else:
            archive = args.archive.resolve() if args.archive else Path(temp) / "upstream.tar.gz"
            if not args.archive:
                urlretrieve(URL, str(archive))
            subprocess.run(["tar", "-xzf", str(archive), "-C", temp] +
                           [PREFIX + "/" + part for part in PARTS], check=True)
            source = Path(temp) / PREFIX
        for part in PARTS:
            if not (source / part).exists():
                raise FileNotFoundError(f"Missing upstream component: {source / part}")
        for part in PARTS:
            destination = ROOT / part
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (source / part).is_dir():
                shutil.copytree(source / part, destination,
                                ignore=shutil.ignore_patterns(".DS_Store", "__pycache__", "*.pyc"))
            else:
                shutil.copy2(source / part, destination)
    print("Assets ready. Install Isaac Gym separately before evaluation.")


if __name__ == "__main__":
    main()
