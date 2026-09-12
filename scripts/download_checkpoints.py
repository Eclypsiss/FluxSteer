#!/usr/bin/env python3
"""Download FluxSteer weights from a public GitHub Release (no Git LFS needed)."""

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "checkpoints" / "manifest.json").open() as handle:
        manifest = json.load(handle)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["all"] + list(manifest["models"]))
    parser.add_argument("--repo", default=manifest["repository"],
                        help="GitHub repository (default: %(default)s)")
    parser.add_argument("--tag", default=manifest["release_tag"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "fluxsteer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.strip("/")
    parts = args.repo.split("/")
    if len(parts) != 2 or any(not p for p in parts):
        parser.error("Expected --repo in owner/repository format")
    selected = manifest["models"] if args.dataset == "all" else {args.dataset: manifest["models"][args.dataset]}
    for dataset, entry in selected.items():
        destination = args.output_dir / entry["filename"]
        url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{entry['filename']}"
        print(f"{dataset}: {url}\n  -> {destination}", flush=True)
        if args.dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != entry["size_bytes"]:
                raise FileExistsError(f"Unexpected existing file size: {destination}; move it aside before downloading")
            print("Already present; skipping.")
            continue
        temporary = destination.with_suffix(destination.suffix + ".partial")
        urlretrieve(url, str(temporary))
        if temporary.stat().st_size != entry["size_bytes"]:
            raise RuntimeError(f"Unexpected download size: {temporary}; check the release asset")
        temporary.rename(destination)
    print("Done. Check checkpoint provenance in README.md (Limitations).")


if __name__ == "__main__":
    main()
