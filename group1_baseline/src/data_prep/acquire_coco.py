"""COCO 2017 acquisition helpers.

This module mirrors the original notebook's data download step but keeps the
logic in `src/` so the workflow notebook stays orchestration-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
import zipfile


TRAIN_ZIP_URL = "http://images.cocodataset.org/zips/train2017.zip"
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def coco_files_status(project_root: Path) -> dict[str, Path]:
    """Return expected COCO paths for status checks."""
    return {
        "train_zip": project_root / "data" / "raw" / "train2017.zip",
        "ann_zip": project_root / "data" / "raw" / "annotations_trainval2017.zip",
        "train_dir": project_root / "data" / "raw" / "train2017",
        "ann_json": project_root / "data" / "raw" / "annotations" / "captions_train2017.json",
    }


def _download_if_missing(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"skip download (exists): {out_path}")
        return
    print(f"downloading: {url}")
    urlretrieve(url, out_path)
    print(f"saved: {out_path}")


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"zip file not found: {zip_path}")
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"extracting: {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def acquire_coco_2017(project_root: Path, download: bool = True, extract: bool = True) -> None:
    """Ensure COCO train images + captions annotations exist under data/raw."""
    paths = coco_files_status(project_root)
    raw_root = project_root / "data" / "raw"

    if download:
        _download_if_missing(TRAIN_ZIP_URL, paths["train_zip"])
        _download_if_missing(ANN_ZIP_URL, paths["ann_zip"])

    if extract:
        if not paths["train_dir"].exists():
            _extract_zip(paths["train_zip"], raw_root)
        else:
            print(f"skip extract (exists): {paths['train_dir']}")

        if not paths["ann_json"].exists():
            _extract_zip(paths["ann_zip"], raw_root)
        else:
            print(f"skip extract (exists): {paths['ann_json']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root that contains data/raw",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading zip files.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extracting zip files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acquire_coco_2017(
        project_root=args.project_root.resolve(),
        download=not args.no_download,
        extract=not args.no_extract,
    )


if __name__ == "__main__":
    main()
