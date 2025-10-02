
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPoint + LightGlue SfM (HLOC) -> COLMAP Layout (Clean, No _aux)
-------------------------------------------------------------------
- Exhaustive pairing only (pairs_from_exhaustive)
- Outputs in COLMAP layout at output root:
    <out>/database.db
    <out>/sparse/0/  (no extra folders)
- No creation of _aux; do not restructure sparse/0 beyond dedup.
- Dedup rule: if both <stem>.bin and <stem>.txt exist in sparse/0, keep .bin and remove .txt.
- Move semantics used to avoid duplicates between work_dir and output_root.

Usage:
    python sp_lightglue_sfm_colmap_clean.py --project /path/to/Proj1 [--output-root /path/to/out]
"""

import argparse
from pathlib import Path
import sys
import shutil
import json
import inspect
import time

# Import HLOC modules
try:
    from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
except Exception as e:
    print("[ERROR] Failed to import HLOC modules. Make sure 'hloc' is installed.", file=sys.stderr)
    raise

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def ensure_images_dir(project_dir: Path) -> Path:
    images = project_dir / "images"
    if not images.exists() or not images.is_dir():
        raise FileNotFoundError(f"'images' folder not found in: {project_dir}")
    imgs = [p for p in images.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise FileNotFoundError(f"No image files found under: {images}")
    return images

def make_image_list_file(images_dir: Path, out_dir: Path) -> Path:
    rels = sorted(str(p.relative_to(images_dir)) for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    list_path = out_dir / "images.txt"
    list_path.write_text("\n".join(rels), encoding="utf-8")
    return list_path

def call_pairs_exhaustive(images_dir: Path, output_pairs: Path, image_list_path: Path):
    sig = inspect.signature(pairs_from_exhaustive.main)
    params = list(sig.parameters.keys())
    try:
        if 'image_dir' in params:
            return pairs_from_exhaustive.main(image_dir=images_dir, output=output_pairs)
        elif 'image_list' in params:
            return pairs_from_exhaustive.main(image_list=image_list_path, output=output_pairs)
        else:
            first = params[0] if params else 'image_dir'
            if 'list' in first:
                return pairs_from_exhaustive.main(image_list_path, output_pairs)
            else:
                return pairs_from_exhaustive.main(images_dir, output_pairs)
    except TypeError:
        try:
            return pairs_from_exhaustive.main(image_list=image_list_path, output=output_pairs)
        except TypeError:
            return pairs_from_exhaustive.main(images_dir, output_pairs)

def call_extract_features(images_dir: Path, export_dir: Path, max_keypoints: int):
    feats_conf = extract_features.confs.get("superpoint_aachen", {
        "output": "feats-superpoint",
        "model": {"name": "superpoint", "nms_radius": 3, "max_keypoints": max_keypoints},
        "preprocessing": {"grayscale": True, "resize_max": 1024},
    })
    try:
        if "model" in feats_conf and max_keypoints is not None:
            feats_conf = {**feats_conf, "model": {**feats_conf["model"], "max_keypoints": max_keypoints}}
    except Exception:
        pass

    try:
        extract_features.main(feats_conf, images_dir, export_dir)
    except TypeError:
        try:
            extract_features.main(conf=feats_conf, image_dir=images_dir, export_dir=export_dir)
        except TypeError:
            image_list = export_dir / "images.txt"
            if not image_list.exists():
                image_list.write_text("", encoding="utf-8")
            extract_features.main(feats_conf, image_list, export_dir)

    candidates = sorted(export_dir.glob("feats-*.h5"), key=lambda p: p.stat().st_mtime)
    if candidates:
        feature_path = candidates[-1]
    else:
        feature_path = export_dir / (feats_conf.get("output", "feats-superpoint") + ".h5")
    features_name = feature_path.stem
    return feature_path, features_name, feats_conf

def find_matches_file(export_dir: Path, preferred_basename: str, t0: float) -> Path:
    preferred = export_dir / (preferred_basename + ".h5")
    if preferred.exists():
        return preferred
    recents = []
    for p in export_dir.rglob("*.h5"):
        try:
            mtime = p.stat().st_mtime
        except Exception:
            continue
        name_lower = p.name.lower()
        if "match" in name_lower and mtime >= t0 - 1.0:
            recents.append((mtime, p))
    if recents:
        recents.sort()
        return recents[-1][1]
    any_matches = sorted(export_dir.rglob("matches*.h5"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if any_matches:
        return any_matches[-1]
    parent = export_dir.parent
    fallback = sorted(parent.rglob("matches*.h5"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if fallback:
        return fallback[-1]
    print("[DEBUG] Could not find matches file. Dump of .h5 under export_dir:")
    for p in export_dir.rglob("*.h5"):
        print("   -", p)
    raise FileNotFoundError("Could not locate the exported matches .h5 file.")

def call_match_features(pairs_path: Path, features_name: str, export_dir: Path):
    matches_conf = match_features.confs.get("lightglue", {
        "output": "matches-superpoint-lightglue",
        "model": {"name": "lightglue", "features": "superpoint"},
    })
    t0 = time.time()
    try:
        match_features.main(matches_conf, pairs_path, features_name, export_dir)
    except TypeError:
        try:
            match_features.main(conf=matches_conf, pairs=pairs_path, features=features_name, export_dir=export_dir)
        except TypeError:
            match_features.main(matches_conf, pairs_path, export_dir, features=features_name)
    match_path = find_matches_file(export_dir, matches_conf.get("output", "matches-superpoint-lightglue"), t0)
    return match_path, matches_conf

def call_reconstruction(sfm_dir: Path, images_dir: Path, pairs_path: Path, feature_path: Path, match_path: Path):
    try:
        return reconstruction.main(sfm_dir, images_dir, pairs_path, feature_path, match_path)
    except TypeError:
        try:
            return reconstruction.main(sfm_dir=sfm_dir, image_dir=images_dir,
                                       pairs=pairs_path, features=feature_path, matches=match_path)
        except TypeError:
            return reconstruction.main(images_dir, sfm_dir, pairs_path, feature_path, match_path)

def locate_sparse_dir(sfm_dir: Path) -> Path | None:
    candidates = []
    for d in sfm_dir.rglob("*"):
        if not d.is_dir():
            continue
        files = {q.name.lower() for q in d.glob("*")}
        if {"cameras.bin","images.bin","points3d.bin"}.issubset(files) or \
           {"cameras.bin","frames.bin","points3d.bin"}.issubset(files) or \
           {"cameras.txt","images.txt","points3d.txt"}.issubset(files):
            candidates.append(d)
    if candidates:
        candidates.sort(key=lambda p: len(p.parts))
        return candidates[0]
    files = {q.name.lower() for q in sfm_dir.glob("*")}
    if {"cameras.bin","images.bin","points3d.bin"}.issubset(files) or \
       {"cameras.bin","frames.bin","points3d.bin"}.issubset(files) or \
       {"cameras.txt","images.txt","points3d.txt"}.issubset(files):
        return sfm_dir
    return None

def move_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if src.stat().st_size == dst.stat().st_size:
                src.unlink()  # identical -> remove src to avoid duplicate
                return
        except Exception:
            pass
        dst.unlink()  # different -> overwrite
    shutil.move(str(src), str(dst))

def dedup_sparse_zero(sparse0: Path):
    """Remove .txt duplicates when a .bin with same stem exists in sparse/0."""
    if not sparse0.exists():
        return
    # Map stem -> has_bin
    has_bin = set(p.stem for p in sparse0.glob("*.bin"))
    for txt in sparse0.glob("*.txt"):
        if txt.stem in has_bin:
            try:
                txt.unlink()
                print(f"[CLEAN] removed duplicate text file: {txt.name}")
            except Exception as e:
                print(f"[WARN] failed to remove {txt}: {e}")

def finalize_to_colmap_layout(project_out_root: Path, work_dir: Path):
    # Move newest database.db (if present)
    db_candidates = list(work_dir.rglob("database.db"))
    if db_candidates:
        db_src = sorted(db_candidates, key=lambda p: p.stat().st_mtime)[-1]
        db_dst = project_out_root / "database.db"
        move_file(db_src, db_dst)
        print(f"[COLMAP] database.db -> {db_dst}")
    else:
        print("[WARN] No database.db found under hloc_work.")

    # Move sparse model files into sparse/0 without restructuring (beyond dedup rule)
    sfm_dir = work_dir / "sfm"
    sparse_src = locate_sparse_dir(sfm_dir) or locate_sparse_dir(work_dir)
    if sparse_src is None:
        print(f"[WARN] Could not find COLMAP sparse model under {sfm_dir}.")
        return

    sparse0 = project_out_root / "sparse" / "0"
    if not sparse0.exists():
        sparse0.mkdir(parents=True, exist_ok=True)

    # Move all files (flat) from sparse_src into sparse/0 (preserve filename only)
    for p in sparse_src.rglob("*"):
        if p.is_dir():
            continue
        dst = sparse0 / p.name
        move_file(p, dst)

    # Deduplicate: remove .txt if .bin of same stem exists
    dedup_sparse_zero(sparse0)

    print(f"[COLMAP] sparse -> {sparse0} (cleaned duplicates)")

def run_sfm_for_project(project_dir: Path, output_root: Path, max_keypoints: int):
    project_dir = project_dir.resolve()
    images = ensure_images_dir(project_dir)

    out_root = (output_root or project_dir).resolve()
    work_dir = out_root / "hloc_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project_dir": str(project_dir),
        "images_dir": str(images),
        "output_root": str(out_root),
        "work_dir": str(work_dir),
        "settings": {
            "max_keypoints": max_keypoints,
            "matching": "lightglue",
            "features": "superpoint",
            "pairing": "exhaustive"
        },
    }
    (work_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sfm_dir = work_dir / "sfm"
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)

    image_list_path = make_image_list_file(images, work_dir)

    # 1) Pairs (exhaustive only)
    pairs_path = work_dir / "pairs-exhaustive.txt"
    print(f"[1/4] Generating exhaustive pairs -> {pairs_path}")
    call_pairs_exhaustive(images, pairs_path, image_list_path)

    # 2) Features
    print(f"[2/4] Extracting SuperPoint features -> {work_dir}")
    feature_path, features_name, _ = call_extract_features(images, work_dir, max_keypoints)

    # 3) Matching
    print(f"[3/4] Matching with LightGlue -> {work_dir}")
    match_path, _ = call_match_features(pairs_path, features_name, work_dir)

    # 4) Reconstruction
    print(f"[4/4] Running reconstruction -> {sfm_dir}")
    call_reconstruction(sfm_dir, images, pairs_path, feature_path, match_path)

    # Finalize with clean layout (no _aux, dedup .txt vs .bin)
    finalize_to_colmap_layout(out_root, work_dir)

    print("[DONE] COLMAP-style outputs ready (clean).")
    print(f"  - Output root: {out_root}")
    print(f"  - database.db: {out_root / 'database.db'} (if present)")
    print(f"  - sparse/0:    {out_root / 'sparse' / '0'}")

def parse_args():
    p = argparse.ArgumentParser(description="Run SfM (SuperPoint+LightGlue) and export COLMAP layout (clean, no _aux).")
    p.add_argument("--project", action="append", required=True,
                   help="Path to a project directory that contains an 'images/' subfolder (COLMAP-style input). "
                        "Use multiple --project args to process more than one dataset.")
    p.add_argument("--output-root", type=str, default=None,
                   help="Where to write COLMAP-style outputs (database.db, sparse/). Defaults to the project directory.")
    p.add_argument("--max-keypoints", type=int, default=4096, help="Max keypoints for SuperPoint.")
    return p.parse_args()

def main():
    args = parse_args()
    projects = [Path(p) for p in args.project]
    output_root = Path(args.output_root).resolve() if args.output_root else None

    for proj in projects:
        print("="*80)
        print(f"Processing project: {proj}")
        run_sfm_for_project(
            project_dir=proj,
            output_root=output_root,
            max_keypoints=args.max_keypoints,
        )

if __name__ == "__main__":
    main()
