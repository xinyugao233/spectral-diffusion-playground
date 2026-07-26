#!/usr/bin/env python3
"""Validate the frozen E005 matched EDM-50K training contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_EDM_COMMIT = "008a4e5316c8e3bfe61a62f874bddba254295afb"
EXPECTED_WRAPPER_SHA256 = (
    "f5ca28db4a167ba5ee6c26bf9ae9cf4bf3215b919a86e7ad8e7f8ae0ad10c142"
)
EXPECTED_ARCHIVE_SHA256 = (
    "795cdc1444465ae4e19e25a0615d05ba0a0e83caa5db6b1b811deaf4c7910dfa"
)
EXPECTED_SOURCE_HASHES = {
    "train.py": "e562dacd2f403e4a9dfe8c857078bb506d8d6825cb88478869cc78b7c4587f05",
    "training/dataset.py": (
        "fd4a37cdcca57563d2c20e7dc22da5b59bc7f15d1f53f87b902bf16c0917c05b"
    ),
    "training/loss.py": (
        "b26f0937bc72b18cc5f9869c86035ca9b872138cdfae7994237e9733951e4415"
    ),
    "training/networks.py": (
        "5db27dcd96674b95c72d5e6491b879cdc35e24039ada3411b4b46a28ed1fe284"
    ),
    "training/training_loop.py": (
        "9cac3720de1bd122a5fb735a133707fbe708daa454e00232390112311ee77391"
    ),
}
REQUIRED_REPOSITORY_PATHS = (
    "scripts/e005_preflight.py",
    "configs/e005_edm50k_matched_40000kimg.yaml",
    "configs/e005_edm50k_matched_40000kimg_manifest.json",
)

EXPECTED_CONFIG = {
    "experiment": {
        "name": "e005_edm50k_matched_40000kimg",
        "seed": 0,
        "edm_root": "/home/xggh8/edm",
        "persistent_scratch_root": "/home/xggh8/scratch/zw-lab",
        "persistent_data_root": "/home/xggh8/data/zw-lab",
        "stage_mode": "full",
    },
    "dataset": {
        "path": "/home/xggh8/datasets/edm/cifar10-32x32-train50k.zip",
        "resolution": 32,
        "use_labels": False,
        "cache": True,
        "subset_size": 50000,
        "subset_seed": 0,
    },
    "training": {
        "arch": "ddpmpp",
        "batch_size": 64,
        "batch_gpu": 64,
        "duration_kimg": 40000,
        "tick_kimg": 1000,
        "snapshot_ticks": 2,
        "state_dump_ticks": 4,
        "lr": 0.001,
        "ema_halflife_kimg": 500,
        "dropout": 0.13,
        "use_fp16": False,
        "xflip": False,
        "workers": 1,
        "model_channels": 128,
        "channel_mult": [2, 2, 2],
        "sigma_data": 0.5,
        "p_mean": -1.2,
        "p_std": 1.2,
    },
}


def parse_scalar(value: str) -> Any:
    """Parse one scalar from the constrained training YAML."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the mapping-only YAML subset used by the frozen config."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            raise ValueError(f"Cannot parse config line: {raw_line!r}")
        while indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        value = value.strip()
        if not value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = parse_scalar(value)
    return root


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config_path: Path) -> dict[str, Any]:
    """Require the complete frozen EDM-50K configuration."""
    config = load_simple_yaml(config_path)
    if config != EXPECTED_CONFIG:
        raise ValueError("Config does not equal the frozen matched EDM-50K contract")
    return config


def validate_manifest(config_path: Path, manifest_path: Path) -> None:
    """Require the manifest to bind the exact config and source identities."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_config_hash = manifest["config"]["sha256"]
    actual_config_hash = sha256_file(config_path)
    if actual_config_hash != expected_config_hash:
        raise ValueError(
            f"Config SHA-256 mismatch: {actual_config_hash} != {expected_config_hash}"
        )
    if manifest["archive"]["sha256"] != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Manifest archive hash does not match the frozen archive")
    if manifest["source"]["edm_commit"] != EXPECTED_EDM_COMMIT:
        raise ValueError("Manifest EDM commit does not match the frozen source")
    if manifest["source"]["wrapper_sha256"] != EXPECTED_WRAPPER_SHA256:
        raise ValueError("Manifest wrapper hash does not match the frozen source")
    if manifest["source"]["file_sha256"] != EXPECTED_SOURCE_HASHES:
        raise ValueError("Manifest EDM source hashes do not match the frozen source")


def git_output(repo_root: Path, *arguments: str) -> str:
    """Run one read-only Git query in the selected repository."""
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_repository(repo_root: Path, expected_commit: str) -> Path:
    """Require an absolute, clean checkout at the explicitly selected commit."""
    if not repo_root.is_absolute():
        raise ValueError(f"Repository root must be absolute: {repo_root}")
    if not repo_root.is_dir():
        raise ValueError(f"Repository root does not exist: {repo_root}")
    resolved_root = repo_root.resolve()
    for relative_path in REQUIRED_REPOSITORY_PATHS:
        path = resolved_root / relative_path
        if not path.is_file():
            raise ValueError(f"Required repository file is missing: {path}")
    if not (resolved_root / ".git").exists():
        raise ValueError(f"Expected Git repository at: {resolved_root}")
    top_level = Path(git_output(resolved_root, "rev-parse", "--show-toplevel"))
    if top_level.resolve() != resolved_root:
        raise ValueError(f"Git top level differs from repository root: {top_level}")
    actual_commit = git_output(resolved_root, "rev-parse", "HEAD")
    print(f"repository_root={resolved_root}")
    print(f"repository_commit={actual_commit}")
    if actual_commit != expected_commit:
        raise ValueError(
            f"Repository commit mismatch: {actual_commit} != {expected_commit}"
        )
    status = git_output(resolved_root, "status", "--porcelain=v1")
    if status:
        raise ValueError(f"Repository worktree is not clean:\n{status}")
    return resolved_root


def require_hash(path: Path, expected: str, label: str) -> None:
    """Raise when one external artifact differs from its frozen hash."""
    actual = sha256_file(path)
    print(f"{label}_sha256={actual}")
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def directory_has_entries(path: Path) -> bool:
    """Return whether a directory exists and contains any entry."""
    return path.exists() and any(path.iterdir())


def validate_output_state(config: dict[str, Any], mode: str) -> None:
    """Prevent implicit resume and accidental output reuse."""
    name = config["experiment"]["name"]
    scratch = Path(config["experiment"]["persistent_scratch_root"]) / name
    final = Path(config["experiment"]["persistent_data_root"]) / name
    populated = [path for path in (scratch, final) if directory_has_entries(path)]
    if mode == "fresh" and populated:
        joined = ", ".join(str(path) for path in populated)
        raise ValueError(f"Fresh mode requires empty or absent output dirs: {joined}")
    if mode == "resume":
        state_files = [
            path
            for directory in (scratch, final)
            for path in directory.glob("training-state-*.pt")
            if path.is_file() and path.stat().st_size > 0
        ]
        if not state_files:
            raise ValueError("Resume mode requires an existing nonempty training state")


def validate_external_artifacts(config: dict[str, Any], mode: str) -> None:
    """Validate the remote source, archive, wrapper, and output state."""
    edm_root = Path(config["experiment"]["edm_root"])
    wrapper = Path("/home/xggh8/projects/zw-lab/src/zwlab_edm/train_subset_sigma.py")
    archive = Path(config["dataset"]["path"])
    commit = subprocess.run(
        ["git", "-C", str(edm_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"edm_commit={commit}")
    if commit != EXPECTED_EDM_COMMIT:
        raise ValueError(f"EDM commit mismatch: {commit} != {EXPECTED_EDM_COMMIT}")
    for relative_path, expected_hash in EXPECTED_SOURCE_HASHES.items():
        require_hash(edm_root / relative_path, expected_hash, relative_path)
    require_hash(wrapper, EXPECTED_WRAPPER_SHA256, "wrapper")
    require_hash(archive, EXPECTED_ARCHIVE_SHA256, "archive")
    validate_output_state(config, mode)


def parse_args() -> argparse.Namespace:
    """Parse preflight arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate tracked artifacts without accessing Hellbender paths.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the frozen configuration and provenance checks."""
    args = parse_args()
    repo_root = validate_repository(args.repo_root, args.expected_repo_commit)
    expected_config = repo_root / "configs/e005_edm50k_matched_40000kimg.yaml"
    expected_manifest = (
        repo_root / "configs/e005_edm50k_matched_40000kimg_manifest.json"
    )
    if args.config.resolve() != expected_config:
        raise ValueError(f"Config is not the frozen repository file: {args.config}")
    if args.manifest.resolve() != expected_manifest:
        raise ValueError(f"Manifest is not the frozen repository file: {args.manifest}")
    config = validate_config(args.config)
    validate_manifest(args.config, args.manifest)
    print(f"config_sha256={sha256_file(args.config)}")
    if not args.config_only:
        validate_external_artifacts(config, args.mode)
    print(f"preflight=pass mode={args.mode} config_only={args.config_only}")


if __name__ == "__main__":
    main()
