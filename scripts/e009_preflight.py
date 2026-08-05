#!/usr/bin/env python3
"""Validate the frozen E009 Stage A training and subset contract."""

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
EXPECTED_NESTED_MANIFEST_SHA256 = (
    "9c505006109829a4046de552e36929b93f34961c1be566d5c705ae83dd5d6580"
)
CONFIG_SPECS = {
    "e009_edm2k_12000kimg.yaml": (2000, 12000, 1000),
    "e009_edm5k_12000kimg.yaml": (5000, 12000, 1000),
    "e009_edm10k_12000kimg.yaml": (10000, 12000, 1000),
    "e009_smoke_edm2k_1kimg.yaml": (2000, 1, 1),
}
ARCHIVE_FILENAMES = {
    2000: "e009_cifar10_subset_2k.zip",
    5000: "e009_cifar10_subset_5k.zip",
    10000: "e009_cifar10_subset_10k.zip",
}
EXPECTED_TRAINING_BASE = {
    "arch": "ddpmpp",
    "batch_size": 64,
    "batch_gpu": 64,
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
    """Load the mapping-only YAML subset used by E009 configs."""
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
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *arguments: str) -> str:
    """Run one read-only Git query."""
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_repository(repo_root: Path, expected_commit: str) -> Path:
    """Require the explicit absolute clean checkout and exact commit."""
    if not repo_root.is_absolute() or not repo_root.is_dir():
        raise ValueError(f"Repository root must be an absolute directory: {repo_root}")
    resolved = repo_root.resolve()
    if Path(git_output(resolved, "rev-parse", "--show-toplevel")).resolve() != resolved:
        raise ValueError("E009_REPO_ROOT is not the Git top level")
    actual_commit = git_output(resolved, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"Repository commit mismatch: {actual_commit} != {expected_commit}"
        )
    status = git_output(resolved, "status", "--porcelain=v1")
    if status:
        raise ValueError(f"Repository worktree is not clean:\n{status}")
    return resolved


def validate_protocol(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate frozen subset identities and pilot/decision boundaries."""
    protocol = json.loads(
        (repo_root / "configs/e009_stage_a_protocol.json").read_text()
    )
    manifest_path = repo_root / "data/e009_nested_subsets_manifest.json"
    if sha256_file(manifest_path) != EXPECTED_NESTED_MANIFEST_SHA256:
        raise ValueError("Nested subset manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if protocol["stage_a"]["dataset_sizes"] != [2000, 5000, 10000]:
        raise ValueError("Stage A dataset sizes changed")
    if protocol["pilot"]["seeds"] != {
        "start": 20000,
        "stop_inclusive": 20127,
        "count": 128,
    }:
        raise ValueError("E009 pilot seeds changed")
    if protocol["pilot"]["excluded_seed_ranges"] != [[0, 255], [10000, 10127]]:
        raise ValueError("Reserved seed exclusions changed")
    if protocol["pilot"]["eligible_count_interval_inclusive"] != [13, 115]:
        raise ValueError("Eligibility interval changed")
    for size_string, record in manifest["subsets"].items():
        index_path = repo_root / "data" / record["index_manifest"]
        if sha256_file(index_path) != record["index_manifest_sha256"]:
            raise ValueError(f"Index manifest hash mismatch for size {size_string}")
    return protocol, manifest


def validate_config(
    repo_root: Path,
    config_path: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Require one of the four exact matched Stage A/smoke configurations."""
    try:
        size, duration, tick = CONFIG_SPECS[config_path.name]
    except KeyError as error:
        raise ValueError(
            f"Config is not an approved E009 config: {config_path}"
        ) from error
    expected_path = repo_root / "configs" / config_path.name
    if config_path.resolve() != expected_path.resolve():
        raise ValueError(f"Config must be the tracked repository file: {expected_path}")
    relative_path = str(expected_path.relative_to(repo_root))
    expected_hash = protocol["artifact_hashes"]["config_sha256"][relative_path]
    if sha256_file(config_path) != expected_hash:
        raise ValueError(f"Config hash mismatch: {config_path}")
    config = load_simple_yaml(config_path)
    experiment = config["experiment"]
    dataset = config["dataset"]
    training = config["training"]

    expected_name = (
        "e009_smoke_edm2k_1kimg"
        if duration == 1
        else f"e009_edm{size // 1000}k_12000kimg"
    )
    expected_archive = (
        Path("/home/xggh8/data/zw-lab/e009_stage_a_subsets") / ARCHIVE_FILENAMES[size]
    )
    if experiment != {
        "name": expected_name,
        "seed": 0,
        "edm_root": "/home/xggh8/edm",
        "persistent_scratch_root": (
            "/cluster/pixstor/zwggh-lab/xinyu/e009_training_scratch"
        ),
        "persistent_data_root": "/home/xggh8/data/zw-lab",
        "run_dir_root": "scratch",
        "stage_mode": "full",
    }:
        raise ValueError("Experiment block differs from the frozen E009 contract")
    if dataset != {
        "path": str(expected_archive),
        "resolution": 32,
        "use_labels": False,
        "cache": True,
        "subset_size": size,
        "subset_seed": 0,
    }:
        raise ValueError("Dataset block differs from the frozen E009 contract")
    expected_training = dict(EXPECTED_TRAINING_BASE)
    expected_training.update(
        {
            "duration_kimg": duration,
            "tick_kimg": tick,
            "snapshot_ticks": 1,
            "state_dump_ticks": 1,
        }
    )
    if training != expected_training:
        raise ValueError("Training block differs from the frozen E009 contract")
    if manifest["subsets"][str(size)]["subset_archive"] != str(expected_archive):
        raise ValueError("Config archive path differs from the subset manifest")
    return config


def directory_has_entries(path: Path) -> bool:
    """Return whether a directory exists and is nonempty."""
    return path.exists() and any(path.iterdir())


def validate_output_state(config: dict[str, Any], mode: str) -> None:
    """Prevent implicit resume and output collisions."""
    name = config["experiment"]["name"]
    scratch = Path(config["experiment"]["persistent_scratch_root"]) / name
    final = Path(config["experiment"]["persistent_data_root"]) / name
    populated = [path for path in (scratch, final) if directory_has_entries(path)]
    if mode == "fresh" and populated:
        raise ValueError(f"Fresh mode output collision: {populated}")
    if mode == "resume":
        states = [
            path
            for root in (scratch, final)
            for path in root.glob("training-state-*.pt")
            if path.stat().st_size > 0
        ]
        if not states:
            raise ValueError("Resume mode requires a nonempty training state")


def require_hash(path: Path, expected: str, label: str) -> None:
    """Require one exact external artifact hash."""
    actual = sha256_file(path)
    print(f"{label}_sha256={actual}")
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def validate_external(
    config: dict[str, Any], manifest: dict[str, Any], mode: str
) -> None:
    """Validate EDM source, wrapper, subset archive, and output state."""
    edm_root = Path(config["experiment"]["edm_root"])
    wrapper = Path("/home/xggh8/projects/zw-lab/src/zwlab_edm/train_subset_sigma.py")
    commit = git_output(edm_root, "rev-parse", "HEAD")
    if commit != EXPECTED_EDM_COMMIT:
        raise ValueError(f"EDM commit mismatch: {commit}")
    for relative_path, expected in EXPECTED_SOURCE_HASHES.items():
        require_hash(edm_root / relative_path, expected, relative_path)
    require_hash(wrapper, EXPECTED_WRAPPER_SHA256, "wrapper")
    size = str(config["dataset"]["subset_size"])
    require_hash(
        Path(config["dataset"]["path"]),
        manifest["subsets"][size]["subset_archive_sha256"],
        f"subset_archive_{size}",
    )
    validate_output_state(config, mode)


def parse_args() -> argparse.Namespace:
    """Parse E009 preflight arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run repository, protocol, config, and external provenance checks."""
    args = parse_args()
    repo_root = validate_repository(args.repo_root, args.expected_repo_commit)
    protocol, manifest = validate_protocol(repo_root)
    config = validate_config(repo_root, args.config, protocol, manifest)
    print(f"config_sha256={sha256_file(args.config)}")
    if not args.config_only:
        validate_external(config, manifest, args.mode)
    print(f"preflight=pass mode={args.mode} config_only={args.config_only}")


if __name__ == "__main__":
    main()
