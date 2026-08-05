#!/usr/bin/env python3
"""Validate the frozen E009 Stage B warm-start smoke contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_EDM_COMMIT = "008a4e5316c8e3bfe61a62f874bddba254295afb"
EXPECTED_ZWLAB_COMMIT = "30dd713b1b062b5f2f71f4b71a6d1fa66791d550"
EXPECTED_EDM_LOOP_SHA256 = (
    "9cac3720de1bd122a5fb735a133707fbe708daa454e00232390112311ee77391"
)
EXPECTED_WRAPPER_SHA256 = (
    "f5ca28db4a167ba5ee6c26bf9ae9cf4bf3215b919a86e7ad8e7f8ae0ad10c142"
)
EXPECTED_STAGE_A_ROOT = Path("/home/xggh8/data/zw-lab/e009_edm5k_12000kimg")
EXPECTED_PARENT_STATE_SHA256 = (
    "1073e68c9f45123b53811a12a56a565f296a5ab846212d22e5027bbd81d685f5"
)
EXPECTED_PARENT_EMA_SHA256 = (
    "a77c19588f9a4f877de961102c16901ee07bbd87e0e4ace6164f92f40c406d58"
)
EXPECTED_ARCHIVE_SHA256 = (
    "1e96a4f7a701bd067f71c725bbe83f1dcd65a750b310f206eee878ce2c07355a"
)
EXPECTED_LINEAGE = "e009_stage_b_edm5k_30000kimg"
EXPECTED_STAGE_A_METADATA = {
    "config_used.yaml": "e4a076c301e3e330872a6088774a5c7d688a18b0639831f5e250d759282868d8",
    "training_options.json": (
        "ca39f38ebb94ee78e2a65ac1f2065efe22c5aa915af3581a2c8e469a649a652f"
    ),
    "run_manifest.json": (
        "d63dbddaaf06422462e8268ffd40a69459947ee35f82ee1887aaab0b9882c81a"
    ),
}
EXPECTED_EDM_SOURCE_HASHES = {
    "training/dataset.py": (
        "fd4a37cdcca57563d2c20e7dc22da5b59bc7f15d1f53f87b902bf16c0917c05b"
    ),
    "training/loss.py": (
        "b26f0937bc72b18cc5f9869c86035ca9b872138cdfae7994237e9733951e4415"
    ),
    "training/networks.py": (
        "5db27dcd96674b95c72d5e6491b879cdc35e24039ada3411b4b46a28ed1fe284"
    ),
    "training/training_loop.py": EXPECTED_EDM_LOOP_SHA256,
}
APPROVED_CONFIGS = {
    "e009_stage_b_edm5k_13000kimg_smoke.yaml": 13000,
    "e009_stage_b_edm5k_30000kimg.yaml": 30000,
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_scalar(value: str) -> Any:
    """Parse one scalar from the constrained Stage B YAML."""
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
    """Load the mapping-only YAML subset used by Stage B configs."""
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
        if value:
            current[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
    return root


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
        raise ValueError("Repository root must be an absolute existing directory")
    resolved = repo_root.resolve()
    if Path(git_output(resolved, "rev-parse", "--show-toplevel")).resolve() != resolved:
        raise ValueError("E009_REPO_ROOT is not the Git top level")
    actual = git_output(resolved, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise ValueError(f"Repository commit mismatch: {actual} != {expected_commit}")
    status = git_output(resolved, "status", "--porcelain=v1")
    if status:
        raise ValueError(f"Repository worktree is not clean:\n{status}")
    return resolved


def require_hash(path: Path, expected: str, label: str) -> None:
    """Require one exact external artifact hash."""
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    print(f"{label}_sha256={actual}")
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def directory_manifest(root: Path) -> dict[str, Any]:
    """Hash every regular file without following symbolic links."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Stage A root must be a real directory: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Stage A manifest refuses symbolic link: {path}")
        if not path.is_file():
            continue
        stat = path.stat()
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        )
    payload = {"root": str(root), "records": records}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def validate_protocol(repo_root: Path) -> dict[str, Any]:
    """Require the amended warm-start and unchanged downstream rules."""
    protocol = json.loads(
        (repo_root / "configs/e009_stage_b_protocol.json").read_text()
    )
    amendment = protocol["warm_start_amendment"]
    if amendment["warm_start"] is not True or amendment["exact_continuation"]:
        raise ValueError("Stage B warm-start interpretation changed")
    if amendment["rng_seed"] != 1 or amendment["starting_exposure_kimg"] != 12000:
        raise ValueError("Stage B seed or starting exposure changed")
    if protocol["evaluation"]["seeds"] != {
        "start": 20000,
        "stop_inclusive": 20127,
        "count": 128,
    }:
        raise ValueError("Stage B evaluation seeds changed")
    if protocol["evaluation"]["excluded_seed_ranges"] != [
        [0, 255],
        [10000, 10127],
    ]:
        raise ValueError("Reserved seed boundaries changed")
    if protocol["evaluation"]["eligible_count_interval_inclusive"] != [13, 115]:
        raise ValueError("Eligibility rule changed")
    return protocol


def validate_config(repo_root: Path, config_path: Path) -> dict[str, Any]:
    """Require one tracked Stage B config and its exact scientific fields."""
    if config_path.name not in APPROVED_CONFIGS:
        raise ValueError(f"Unapproved Stage B config: {config_path.name}")
    expected_path = repo_root / "configs" / config_path.name
    if config_path.resolve() != expected_path.resolve():
        raise ValueError(f"Config must be tracked at {expected_path}")
    protocol = validate_protocol(repo_root)
    expected_hash = protocol["artifact_hashes"][
        str(config_path.resolve().relative_to(repo_root))
    ]
    require_hash(config_path, expected_hash, "stage_b_config")
    config = load_simple_yaml(config_path)
    experiment = config["experiment"]
    dataset = config["dataset"]
    training = config["training"]
    warm = config["warm_start"]
    if experiment != {
        "name": EXPECTED_LINEAGE,
        "seed": 1,
        "edm_root": "/home/xggh8/edm",
        "persistent_scratch_root": (
            "/cluster/pixstor/zwggh-lab/xinyu/e009_training_scratch"
        ),
        "persistent_data_root": "/home/xggh8/data/zw-lab",
        "run_dir_root": "scratch",
        "stage_mode": "full",
        "warm_start": True,
    }:
        raise ValueError("Stage B experiment block changed")
    if dataset != {
        "path": (
            "/home/xggh8/data/zw-lab/e009_stage_a_subsets/" "e009_cifar10_subset_5k.zip"
        ),
        "resolution": 32,
        "use_labels": False,
        "cache": True,
        "subset_size": 5000,
        "subset_seed": 0,
    }:
        raise ValueError("Stage B dataset block changed")
    expected_training = {
        "arch": "ddpmpp",
        "batch_size": 64,
        "batch_gpu": 64,
        "duration_kimg": APPROVED_CONFIGS[config_path.name],
        "tick_kimg": 1000,
        "snapshot_ticks": 1,
        "state_dump_ticks": 1,
        "lr": 0.001,
        "ema_halflife_kimg": 500,
        "dropout": 0.13,
        "use_fp16": False,
        "xflip": False,
        "workers": 0,
        "model_channels": 128,
        "channel_mult": [2, 2, 2],
        "sigma_data": 0.5,
        "p_mean": -1.2,
        "p_std": 1.2,
    }
    if training != expected_training:
        raise ValueError("Stage B training block changed")
    if warm != {
        "enabled": True,
        "seed": 1,
        "start_kimg": 12000,
        "parent_training_state": str(
            EXPECTED_STAGE_A_ROOT / "training-state-012000.pt"
        ),
        "parent_training_state_sha256": EXPECTED_PARENT_STATE_SHA256,
        "parent_ema_snapshot": str(
            EXPECTED_STAGE_A_ROOT / "network-snapshot-012000.pkl"
        ),
        "parent_ema_snapshot_sha256": EXPECTED_PARENT_EMA_SHA256,
        "state_schema_version": 2,
    }:
        raise ValueError("Stage B warm-start block changed")
    return config


def validate_external(config: dict[str, Any]) -> None:
    """Validate source, archive, parent identities, and output isolation."""
    edm_root = Path(config["experiment"]["edm_root"])
    if git_output(edm_root, "rev-parse", "HEAD") != EXPECTED_EDM_COMMIT:
        raise ValueError("EDM source commit mismatch")
    for relative_path, expected in EXPECTED_EDM_SOURCE_HASHES.items():
        require_hash(edm_root / relative_path, expected, relative_path)
    zwlab_root = Path("/home/xggh8/projects/zw-lab")
    if git_output(zwlab_root, "rev-parse", "HEAD") != EXPECTED_ZWLAB_COMMIT:
        raise ValueError("zw-lab source commit mismatch")
    require_hash(
        Path("/home/xggh8/projects/zw-lab/src/zwlab_edm/train_subset_sigma.py"),
        EXPECTED_WRAPPER_SHA256,
        "zwlab_wrapper",
    )
    require_hash(Path(config["dataset"]["path"]), EXPECTED_ARCHIVE_SHA256, "archive")
    warm = config["warm_start"]
    require_hash(
        Path(warm["parent_training_state"]),
        EXPECTED_PARENT_STATE_SHA256,
        "parent_training_state",
    )
    require_hash(
        Path(warm["parent_ema_snapshot"]),
        EXPECTED_PARENT_EMA_SHA256,
        "parent_ema_snapshot",
    )
    for filename, expected in EXPECTED_STAGE_A_METADATA.items():
        require_hash(EXPECTED_STAGE_A_ROOT / filename, expected, f"stage_a_{filename}")
    name = config["experiment"]["name"]
    output_paths = [
        Path(config["experiment"]["persistent_scratch_root"]) / name,
        Path(config["experiment"]["persistent_data_root"]) / name,
    ]
    collisions = [path for path in output_paths if path.exists()]
    if collisions:
        raise ValueError(f"Immutable Stage B output collision: {collisions}")


def parse_args() -> argparse.Namespace:
    """Parse Stage B preflight arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-a-manifest-output", type=Path)
    parser.add_argument("--config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run all prospective Stage B smoke gates."""
    args = parse_args()
    repo_root = validate_repository(args.repo_root, args.expected_repo_commit)
    validate_protocol(repo_root)
    config = validate_config(repo_root, args.config)
    print(f"config_sha256={sha256_file(args.config)}")
    if not args.config_only:
        validate_external(config)
        if args.stage_a_manifest_output is None:
            raise ValueError("Full preflight requires --stage-a-manifest-output")
        manifest = directory_manifest(EXPECTED_STAGE_A_ROOT)
        args.stage_a_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.stage_a_manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(f"stage_a_manifest_sha256={manifest['manifest_sha256']}")
    print(f"preflight=pass config_only={args.config_only}")


if __name__ == "__main__":
    main()
