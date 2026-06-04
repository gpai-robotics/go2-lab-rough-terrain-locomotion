#!/usr/bin/env python3
"""Validate a deployment bundle manifest and referenced artifact paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = [
    "policy_name",
    "source_checkpoint",
    "task",
    "phase",
    "policy_kind",
    "deployable_observation_groups",
    "control_rate_hz",
    "exported_artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing bundle manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    missing = [field for field in REQUIRED_FIELDS if field not in manifest]
    if missing:
        raise SystemExit(f"Manifest missing required fields: {', '.join(missing)}")

    for artifact in manifest.get("exported_artifacts", []):
        artifact_path = bundle_dir / artifact
        if not artifact_path.exists():
            raise SystemExit(f"Missing exported artifact: {artifact_path}")

    print(f"Bundle is structurally valid: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
