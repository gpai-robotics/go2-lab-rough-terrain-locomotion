#!/usr/bin/env python3
"""Create or refresh a local deployment bundle manifest for the Go2 lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2_deploy_contract import build_manifest, default_bundle_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-name", required=True, help="Frozen candidate name.")
    parser.add_argument("--source-checkpoint", required=True, help="Training checkpoint used for export.")
    parser.add_argument("--task", required=True, help="Registered IsaacLab task id.")
    parser.add_argument("--phase", required=True, help="Training lineage or experiment phase label.")
    parser.add_argument(
        "--bundle-dir",
        default="",
        help="Bundle directory. Defaults to artifacts/exported/<policy-name> inside this repo.",
    )
    parser.add_argument("--freeze-note", default="", help="Optional note recorded in the manifest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else default_bundle_dir(args.policy_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        policy_name=args.policy_name,
        source_checkpoint=str(Path(args.source_checkpoint).expanduser().resolve()),
        task=args.task,
        phase=args.phase,
        freeze_note=args.freeze_note,
    )
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote bundle manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
