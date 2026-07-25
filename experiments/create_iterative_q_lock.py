"""Create an immutable iterative-Q policy lock for roll-in generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


WINDOWS_H = [
    [108, 179],
    [180, 251],
    [252, 323],
    [324, 395],
    [396, 467],
    [468, 539],
    [540, 611],
    [612, 680],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--residual-margin", type=float, required=True)
    parser.add_argument("--economic-margin-eur", type=float, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    out_path = Path(args.out_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if out_path.exists():
        raise FileExistsError(out_path)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_configuration = checkpoint_payload.get("configuration", {})
    if model_configuration.get("q_head") != "iterative_action_q":
        raise ValueError("checkpoint is not an iterative state-only Q model")

    payload = {
        "protocol_id": args.protocol_id,
        "locked_checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "observation_input": model_configuration.get(
            "observation_input", "state_only"
        ),
        "policy": {
            "required_heads": 4,
            "residual_margin": args.residual_margin,
            "economic_margin_eur": args.economic_margin_eur,
            "max_overrides": 8,
            "one_override_per_window": True,
            "windows_h": WINDOWS_H,
        },
        "uses_mpc_for_training_or_selection": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} sha256={payload['checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    main()
