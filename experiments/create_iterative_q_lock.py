"""Create an immutable iterative-Q policy lock for roll-in generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


DEFAULT_WINDOWS_H = [
    [108, 179],
    [180, 251],
    [252, 323],
    [324, 395],
    [396, 467],
    [468, 539],
    [540, 611],
    [612, 680],
]


def parse_windows_h(value: str) -> list[list[int]]:
    windows = []
    for item in value.split(","):
        try:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "windows must be comma-separated START-END pairs"
            ) from exc
        if start < 0 or end < start:
            raise argparse.ArgumentTypeError(
                "window bounds must satisfy 0 <= START <= END"
            )
        windows.append([start, end])
    if not windows:
        raise argparse.ArgumentTypeError("at least one policy window is required")
    if any(
        current[0] <= previous[1]
        for previous, current in zip(windows, windows[1:])
    ):
        raise argparse.ArgumentTypeError(
            "policy windows must be ordered and non-overlapping"
        )
    return windows


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--residual-margin", type=float, required=True)
    parser.add_argument("--economic-margin-eur", type=float, required=True)
    parser.add_argument("--max-overrides", type=int, default=8)
    parser.add_argument(
        "--windows-h",
        type=parse_windows_h,
        default=[list(window) for window in DEFAULT_WINDOWS_H],
    )
    args = parser.parse_args(argv)
    if args.max_overrides <= 0:
        parser.error("max overrides must be positive")
    if args.max_overrides > len(args.windows_h):
        parser.error("max overrides cannot exceed the policy window count")
    return args


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
    if model_configuration.get("q_head") not in {
        "iterative_action_q",
        "iterative_action_q_future_v4_24_72",
        "iterative_action_q_future_summary",
    }:
        raise ValueError("checkpoint is not an iterative Q model")

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
            "max_overrides": int(args.max_overrides),
            "one_override_per_window": True,
            "windows_h": args.windows_h,
        },
        "uses_mpc_for_training_or_selection": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} sha256={payload['checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    main()
