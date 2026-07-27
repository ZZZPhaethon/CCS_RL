"""Extract learned future-residual gate scales from adapter checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.run_root).glob("*/model_seed_*/iterative_action_q.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        configuration = checkpoint["configuration"]
        raw_scale = float(checkpoint["model_state_dict"]["future_scale"])
        limit = float(configuration["future_residual_scale_limit"])
        rows.append(
            {
                "candidate": path.parents[1].name,
                "model_seed": int(path.parent.name.rsplit("_", 1)[1]),
                "raw_scale": raw_scale,
                "effective_scale": limit * math.tanh(raw_scale),
            }
        )
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
