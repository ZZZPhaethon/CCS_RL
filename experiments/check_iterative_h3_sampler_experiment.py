"""Validate the locked inputs for the H3 iterative-Q sampler experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


EXPECTED_VALIDATION_SEEDS = list(range(8100001, 8100021))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--seed-checkpoint-root", required=True)
    parser.add_argument("--out-path", required=True)
    return parser.parse_args(argv)


def _checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    configuration = payload.get("configuration", {})
    if int(configuration.get("heads", 0)) != 5:
        raise ValueError(f"expected a five-head checkpoint: {path}")
    if configuration.get("observation_input") != "shared_future_summary":
        raise ValueError(
            f"expected shared-future checkpoint input: {path}"
        )
    return {
        "path": str(path),
        "heads": int(configuration["heads"]),
        "observation_input": configuration["observation_input"],
    }


def run(args):
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("formal_test_access") is not False:
        raise ValueError("protocol must forbid formal-test access")
    if protocol.get("formal_test_stage_included") is not False:
        raise ValueError("formal-test stage must be omitted")
    if protocol.get("controller_validation_seeds") != (
        EXPECTED_VALIDATION_SEEDS
    ):
        raise ValueError("controller validation seeds are not locked")
    gate = protocol["collection_gate"]
    if (
        int(gate["required_heads"]) != 3
        or int(gate["ensemble_heads"]) != 5
        or float(gate["residual_margin"]) != 0.4
    ):
        raise ValueError("collection gate must be fixed at H3/M0.4")
    source_run = Path(args.source_run)
    for split in ("train", "validation"):
        path = source_run / "g0" / f"{split}_merged.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoints = {
        "model_seed_0": _checkpoint(
            source_run / "p1" / "iterative_action_q.pt"
        )
    }
    seed_root = Path(args.seed_checkpoint_root)
    for model_seed in (1, 2):
        checkpoints[f"model_seed_{model_seed}"] = _checkpoint(
            seed_root
            / f"model_seed_{model_seed}"
            / "p1"
            / "iterative_action_q.pt"
        )
    result = {
        "kind": "iterative_h3_sampler_environment_check",
        "protocol_id": protocol["protocol_id"],
        "formal_test_access": False,
        "controller_validation_seed_count": len(
            EXPECTED_VALIDATION_SEEDS
        ),
        "collection_gate": gate,
        "checkpoints": checkpoints,
        "torch_version": torch.__version__,
        "cuda_available_on_check_node": bool(torch.cuda.is_available()),
    }
    out_path = Path(args.out_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
