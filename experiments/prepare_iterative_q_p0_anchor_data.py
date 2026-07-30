"""Add the Greedy/FOLLOW P0 action as an explicit teacher anchor in G0 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = (
    "states",
    "actions",
    "return_to_go",
    "scenario_seed",
    "root_time_h",
)
OPTIONAL_FIELDS = ("future_summaries", "future_forecasts")


def prepare(source: Path, output: Path) -> None:
    with np.load(source, allow_pickle=False) as loaded:
        arrays = {field: loaded[field].copy() for field in REQUIRED_FIELDS}
        for field in OPTIONAL_FIELDS:
            if field in loaded:
                arrays[field] = loaded[field].copy()
        metadata = json.loads(str(loaded["metadata_json"]))

    follow_action = int(metadata["follow_action_index"])
    if np.any(arrays["actions"][:, 0] == follow_action):
        raise ValueError(f"{source}: G0 already contains the FOLLOW action")

    keys = np.stack(
        (arrays["scenario_seed"], arrays["root_time_h"]), axis=1
    )
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    first_indices.sort()
    root_count = len(first_indices)

    prepared = {}
    for field, values in arrays.items():
        if field == "actions":
            anchor_rows = np.full(
                (root_count, *values.shape[1:]),
                follow_action,
                dtype=values.dtype,
            )
        elif field == "return_to_go":
            anchor_rows = np.zeros(
                (root_count, *values.shape[1:]), dtype=values.dtype
            )
        else:
            anchor_rows = values[first_indices].copy()
        prepared[field] = np.concatenate((values, anchor_rows), axis=0)

    prepared["anchor_action"] = np.full(
        len(prepared["actions"]), follow_action, dtype=np.int16
    )
    metadata.update(
        {
            "anchors_in_data": True,
            "anchor_policy": "greedy_follow_p0",
            "source_data": str(source),
            "source_rows": int(len(arrays["actions"])),
            "anchor_rows_added": int(root_count),
            "prepared_rows": int(len(prepared["actions"])),
        }
    )
    prepared["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
