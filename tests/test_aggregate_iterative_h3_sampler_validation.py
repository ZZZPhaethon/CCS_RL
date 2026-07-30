import csv
import json

from experiments.aggregate_iterative_h3_sampler_validation import (
    GATES,
    MODEL_SEEDS,
    VALIDATION_SEEDS,
    VARIANTS,
    parse_args,
    run,
)


def test_p2_aggregation_selects_lower_cost_reweight_route(tmp_path):
    offsets = {
        "b_gate_only": 0.0,
        "c_dedup_balanced": -100.0,
        "d_dedup_advantage": 50.0,
    }
    fields = [
        "gate",
        "seed",
        "delta_total_cost_eur",
        "vented_t",
        "override_events",
        "stored_t",
        "unit_cost_eur_per_t",
    ]
    for variant in VARIANTS:
        for model_seed in MODEL_SEEDS:
            out_dir = (
                tmp_path
                / "validation"
                / variant
                / f"model_seed_{model_seed}"
            )
            out_dir.mkdir(parents=True)
            (out_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "validation_only": True,
                        "eval_seeds": list(VALIDATION_SEEDS),
                    }
                ),
                encoding="utf-8",
            )
            with (out_dir / "evaluation.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for gate in GATES:
                    for seed_index, seed in enumerate(VALIDATION_SEEDS):
                        writer.writerow(
                            {
                                "gate": gate,
                                "seed": seed,
                                "delta_total_cost_eur": (
                                    offsets[variant]
                                    + model_seed
                                    + seed_index
                                ),
                                "vented_t": 0.0,
                                "override_events": 2.0,
                                "stored_t": 1.0,
                                "unit_cost_eur_per_t": 1.0,
                            }
                        )

    result = run(parse_args(["--run-root", str(tmp_path)]))

    assert (
        result["selection"]["selected_reweight_variant"]
        == "c_dedup_balanced"
    )
    assert (tmp_path / "p2_validation_metrics.csv").is_file()
    assert (
        tmp_path
        / "figures"
        / "p2_validation_sampling_comparison.svg"
    ).is_file()
