from experiments import evaluate_reference_policies as subject


def test_parse_args_accepts_unique_seeds_and_policies(tmp_path):
    args = subject.parse_args(
        [
            "--eval-seeds",
            "130",
            "131",
            "--policies",
            "mpc",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert args.eval_seeds == [130, 131]
    assert args.policies == ["mpc"]


def test_policy_names_match_reference_rows():
    assert subject.POLICY_NAMES == {
        "greedy": "greedy",
        "mpc": "RollingNativeMpcController",
    }
