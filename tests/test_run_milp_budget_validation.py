from experiments import run_milp_budget_validation as validation


def test_locked_milp_validation_commands_use_expected_budgets():
    args = validation.parse_args([])

    rolling, full = validation.build_tasks(args)

    assert args.seeds == validation.DEFAULT_SEEDS
    assert args.run_label == "context_fixed_v2"
    assert len(rolling) == 6
    assert len(full) == 3
    assert all("--solver-threads" in task.command for task in rolling + full)
    assert {task.command[task.command.index("--rolling-time-limit-seconds") + 1]
            for task in rolling} == {"30", "300"}
    assert all(
        task.command[task.command.index("--full-milp-time-limit-seconds") + 1]
        == "7200"
        for task in full
    )
    assert all(
        task.command[task.command.index("--full-milp-horizon-hours") + 1]
        == "720"
        for task in full
    )
    assert all(args.run_label in str(task.output_dir) for task in rolling + full)
