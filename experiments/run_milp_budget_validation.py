"""Run the locked MILP compute-budget validation on controller-validation seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "smoke_test_paper_controllers.py"
RESULTS_ROOT = ROOT / "experiments_results"
DEFAULT_SEEDS = (8_100_001, 8_100_002, 8_100_003)


@dataclass(frozen=True)
class Task:
    task_id: str
    controller: str
    seed: int
    output_dir: Path
    log_path: Path
    command: tuple[str, ...]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--rolling-time-limits-seconds",
        type=int,
        nargs="+",
        default=(30, 300),
    )
    parser.add_argument(
        "--full-milp-time-limit-seconds",
        type=int,
        default=7_200,
    )
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--rolling-parallel-jobs", type=int, default=2)
    parser.add_argument("--full-milp-parallel-jobs", type=int, default=1)
    parser.add_argument("--run-label", default="context_fixed_v2")
    parser.add_argument(
        "--state-path",
        type=Path,
        default=RESULTS_ROOT / "milp_budget_validation_state.json",
    )
    args = parser.parse_args(argv)
    positive_values = (
        *args.rolling_time_limits_seconds,
        args.full_milp_time_limit_seconds,
        args.solver_threads,
        args.rolling_parallel_jobs,
        args.full_milp_parallel_jobs,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("time limits, threads and parallel-job counts must be positive")
    if Path(args.run_label).name != args.run_label:
        parser.error("run label must be a single path-safe name")
    return args


def _common_command(args, task: Task) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--out-dir",
        str(task.output_dir),
        "--controllers",
        task.controller,
        "--seed",
        str(task.seed),
        "--online-episode-hours",
        "720",
        "--forecast-context-hours",
        "168",
        "--solver-threads",
        str(args.solver_threads),
    ]


def build_tasks(args) -> tuple[list[Task], list[Task]]:
    rolling_tasks: list[Task] = []
    full_tasks: list[Task] = []
    for seed in args.seeds:
        for limit_s in args.rolling_time_limits_seconds:
            output_dir = (
                RESULTS_ROOT
                / "E1"
                / f"rolling_milp_budget_validation_{args.run_label}"
                / f"{limit_s}s"
                / f"seed_{seed}"
            )
            task = Task(
                task_id=f"rolling_{limit_s}s_seed_{seed}",
                controller="rolling_milp",
                seed=seed,
                output_dir=output_dir,
                log_path=output_dir.parent / f"seed_{seed}.log",
                command=(),
            )
            command = [
                *_common_command(args, task),
                "--rolling-replan-hours",
                "24",
                "--rolling-planning-horizon-hours",
                "168",
                "--rolling-time-limit-seconds",
                str(limit_s),
                "--purpose",
                "rolling_milp_time_budget_validation_not_formal_results",
            ]
            rolling_tasks.append(
                Task(**{**asdict(task), "command": tuple(command)})
            )

        output_dir = (
            RESULTS_ROOT
            / "E5"
            / f"full_milp_2h_validation_{args.run_label}"
            / f"seed_{seed}"
        )
        task = Task(
            task_id=f"full_2h_seed_{seed}",
            controller="full_milp",
            seed=seed,
            output_dir=output_dir,
            log_path=output_dir.parent / f"seed_{seed}.log",
            command=(),
        )
        command = [
            *_common_command(args, task),
            "--full-milp-horizon-hours",
            "720",
            "--full-milp-time-limit-seconds",
            str(args.full_milp_time_limit_seconds),
            "--purpose",
            "full_milp_2h_validation_not_formal_results",
        ]
        full_tasks.append(
            Task(**{**asdict(task), "command": tuple(command)})
        )
    return rolling_tasks, full_tasks


def _write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args) -> int:
    cplex = shutil.which("cplex")
    if not cplex:
        raise RuntimeError("CPLEX executable is not available on PATH")
    rolling_tasks, full_tasks = build_tasks(args)
    tasks = [*rolling_tasks, *full_tasks]
    collisions = [
        str(path)
        for task in tasks
        for path in (task.output_dir, task.log_path)
        if path.exists()
    ]
    if args.state_path.exists():
        collisions.append(str(args.state_path))
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing experiment artifacts: "
            + ", ".join(collisions)
        )

    state_lock = threading.Lock()
    state = {
        "purpose": "validation_only_not_formal_test",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "cplex_executable": cplex,
        "solver_threads_per_process": args.solver_threads,
        "run_label": args.run_label,
        "rolling_parallel_jobs": args.rolling_parallel_jobs,
        "full_milp_parallel_jobs": args.full_milp_parallel_jobs,
        "tasks": {
            task.task_id: {
                "controller": task.controller,
                "seed": task.seed,
                "output_dir": str(task.output_dir),
                "log_path": str(task.log_path),
                "command": list(task.command),
                "status": "pending",
                "return_code": None,
                "wall_clock_seconds": None,
            }
            for task in tasks
        },
    }
    _write_state(args.state_path, state)

    def execute(task: Task) -> tuple[str, int, float]:
        task.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with state_lock:
            state["tasks"][task.task_id]["status"] = "running"
            state["tasks"][task.task_id]["started_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            _write_state(args.state_path, state)
        started = time.perf_counter()
        with task.log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                task.command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        elapsed = time.perf_counter() - started
        return task.task_id, completed.returncode, elapsed

    futures = {}
    with (
        ThreadPoolExecutor(
            max_workers=args.rolling_parallel_jobs
        ) as rolling_pool,
        ThreadPoolExecutor(
            max_workers=args.full_milp_parallel_jobs
        ) as full_pool,
    ):
        for task in rolling_tasks:
            futures[rolling_pool.submit(execute, task)] = task
        for task in full_tasks:
            futures[full_pool.submit(execute, task)] = task
        for future in as_completed(futures):
            task_id, return_code, elapsed = future.result()
            with state_lock:
                entry = state["tasks"][task_id]
                entry["status"] = (
                    "completed" if return_code == 0 else "failed"
                )
                entry["return_code"] = return_code
                entry["wall_clock_seconds"] = elapsed
                entry["finished_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                _write_state(args.state_path, state)

    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_state(args.state_path, state)
    return int(
        any(entry["status"] != "completed" for entry in state["tasks"].values())
    )


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
