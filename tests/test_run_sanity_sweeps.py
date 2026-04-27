"""Tests for scripts.run_sanity_sweeps."""

from sgtr_rl.scripts.run_sanity_sweeps import SweepRun, build_run_plan, build_train_command


def test_build_run_plan_defaults_to_size_sweep():
    plan = build_run_plan([])

    assert [run.name for run in plan] == ["ids_1", "ids_10", "ids_40", "ids_80"]
    assert plan[0].args == ["--max_train_ids", "1", "--batch_size", "2", "--subset_seed", "42"]


def test_build_run_plan_all_includes_all_suites():
    plan = build_run_plan(["all"])

    assert [run.name for run in plan] == [
        "ids_1",
        "ids_10",
        "ids_40",
        "ids_80",
        "ids_80_rand_labels",
        "ids_10_seed_41",
        "ids_10_seed_42",
        "ids_10_seed_43",
    ]


def test_build_train_command_passes_common_args():
    command = build_train_command(
        config="experiments/04_sanity_sft_pw_vs_qwen/config.yaml",
        runtime="runtimes/local_gpu.yaml",
        group="sanity",
        exists="skip",
        num_epochs=400,
        max_steps=400,
        run=SweepRun(
            name="ids_10",
            args=["--max_train_ids", "10", "--batch_size", "4"],
            description="10 IDs",
        ),
    )

    assert command == [
        command[0],
        "-m",
        "scripts.train",
        "--config",
        "experiments/04_sanity_sft_pw_vs_qwen/config.yaml",
        "--exists",
        "skip",
        "--runtime",
        "runtimes/local_gpu.yaml",
        "--group",
        "sanity",
        "--num_epochs",
        "400",
        "--max_steps",
        "400",
        "--max_train_ids",
        "10",
        "--batch_size",
        "4",
    ]
