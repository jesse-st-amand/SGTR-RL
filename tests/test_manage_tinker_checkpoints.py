"""Tests for scripts.manage_tinker_checkpoints."""

from datetime import UTC, datetime

from scripts.manage_tinker_checkpoints import (
    CheckpointRecord,
    _format_size,
    _group_runs,
    _parse_cutoff,
    _print_breakdown,
    _select_runs_to_delete,
    _split_tinker_path,
)


def _record(
    run_id: str,
    checkpoint_id: str,
    checkpoint_type: str,
    time: datetime,
    size_bytes: int = 100,
) -> CheckpointRecord:
    return CheckpointRecord(
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        checkpoint_type=checkpoint_type,
        tinker_path=f"tinker://{run_id}/{checkpoint_id}",
        size_bytes=size_bytes,
        time=time,
        expires_at=None,
    )


class TestSplitTinkerPath:
    def test_split_tinker_path(self):
        run_id, checkpoint_id = _split_tinker_path(
            "tinker://abc123:train:0/sampler_weights/final"
        )
        assert run_id == "abc123:train:0"
        assert checkpoint_id == "sampler_weights/final"


class TestParseCutoff:
    def test_parse_cutoff(self):
        result = _parse_cutoff("2026-03-01")
        assert result == datetime(2026, 3, 1, tzinfo=UTC)


class TestGroupRuns:
    def test_group_runs_aggregates_by_run(self):
        records = [
            _record(
                "run-b",
                "weights/final",
                "training",
                datetime(2026, 3, 2, tzinfo=UTC),
                size_bytes=300,
            ),
            _record(
                "run-a",
                "weights/final",
                "training",
                datetime(2026, 3, 1, tzinfo=UTC),
                size_bytes=100,
            ),
            _record(
                "run-a",
                "sampler_weights/final",
                "sampler",
                datetime(2026, 3, 1, 0, 1, tzinfo=UTC),
                size_bytes=200,
            ),
        ]

        grouped = _group_runs(records)

        assert [run.run_id for run in grouped] == ["run-b", "run-a"]
        assert grouped[1].total_size_bytes == 300
        assert grouped[1].first_time == datetime(2026, 3, 1, tzinfo=UTC)
        assert grouped[1].last_time == datetime(2026, 3, 1, 0, 1, tzinfo=UTC)


class TestSelectRunsToDelete:
    def test_select_by_explicit_run_id(self):
        runs = _group_runs(
            [
                _record("run-a", "weights/final", "training", datetime(2026, 3, 1, tzinfo=UTC)),
                _record("run-b", "weights/final", "training", datetime(2026, 3, 2, tzinfo=UTC)),
            ]
        )

        selected = _select_runs_to_delete(
            runs,
            run_ids={"run-a"},
            before=None,
            keep_recent_runs=0,
        )

        assert [run.run_id for run in selected] == ["run-a"]

    def test_select_by_age(self):
        runs = _group_runs(
            [
                _record("run-a", "weights/final", "training", datetime(2026, 2, 20, tzinfo=UTC)),
                _record("run-b", "weights/final", "training", datetime(2026, 3, 5, tzinfo=UTC)),
            ]
        )

        selected = _select_runs_to_delete(
            runs,
            run_ids=set(),
            before=datetime(2026, 3, 1, tzinfo=UTC),
            keep_recent_runs=0,
        )

        assert [run.run_id for run in selected] == ["run-a"]

    def test_keep_recent_runs_overrides_other_filters(self):
        runs = _group_runs(
            [
                _record("run-old", "weights/final", "training", datetime(2026, 2, 1, tzinfo=UTC)),
                _record("run-new", "weights/final", "training", datetime(2026, 2, 2, tzinfo=UTC)),
            ]
        )

        selected = _select_runs_to_delete(
            runs,
            run_ids={"run-new", "run-old"},
            before=datetime(2026, 3, 1, tzinfo=UTC),
            keep_recent_runs=1,
        )

        assert [run.run_id for run in selected] == ["run-old"]


class TestFormatSize:
    def test_format_size(self):
        assert _format_size(1024) == "1.0 KiB"


class TestPrintBreakdown:
    def test_print_breakdown(self, capsys):
        runs = _group_runs(
            [
                _record(
                    "run-a",
                    "weights/final",
                    "training",
                    datetime(2026, 3, 1, tzinfo=UTC),
                    size_bytes=1024,
                ),
                _record(
                    "run-a",
                    "sampler_weights/final",
                    "sampler",
                    datetime(2026, 3, 1, 0, 1, tzinfo=UTC),
                    size_bytes=2048,
                ),
                _record(
                    "run-b",
                    "weights/final",
                    "training",
                    datetime(2026, 3, 2, tzinfo=UTC),
                    size_bytes=512,
                ),
            ]
        )

        _print_breakdown(runs, local_labels={"run-a": "local-a"})

        output = capsys.readouterr().out
        assert "Breakdown:" in output
        assert "Runs: 2" in output
        assert "Checkpoints: 3" in output
        assert "training: 2" in output
        assert "sampler: 1" in output
        assert "With local run label: 1" in output
