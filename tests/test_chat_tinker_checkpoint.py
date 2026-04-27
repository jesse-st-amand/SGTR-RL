"""Tests for scripts.chat_tinker_checkpoint."""

import json
from types import SimpleNamespace

import pytest
import yaml

from sgtr_rl.scripts.chat_tinker_checkpoint import (
    ChatTarget,
    _default_transcript_path,
    _find_latest_run,
    _load_prompt_text,
    _load_prompt_turns_from_jsonl,
    _normalize_turn_messages,
    _open_transcript_writer,
    _read_latest_sampler_path,
    _read_multiline_message,
    _resolve_target,
    _run_one_prompt,
    _run_prompt_turns,
    _shuffle_prompt_turns,
)


def test_read_latest_sampler_path_returns_last_sampler_entry(tmp_path):
    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    manifest_path = checkpoints_dir / "checkpoints.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({"name": "epoch_1", "sampler_path": "tinker://run/sampler_weights/1"}),
                json.dumps({"name": "final", "sampler_path": "tinker://run/sampler_weights/final"}),
            ]
        )
        + "\n"
    )

    assert _read_latest_sampler_path(run_dir) == "tinker://run/sampler_weights/final"


def test_resolve_target_from_run_dir_infers_model_and_sampler(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "experiment_name": "test",
                "model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "lora_rank": 16},
                "hyperparameters": {},
                "data": {},
            },
            f,
        )
    with open(run_dir / "checkpoints" / "checkpoints.jsonl", "w") as f:
        f.write(
            json.dumps(
                {
                    "name": "final",
                    "sampler_path": "tinker://abc123/sampler_weights/final",
                }
            )
            + "\n"
        )

    target = _resolve_target(
        run_dir=str(run_dir),
        latest=False,
        sampler_path=None,
        base_model=None,
    )

    assert target.base_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert target.sampler_path == "tinker://abc123/sampler_weights/final"
    assert target.run_dir == run_dir


def test_resolve_target_requires_base_model_with_sampler_path():
    with pytest.raises(ValueError, match="--base-model is required"):
        _resolve_target(
            run_dir=None,
            latest=False,
            sampler_path="tinker://abc123/sampler_weights/final",
            base_model=None,
        )


def test_find_latest_run_prefers_latest_timestamp(tmp_path):
    older = tmp_path / "results" / "smoke" / "demo__20260311_141740"
    newer = tmp_path / "results" / "verification" / "demo__20260311_142149"
    for run_dir in (older, newer):
        (run_dir / "checkpoints").mkdir(parents=True)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "experiment_name": "test",
                    "model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "lora_rank": 16},
                    "hyperparameters": {},
                    "data": {},
                },
                f,
            )
        with open(run_dir / "checkpoints" / "checkpoints.jsonl", "w") as f:
            f.write(
                json.dumps(
                    {
                        "name": "final",
                        "sampler_path": f"tinker://{run_dir.name}/sampler_weights/final",
                    }
                )
                + "\n"
            )

    assert _find_latest_run(tmp_path / "results") == newer


def test_resolve_target_from_latest_run(tmp_path):
    results_dir = tmp_path / "results"
    run_dir = results_dir / "verification" / "demo__20260311_142149"
    (run_dir / "checkpoints").mkdir(parents=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "experiment_name": "test",
                "model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "lora_rank": 16},
                "hyperparameters": {},
                "data": {},
            },
            f,
        )
    with open(run_dir / "checkpoints" / "checkpoints.jsonl", "w") as f:
        f.write(
            json.dumps(
                {
                    "name": "final",
                    "sampler_path": "tinker://abc123/sampler_weights/final",
                }
            )
            + "\n"
        )

    target = _resolve_target(
        run_dir=None,
        latest=True,
        sampler_path=None,
        base_model=None,
        results_dir=str(results_dir),
    )

    assert target.base_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert target.sampler_path == "tinker://abc123/sampler_weights/final"
    assert target.run_dir == run_dir


def test_default_transcript_path_uses_run_dir_chat_logs(tmp_path):
    target = ChatTarget(
        sampler_path="tinker://abc123:train:0/sampler_weights/final",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        run_dir=tmp_path / "run",
    )

    path = _default_transcript_path(target)

    assert path.parent == target.run_dir / "chat_logs"
    assert path.name.startswith("chat_")
    assert path.suffix == ".jsonl"


def test_open_transcript_writer_writes_jsonl_events(tmp_path):
    target = ChatTarget(
        sampler_path="tinker://abc123:train:0/sampler_weights/final",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        run_dir=tmp_path / "run",
    )
    logs_dir = tmp_path / "logs"

    writer = _open_transcript_writer(
        target=target,
        transcript_file=str(tmp_path / "transcripts" / "chat.jsonl"),
        disabled=False,
        logs_dir=logs_dir,
    )
    assert writer is not None
    writer.log_event("session_start", interactive=True)
    writer.log_event("user", turn=1, content="hello")
    writer.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "transcripts" / "chat.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == ["session_start", "user"]
    assert records[1]["content"] == "hello"
    assert len(writer.paths) == 2
    mirror_records = [
        json.loads(line)
        for line in writer.mirror_paths[0].read_text().splitlines()
    ]
    assert mirror_records == records
    assert writer.mirror_paths[0].parent == logs_dir


def test_read_multiline_message_collects_until_send():
    lines = iter(["first line", "second line", "/send"])

    message = _read_multiline_message(input_fn=lambda _prompt: next(lines))

    assert message == "first line\nsecond line"


def test_read_multiline_message_cancel_returns_none():
    lines = iter(["first line", "/cancel"])

    message = _read_multiline_message(input_fn=lambda _prompt: next(lines))

    assert message is None


def test_load_prompt_text_prefers_prompt_string():
    assert _load_prompt_text("hello", None) == "hello"


def test_load_prompt_text_reads_file(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("hello from file", encoding="utf-8")

    assert _load_prompt_text(None, str(prompt_path)) == "hello from file"


def test_load_prompt_text_rejects_both_prompt_sources(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("hello from file", encoding="utf-8")

    with pytest.raises(ValueError, match="Use only one of --prompt or --prompt-file"):
        _load_prompt_text("hello", str(prompt_path))


def test_normalize_turn_messages_accepts_string_and_message_lists():
    assert _normalize_turn_messages("hello", source="row1") == [
        {"role": "user", "content": "hello"}
    ]
    assert _normalize_turn_messages(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        source="row2",
    ) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_load_prompt_turns_from_jsonl_reads_rows_and_limit(tmp_path):
    path = tmp_path / "prompts.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "first"}) + "\n")
        f.write(json.dumps({"prompt": [{"role": "user", "content": "second"}]}) + "\n")

    turns = _load_prompt_turns_from_jsonl(str(path), prompt_field="prompt", limit=1)
    assert turns == [[{"role": "user", "content": "first"}]]

    turns = _load_prompt_turns_from_jsonl(str(path), prompt_field="prompt", limit=None)
    assert turns == [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]


def test_load_prompt_turns_from_jsonl_requires_prompt_field(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing prompt field"):
        _load_prompt_turns_from_jsonl(str(path), prompt_field="prompt", limit=None)


def test_shuffle_prompt_turns_is_deterministic():
    turns = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
        [{"role": "user", "content": "third"}],
    ]

    shuffled_1 = _shuffle_prompt_turns(turns, seed=42)
    shuffled_2 = _shuffle_prompt_turns(turns, seed=42)

    assert shuffled_1 == shuffled_2
    assert sorted(turn[0]["content"] for turn in shuffled_1) == ["first", "second", "third"]


def test_run_prompt_turns_keeps_shared_conversation_context(capsys):
    seen_prompts = []

    class FakeRenderer:
        def build_generation_prompt(self, messages):
            seen_prompts.append([dict(message) for message in messages])
            return "rendered-prompt"

        def parse_response(self, tokens):
            return {"content": tokens[0]}, None

    class FakeFuture:
        def __init__(self, token):
            self._token = token

        def result(self):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[self._token])])

    class FakeSamplingClient:
        def __init__(self):
            self.calls = 0

        def sample(self, *, prompt, num_samples, sampling_params):
            assert prompt == "rendered-prompt"
            assert num_samples == 1
            assert sampling_params == "params"
            self.calls += 1
            return FakeFuture(f"assistant-{self.calls}")

    _run_prompt_turns(
        sampling_client=FakeSamplingClient(),
        renderer=FakeRenderer(),
        sampling_params="params",
        prompt_turns=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        system_prompt="system",
        transcript=None,
    )

    captured = capsys.readouterr()
    assert "turn 1/2" in captured.out
    assert "turn 2/2" in captured.out
    assert seen_prompts == [
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
        ],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "assistant-1"},
            {"role": "user", "content": "second"},
        ],
    ]


def test_run_one_prompt_uses_renderer_and_sampling_client():
    rendered_prompts = []

    class FakeRenderer:
        def build_generation_prompt(self, messages):
            rendered_prompts.append(messages)
            return "rendered-prompt"

        def parse_response(self, tokens):
            assert tokens == ["hello"]
            return {"content": "assistant text"}, None

    class FakeFuture:
        def result(self):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=["hello"])])

    class FakeSamplingClient:
        def sample(self, *, prompt, num_samples, sampling_params):
            assert prompt == "rendered-prompt"
            assert num_samples == 1
            assert sampling_params == "params"
            return FakeFuture()

    response = _run_one_prompt(
        sampling_client=FakeSamplingClient(),
        renderer=FakeRenderer(),
        sampling_params="params",
        prompt="hello there",
        system_prompt="be helpful",
        transcript=None,
    )

    assert rendered_prompts == [
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello there"},
        ]
    ]
    assert response == "assistant text"
