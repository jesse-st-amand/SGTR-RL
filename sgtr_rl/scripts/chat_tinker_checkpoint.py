"""Interactive chat with a saved Tinker sampler checkpoint.

Examples:
    python -m scripts.chat_tinker_checkpoint \
        --run-dir results/verification/01_sft_pw_vs_qwen__20260311_142149

    python -m scripts.chat_tinker_checkpoint \
        --sampler-path tinker://.../sampler_weights/final \
        --base-model meta-llama/Llama-3.1-8B-Instruct

    python -m scripts.chat_tinker_checkpoint \
        --run-dir results/verification/01_sft_pw_vs_qwen__20260311_142149 \
        --prompt "Hello there"
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from sgtr_rl.config import load_training_config


@dataclass(frozen=True)
class ChatTarget:
    """Resolved Tinker checkpoint target."""

    sampler_path: str
    base_model: str
    run_dir: Path | None = None


@dataclass
class TranscriptWriter:
    """Append-only JSONL transcript writer."""

    paths: tuple[Path, ...]
    _handles: tuple[Any, ...]

    @property
    def path(self) -> Path:
        """Primary transcript path."""
        return self.paths[0]

    @property
    def mirror_paths(self) -> tuple[Path, ...]:
        """Additional mirrored transcript paths."""
        return self.paths[1:]

    def log_event(self, event: str, **payload: Any) -> None:
        """Append one JSONL event and flush immediately."""
        record = {
            "time": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
        line = json.dumps(record) + "\n"
        for handle in self._handles:
            handle.write(line)
            handle.flush()

    def close(self) -> None:
        """Close the underlying transcript file."""
        for handle in self._handles:
            handle.close()


def _load_tinker_env() -> None:
    """Load credentials from the repo .env if present."""
    load_dotenv(Path(".env"))


def _sanitize_path_part(value: str) -> str:
    """Convert a free-form string into a filesystem-friendly path part."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _read_latest_sampler_path(run_dir: Path) -> str:
    """Read the latest sampler checkpoint path from a run manifest."""
    manifest_path = run_dir / "checkpoints" / "checkpoints.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")

    latest_sampler_path: str | None = None
    with open(manifest_path) as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            sampler_path = payload.get("sampler_path")
            if sampler_path:
                latest_sampler_path = sampler_path

    if latest_sampler_path is None:
        raise ValueError(f"No sampler_path found in {manifest_path}")
    return latest_sampler_path


def _extract_run_timestamp(run_dir: Path) -> tuple[int, int, int, int, int, int] | None:
    """Parse ``__YYYYMMDD_HHMMSS`` from a run directory name when present."""
    parts = run_dir.name.rsplit("__", maxsplit=1)
    if len(parts) != 2:
        return None
    date_part, time_part = parts[1].split("_", maxsplit=1)
    if len(date_part) != 8 or len(time_part) != 6:
        return None
    if not (date_part.isdigit() and time_part.isdigit()):
        return None
    return (
        int(date_part[0:4]),
        int(date_part[4:6]),
        int(date_part[6:8]),
        int(time_part[0:2]),
        int(time_part[2:4]),
        int(time_part[4:6]),
    )


def _find_latest_run(results_dir: Path = Path("results")) -> Path:
    """Find the most recent results run with a usable sampler checkpoint."""
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    candidates: list[tuple[tuple[int, int, int, int, int, int] | None, float, Path]] = []
    for manifest_path in results_dir.rglob("checkpoints/checkpoints.jsonl"):
        run_dir = manifest_path.parent.parent
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        try:
            _read_latest_sampler_path(run_dir)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        timestamp = _extract_run_timestamp(run_dir)
        candidates.append((timestamp, run_dir.stat().st_mtime, run_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No results run with a sampler checkpoint found under {results_dir}"
        )

    candidates.sort(key=lambda item: (item[0] is not None, item[0], item[1]))
    return candidates[-1][2]


def _default_transcript_path(target: ChatTarget) -> Path:
    """Choose a default transcript location for a chat session."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return _default_transcript_path_for_timestamp(target, timestamp)


def _chat_slug(target: ChatTarget) -> str:
    """Stable filesystem-friendly identifier for transcript filenames."""
    if target.run_dir is not None:
        return _sanitize_path_part(target.run_dir.name)
    run_id = target.sampler_path.removeprefix("tinker://").split("/", maxsplit=1)[0]
    return _sanitize_path_part(run_id)


def _default_transcript_path_for_timestamp(target: ChatTarget, timestamp: str) -> Path:
    """Default run-local transcript path for a specific timestamp."""
    if target.run_dir is not None:
        transcript_dir = target.run_dir / "chat_logs"
        transcript_name = f"chat_{timestamp}.jsonl"
    else:
        transcript_dir = Path("chat_logs")
        transcript_name = f"chat_{_chat_slug(target)}_{timestamp}.jsonl"
    return transcript_dir / transcript_name


def _default_logs_transcript_path(
    target: ChatTarget,
    timestamp: str,
    *,
    logs_dir: Path = Path("logs"),
) -> Path:
    """Default mirrored transcript path under repo-root logs/."""
    return logs_dir / f"chat_tinker_checkpoint_{_chat_slug(target)}_{timestamp}.jsonl"


def _open_transcript_writer(
    *,
    target: ChatTarget,
    transcript_file: str | None,
    disabled: bool,
    logs_dir: Path = Path("logs"),
) -> TranscriptWriter | None:
    """Open a transcript file unless logging is disabled."""
    if disabled:
        return None
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    primary_path = (
        Path(transcript_file)
        if transcript_file
        else _default_transcript_path_for_timestamp(target, timestamp)
    )
    mirror_path = _default_logs_transcript_path(target, timestamp, logs_dir=logs_dir)

    unique_paths: list[Path] = []
    for path in (primary_path, mirror_path):
        if path not in unique_paths:
            unique_paths.append(path)

    handles = []
    for path in unique_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        handles.append(path.open("a", encoding="utf-8"))
    return TranscriptWriter(paths=tuple(unique_paths), _handles=tuple(handles))


def _resolve_target(
    *,
    run_dir: str | None,
    latest: bool,
    sampler_path: str | None,
    base_model: str | None,
    results_dir: str | None = None,
) -> ChatTarget:
    """Resolve a chat target from CLI inputs."""
    if run_dir or latest:
        run_path = _find_latest_run(Path(results_dir or "results")) if latest else Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        config_path = run_path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Run config not found: {config_path}")
        config = load_training_config(config_path)
        return ChatTarget(
            sampler_path=_read_latest_sampler_path(run_path),
            base_model=config.model_name,
            run_dir=run_path,
        )

    if sampler_path is None:
        raise ValueError("Either --run-dir or --sampler-path is required")
    if base_model is None:
        raise ValueError("--base-model is required when using --sampler-path")
    return ChatTarget(sampler_path=sampler_path, base_model=base_model)


def _build_session(target: ChatTarget) -> tuple[Any, Any]:
    """Create the Tinker sampling client and renderer."""
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    _load_tinker_env()

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=target.sampler_path)

    tokenizer = get_tokenizer(target.base_model)
    renderer_name = model_info.get_recommended_renderer_name(target.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return sampling_client, renderer


def _build_sampling_params(
    *,
    renderer: Any,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Any:
    """Create immutable Tinker sampling params for chat."""
    from tinker import types

    return types.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
        top_p=top_p,
    )


def _sample_assistant_turn(
    *,
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    messages: list[dict[str, str]],
) -> str:
    """Generate one assistant response for the current conversation."""
    from tinker_cookbook import renderers as renderer_utils

    prompt = renderer.build_generation_prompt(messages)
    result = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()
    sequence = result.sequences[0]
    parsed_msg, _ = renderer.parse_response(sequence.tokens)
    return renderer_utils.get_text_content(parsed_msg).strip()


def _run_one_prompt(
    *,
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    prompt: str,
    system_prompt: str | None,
    transcript: TranscriptWriter | None = None,
) -> str:
    """Run a single prompt against the checkpoint."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if transcript is not None:
        transcript.log_event("user", turn=1, content=prompt)
    response = _sample_assistant_turn(
        sampling_client=sampling_client,
        renderer=renderer,
        sampling_params=sampling_params,
        messages=messages,
    )
    if transcript is not None:
        transcript.log_event("assistant", turn=1, content=response)
    return response


def _read_multiline_message(input_fn: Callable[[str], str] = input) -> str | None:
    """Read a multiline user message until `/send` or `/cancel`."""
    print("paste mode: paste text, then type /send on its own line. /cancel aborts.")
    lines: list[str] = []
    while True:
        try:
            line = input_fn("... ")
        except EOFError:
            print()
            return None
        except KeyboardInterrupt:
            print("\nmultiline entry cancelled")
            return None

        if line == "/send":
            message = "\n".join(lines).strip()
            return message or None
        if line == "/cancel":
            print("multiline entry cancelled")
            return None
        lines.append(line)


def _load_prompt_text(prompt: str | None, prompt_file: str | None) -> str | None:
    """Resolve one-shot prompt text from CLI args."""
    if prompt is not None and prompt_file is not None:
        raise ValueError("Use only one of --prompt or --prompt-file")
    if prompt is not None:
        return prompt
    if prompt_file is not None:
        return Path(prompt_file).read_text(encoding="utf-8")
    return None


def _normalize_turn_messages(value: Any, *, source: str) -> list[dict[str, str]]:
    """Convert a JSONL prompt value into a list of chat messages."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]

    if isinstance(value, list):
        messages: list[dict[str, str]] = []
        for idx, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{source}: message {idx} is not an object")
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(f"{source}: message {idx} must contain string role/content")
            messages.append({"role": role, "content": content})
        if messages:
            return messages

    raise ValueError(f"{source}: expected a string or list of role/content messages")


def _load_prompt_turns_from_jsonl(
    jsonl_path: str,
    *,
    prompt_field: str,
    limit: int | None,
) -> list[list[dict[str, str]]]:
    """Load a sequence of prompt turns from JSONL."""
    path = Path(jsonl_path)
    turns: list[list[dict[str, str]]] = []

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if prompt_field not in record:
                raise ValueError(f"{path}:{line_no}: missing prompt field {prompt_field!r}")
            turns.append(
                _normalize_turn_messages(
                    record[prompt_field],
                    source=f"{path}:{line_no}:{prompt_field}",
                )
            )
            if limit is not None and len(turns) >= limit:
                break

    if not turns:
        raise ValueError(f"No prompts found in {path}")
    return turns


def _shuffle_prompt_turns(
    prompt_turns: list[list[dict[str, str]]],
    *,
    seed: int | None,
) -> list[list[dict[str, str]]]:
    """Return a shuffled copy of prompt turns."""
    shuffled = list(prompt_turns)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _interactive_loop(
    *,
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    system_prompt: str | None,
    transcript: TranscriptWriter | None = None,
) -> None:
    """Run a simple REPL for multi-turn chat."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print("Commands: /paste, /reset, /quit")
    turn = 1
    while True:
        try:
            user_text = input("\nyou> ").strip()
        except EOFError:
            if transcript is not None:
                transcript.log_event("session_end", reason="eof")
            print()
            break
        except KeyboardInterrupt:
            if transcript is not None:
                transcript.log_event("session_end", reason="keyboard_interrupt")
            print("\n")
            break

        if not user_text:
            continue
        if user_text in {"/quit", "/exit"}:
            if transcript is not None:
                transcript.log_event("session_end", reason="quit_command")
            break
        if user_text == "/paste":
            pasted_text = _read_multiline_message()
            if pasted_text is None:
                continue
            user_text = pasted_text
        if user_text == "/reset":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            turn = 1
            if transcript is not None:
                transcript.log_event("reset")
            print("conversation reset")
            continue

        messages.append({"role": "user", "content": user_text})
        if transcript is not None:
            transcript.log_event("user", turn=turn, content=user_text)
        assistant_text = _sample_assistant_turn(
            sampling_client=sampling_client,
            renderer=renderer,
            sampling_params=sampling_params,
            messages=messages,
        )
        print(f"assistant> {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})
        if transcript is not None:
            transcript.log_event("assistant", turn=turn, content=assistant_text)
        turn += 1


def _run_prompt_turns(
    *,
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    prompt_turns: list[list[dict[str, str]]],
    system_prompt: str | None,
    transcript: TranscriptWriter | None = None,
) -> None:
    """Run a fixed sequence of prompts within one shared conversation state."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    total_turns = len(prompt_turns)
    for turn_idx, turn_messages in enumerate(prompt_turns, start=1):
        print(f"\nturn {turn_idx}/{total_turns}")
        messages.extend(turn_messages)
        if transcript is not None:
            if len(turn_messages) == 1 and turn_messages[0]["role"] == "user":
                transcript.log_event("user", turn=turn_idx, content=turn_messages[0]["content"])
            else:
                transcript.log_event("user", turn=turn_idx, messages=turn_messages)
        assistant_text = _sample_assistant_turn(
            sampling_client=sampling_client,
            renderer=renderer,
            sampling_params=sampling_params,
            messages=messages,
        )
        print(f"assistant> {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})
        if transcript is not None:
            transcript.log_event("assistant", turn=turn_idx, content=assistant_text)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Chat with a saved Tinker checkpoint")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--run-dir",
        help="Results run directory containing config.yaml and checkpoints/checkpoints.jsonl",
    )
    source_group.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent run under results/ with a sampler checkpoint",
    )
    source_group.add_argument(
        "--sampler-path",
        help="Explicit Tinker sampler checkpoint path (tinker://.../sampler_weights/...)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results root to scan when using --latest (default: results)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model HF id. Required with --sampler-path; inferred from --run-dir.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional one-shot prompt. If omitted, starts an interactive chat loop.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional UTF-8 text file for one-shot prompting.",
    )
    parser.add_argument(
        "--prompt-jsonl",
        default=None,
        help="Optional JSONL file of prompts to send sequentially in one conversation.",
    )
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="Field to read from each JSONL row when using --prompt-jsonl (default: prompt).",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=None,
        help="Optional maximum number of JSONL prompts to run.",
    )
    parser.add_argument(
        "--shuffle-prompts",
        action="store_true",
        help="Shuffle JSONL prompt rows before running them sequentially.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed used with --shuffle-prompts (default: 42).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt to prepend to the conversation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum assistant tokens to generate per turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy decoding).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0).",
    )
    parser.add_argument(
        "--transcript-file",
        default=None,
        help="Optional primary JSONL transcript path. A mirror is also written under logs/.",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Disable transcript logging for this chat session.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt_sources = [
        args.prompt is not None,
        args.prompt_file is not None,
        args.prompt_jsonl is not None,
    ]
    if sum(prompt_sources) > 1:
        raise ValueError("Use only one of --prompt, --prompt-file, or --prompt-jsonl")

    prompt_text = _load_prompt_text(args.prompt, args.prompt_file)
    prompt_turns = (
        _load_prompt_turns_from_jsonl(
            args.prompt_jsonl,
            prompt_field=args.prompt_field,
            limit=args.prompt_limit,
        )
        if args.prompt_jsonl is not None
        else None
    )
    if prompt_turns is not None and args.shuffle_prompts:
        prompt_turns = _shuffle_prompt_turns(prompt_turns, seed=args.shuffle_seed)

    target = _resolve_target(
        run_dir=args.run_dir,
        latest=args.latest,
        sampler_path=args.sampler_path,
        base_model=args.base_model,
        results_dir=args.results_dir,
    )
    sampling_client, renderer = _build_session(target)
    sampling_params = _build_sampling_params(
        renderer=renderer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    transcript = _open_transcript_writer(
        target=target,
        transcript_file=args.transcript_file,
        disabled=args.no_transcript,
    )

    print(f"base_model:   {target.base_model}")
    print(f"sampler_path: {target.sampler_path}")
    if target.run_dir is not None:
        print(f"run_dir:      {target.run_dir}")
    if transcript is not None:
        print(f"transcript:   {transcript.path}")
        for mirror_path in transcript.mirror_paths:
            print(f"log_copy:     {mirror_path}")
        transcript.log_event(
            "session_start",
            base_model=target.base_model,
            sampler_path=target.sampler_path,
            run_dir=str(target.run_dir) if target.run_dir is not None else None,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            interactive=prompt_text is None and prompt_turns is None,
            prompt_jsonl=args.prompt_jsonl,
            prompt_field=args.prompt_field if args.prompt_jsonl is not None else None,
            prompt_limit=args.prompt_limit if args.prompt_jsonl is not None else None,
            shuffle_prompts=args.shuffle_prompts if args.prompt_jsonl is not None else None,
            shuffle_seed=args.shuffle_seed if args.prompt_jsonl is not None else None,
        )

    try:
        if prompt_text is not None:
            response = _run_one_prompt(
                sampling_client=sampling_client,
                renderer=renderer,
                sampling_params=sampling_params,
                prompt=prompt_text,
                system_prompt=args.system_prompt,
                transcript=transcript,
            )
            print(response)
            if transcript is not None:
                transcript.log_event("session_end", reason="one_shot")
            return

        if prompt_turns is not None:
            _run_prompt_turns(
                sampling_client=sampling_client,
                renderer=renderer,
                sampling_params=sampling_params,
                prompt_turns=prompt_turns,
                system_prompt=args.system_prompt,
                transcript=transcript,
            )
            if transcript is not None:
                transcript.log_event(
                    "session_end",
                    reason="prompt_jsonl",
                    turns=len(prompt_turns),
                    shuffled=args.shuffle_prompts,
                    shuffle_seed=args.shuffle_seed if args.shuffle_prompts else None,
                )
            return

        _interactive_loop(
            sampling_client=sampling_client,
            renderer=renderer,
            sampling_params=sampling_params,
            system_prompt=args.system_prompt,
            transcript=transcript,
        )
    finally:
        if transcript is not None:
            transcript.close()


if __name__ == "__main__":
    main()
