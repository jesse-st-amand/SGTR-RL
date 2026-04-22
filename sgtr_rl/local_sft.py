"""Local single-node SFT training with Transformers + PEFT."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sgtr_rl.artifacts import JsonlMetricsLogger, atomic_write_json, update_run_status
from sgtr_rl.benchmarks import should_run_training_eval
from sgtr_rl.config import TrainingConfig
from sgtr_rl.data import build_conversation
from sgtr_rl.local_eval import run_benchmark_evals, run_train_panel_eval, run_val_eval
from sgtr_rl.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


@dataclass
class LocalTrainingContext:
    """Shared state for local HF training and evaluation."""

    config: TrainingConfig
    model: Any
    tokenizer: Any
    optimizer: Any
    device: torch.device
    dtype: torch.dtype
    runtime: RuntimeConfig
    metrics_logger: JsonlMetricsLogger


def _resolve_device(runtime: RuntimeConfig) -> torch.device:
    device = runtime.local.device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(runtime: RuntimeConfig, device: torch.device) -> torch.dtype:
    dtype = runtime.local.dtype
    if dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _render_messages(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    parts = []
    for msg in messages:
        parts.append(f"{msg['role'].upper()}: {msg['content']}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _make_training_example(
    item: dict,
    *,
    tokenizer,
    use_system_prompt: bool,
    max_seq_length: int,
) -> dict[str, list[int]]:
    messages = build_conversation(item, use_system_prompt)
    prompt_text = _render_messages(tokenizer, messages, add_generation_prompt=True)
    full_text = _render_messages(
        tokenizer,
        messages + [{"role": "assistant", "content": item["target"]}],
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if not full_ids[: len(prompt_ids)] == prompt_ids:
        raise ValueError("Prompt tokenization is not a prefix of the training example")

    input_ids = full_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if len(input_ids) > max_seq_length:
        overflow = len(input_ids) - max_seq_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def _collate_training_examples(
    batch: list[dict[str, list[int]]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def _loss_sum_and_token_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_count = shift_labels.ne(-100).sum()
    loss_sum = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, token_count


def _first_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    correct = 0
    total = 0
    for row_logits, row_labels in zip(shift_logits, shift_labels):
        positions = row_labels.ne(-100).nonzero(as_tuple=False)
        if len(positions) == 0:
            continue
        first_idx = int(positions[0].item())
        pred = int(row_logits[first_idx].argmax(dim=-1).item())
        target = int(row_labels[first_idx].item())
        correct += int(pred == target)
        total += 1
    return correct / total if total else 0.0


def setup_local_training(config: TrainingConfig, runtime: RuntimeConfig) -> LocalTrainingContext:
    """Load model/tokenizer, apply LoRA, and create optimizer/loggers."""
    _seed_everything(config.seed)
    device = _resolve_device(runtime)
    dtype = _resolve_dtype(runtime, device)

    logger.info(
        "Loading local model %s on %s (dtype=%s, 4bit=%s)",
        config.model_name,
        device,
        dtype,
        runtime.local.load_in_4bit,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=runtime.local.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "cache_dir": runtime.local.cache_dir,
        "torch_dtype": dtype,
    }
    if runtime.local.attention_implementation:
        model_kwargs["attn_implementation"] = runtime.local.attention_implementation
    if runtime.local.load_in_4bit:
        if device.type != "cuda":
            raise ValueError("4-bit loading requires CUDA")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if not runtime.local.load_in_4bit:
        model.to(device)
    else:
        model = prepare_model_for_kbit_training(model)

    if runtime.local.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_alpha = runtime.local.lora_alpha or (config.lora_rank * 2)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=runtime.local.lora_dropout,
        bias="none",
        target_modules=runtime.local.target_modules,
    )
    model = get_peft_model(model, peft_config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate)

    metrics_logger = JsonlMetricsLogger(
        run_dir=config.run_dir,
        experiment_name=config.experiment_name,
        config_payload={
            "training": config.model_dump(mode="json"),
            "runtime": runtime.model_dump(mode="json"),
        },
        wandb_project=config.wandb_project,
    )

    logger.info(
        "Local training ready: %s trainable tensors, output=%s",
        len(trainable_params),
        config.run_dir,
    )

    return LocalTrainingContext(
        config=config,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        dtype=dtype,
        runtime=runtime,
        metrics_logger=metrics_logger,
    )


def save_local_checkpoint(
    ctx: LocalTrainingContext,
    config: TrainingConfig,
    *,
    global_step: int,
    epoch: int,
) -> None:
    """Save the final adapter checkpoint and a small manifest."""
    checkpoint_dir = Path(config.run_dir) / "checkpoints" / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ctx.model.save_pretrained(checkpoint_dir, safe_serialization=True)
    ctx.tokenizer.save_pretrained(checkpoint_dir)
    atomic_write_json(
        checkpoint_dir / "checkpoint_manifest.json",
        {
            "backend": "local",
            "algorithm": config.algorithm,
            "base_model": config.model_name,
            "lora_rank": config.lora_rank,
            "epoch": epoch,
            "global_step": global_step,
            "runtime": ctx.runtime.model_dump(mode="json"),
        },
    )
    logger.info("Saved final adapter checkpoint to %s", checkpoint_dir)


def load_local_checkpoint_for_eval(
    config: TrainingConfig,
    runtime: RuntimeConfig,
    *,
    checkpoint_dir: str | Path,
    eval_run_dir: str | Path,
    wandb_project: str | None = None,
) -> LocalTrainingContext:
    """Load a saved local adapter checkpoint for posthoc evaluation."""
    checkpoint_path = Path(checkpoint_dir)
    device = _resolve_device(runtime)
    dtype = _resolve_dtype(runtime, device)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        cache_dir=runtime.local.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "cache_dir": runtime.local.cache_dir,
        "torch_dtype": dtype,
    }
    if runtime.local.attention_implementation:
        model_kwargs["attn_implementation"] = runtime.local.attention_implementation
    if runtime.local.load_in_4bit:
        if device.type != "cuda":
            raise ValueError("4-bit loading requires CUDA")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if not runtime.local.load_in_4bit:
        base_model.to(device)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    metrics_logger = JsonlMetricsLogger(
        run_dir=eval_run_dir,
        experiment_name=f"{config.experiment_name}_checkpoint_eval",
        config_payload={
            "training": config.model_dump(mode="json"),
            "runtime": runtime.model_dump(mode="json"),
            "checkpoint_dir": str(checkpoint_path),
        },
        wandb_project=wandb_project,
    )

    return LocalTrainingContext(
        config=config,
        model=model,
        tokenizer=tokenizer,
        optimizer=None,
        device=device,
        dtype=dtype,
        runtime=runtime,
        metrics_logger=metrics_logger,
    )


def load_local_base_model_for_eval(
    config: TrainingConfig,
    runtime: RuntimeConfig,
    *,
    eval_run_dir: str | Path,
    wandb_project: str | None = None,
) -> LocalTrainingContext:
    """Load the base local model without any adapter for posthoc evaluation."""
    device = _resolve_device(runtime)
    dtype = _resolve_dtype(runtime, device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=runtime.local.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "cache_dir": runtime.local.cache_dir,
        "torch_dtype": dtype,
    }
    if runtime.local.attention_implementation:
        model_kwargs["attn_implementation"] = runtime.local.attention_implementation
    if runtime.local.load_in_4bit:
        if device.type != "cuda":
            raise ValueError("4-bit loading requires CUDA")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if not runtime.local.load_in_4bit:
        model.to(device)
    model.eval()

    metrics_logger = JsonlMetricsLogger(
        run_dir=eval_run_dir,
        experiment_name=f"{config.experiment_name}_base_model_eval",
        config_payload={
            "training": config.model_dump(mode="json"),
            "runtime": runtime.model_dump(mode="json"),
            "base_model_only": True,
        },
        wandb_project=wandb_project,
    )

    return LocalTrainingContext(
        config=config,
        model=model,
        tokenizer=tokenizer,
        optimizer=None,
        device=device,
        dtype=dtype,
        runtime=runtime,
        metrics_logger=metrics_logger,
    )


def train_local_sft(
    config: TrainingConfig,
    runtime: RuntimeConfig,
    prompts: list[dict],
    val_prompts: list[dict],
) -> int:
    """Run local SFT with periodic evaluation and final adapter save."""
    ctx = setup_local_training(config, runtime)
    global_step = 0
    current_epoch = 0
    try:
        logger.info("Running epoch 0 baseline evaluation (untrained local model)...")
        run_val_eval(
            val_prompts,
            ctx,
            step=0,
            epoch=0,
            run_dir=config.run_dir,
            use_system_prompt=config.use_system_prompt,
            eval_trigger=config.eval_trigger,
            diagnostic_num_examples=config.eval_diagnostic_num_examples,
            diagnostic_example_ids=config.eval_diagnostic_example_ids,
        )
        run_train_panel_eval(
            prompts,
            ctx,
            step=0,
            epoch=0,
            run_dir=config.run_dir,
            use_system_prompt=config.use_system_prompt,
            eval_trigger=config.eval_trigger,
            diagnostic_num_examples=config.train_diagnostic_num_examples,
            diagnostic_example_ids=config.train_diagnostic_example_ids,
        )
        run_benchmark_evals(
            config.benchmark_evals,
            ctx,
            step=0,
            epoch=0,
            total_epochs=config.num_epochs,
            run_dir=config.run_dir,
            use_system_prompt=config.use_system_prompt,
            eval_trigger=config.eval_trigger,
        )
        update_run_status(
            config.run_dir,
            "running",
            backend="local",
            algorithm=config.algorithm,
            step=0,
            epoch=0,
        )

        batch_size = config.batch_size
        n_batches = len(prompts) // batch_size
        max_steps = config.max_steps
        if n_batches == 0:
            raise ValueError(
                f"Not enough training records ({len(prompts)}) for batch_size={batch_size}. "
                "Reduce batch_size or increase train data."
            )
        total_steps = n_batches * config.num_epochs if max_steps is None else min(
            n_batches * config.num_epochs,
            max_steps,
        )
        logger.info(
            "Local SFT: %s epochs, %s batches/epoch, batch_size=%s, total_steps=%s",
            config.num_epochs,
            n_batches,
            batch_size,
            total_steps,
        )

        completed_epochs = 0
        stopped_early = False
        train_eval_prompts = list(prompts)
        for epoch in range(config.num_epochs):
            current_epoch = epoch + 1
            random.shuffle(prompts)
            ctx.model.train()
            tokenizer = ctx.tokenizer
            tokenizer.padding_side = "right"

            for batch_idx in range(n_batches):
                t_start = time.time()
                batch = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                features = [
                    _make_training_example(
                        item,
                        tokenizer=tokenizer,
                        use_system_prompt=config.use_system_prompt,
                        max_seq_length=runtime.local.max_seq_length,
                    )
                    for item in batch
                ]
                tensors = _collate_training_examples(
                    features,
                    pad_token_id=tokenizer.pad_token_id,
                    device=ctx.device,
                )

                outputs = ctx.model(**tensors)
                loss = outputs.loss
                loss.backward()
                ctx.optimizer.step()
                ctx.optimizer.zero_grad(set_to_none=True)

                loss_sum, token_count = _loss_sum_and_token_count(outputs.logits, tensors["labels"])
                train_nll = float((loss_sum / token_count.clamp_min(1)).item())
                train_acc = _first_token_accuracy(outputs.logits.detach(), tensors["labels"])
                elapsed = time.time() - t_start

                global_step += 1
                logger.info(
                    "[epoch %s/%s] batch %s/%s | nll=%.4f | acc=%.1f%% | %.1fs",
                    current_epoch,
                    config.num_epochs,
                    batch_idx + 1,
                    n_batches,
                    train_nll,
                    train_acc * 100.0,
                    elapsed,
                )
                ctx.metrics_logger.log_metrics(
                    {
                        "train/nll": train_nll,
                        "train/accuracy": train_acc,
                        "train/batch_time_s": elapsed,
                    },
                    step=global_step,
                )

                if config.eval_trigger == "step" and should_run_training_eval(
                    trigger=config.eval_trigger,
                    frequency=config.eval_frequency,
                    step=global_step,
                    epoch=current_epoch,
                    total_steps=total_steps,
                    total_epochs=config.num_epochs,
                ):
                    run_val_eval(
                        val_prompts,
                        ctx,
                        step=global_step,
                        epoch=current_epoch,
                        run_dir=config.run_dir,
                        use_system_prompt=config.use_system_prompt,
                        eval_trigger=config.eval_trigger,
                        diagnostic_num_examples=config.eval_diagnostic_num_examples,
                        diagnostic_example_ids=config.eval_diagnostic_example_ids,
                    )
                    run_train_panel_eval(
                        train_eval_prompts,
                        ctx,
                        step=global_step,
                        epoch=current_epoch,
                        run_dir=config.run_dir,
                        use_system_prompt=config.use_system_prompt,
                        eval_trigger=config.eval_trigger,
                        diagnostic_num_examples=config.train_diagnostic_num_examples,
                        diagnostic_example_ids=config.train_diagnostic_example_ids,
                    )
                    run_benchmark_evals(
                        config.benchmark_evals,
                        ctx,
                        step=global_step,
                        epoch=current_epoch,
                        total_epochs=config.num_epochs,
                        schedule_index=global_step,
                        schedule_total=total_steps,
                        run_dir=config.run_dir,
                        use_system_prompt=config.use_system_prompt,
                        eval_trigger=config.eval_trigger,
                    )

                if max_steps is not None and global_step >= max_steps:
                    stopped_early = True
                    logger.info(
                        "Reached max_steps=%s at epoch %s batch %s/%s",
                        max_steps,
                        current_epoch,
                        batch_idx + 1,
                        n_batches,
                    )
                    break

            completed_epochs = current_epoch
            if stopped_early and batch_idx + 1 < n_batches:
                logger.info(
                    "Stopped during epoch %s after %s batches",
                    current_epoch,
                    batch_idx + 1,
                )
            else:
                logger.info("Epoch %s complete", current_epoch)
            if config.eval_trigger == "epoch" and should_run_training_eval(
                trigger=config.eval_trigger,
                frequency=config.eval_frequency,
                step=global_step,
                epoch=current_epoch,
                total_steps=total_steps,
                total_epochs=config.num_epochs,
            ):
                run_val_eval(
                    val_prompts,
                    ctx,
                    step=global_step,
                    epoch=current_epoch,
                    run_dir=config.run_dir,
                    use_system_prompt=config.use_system_prompt,
                    eval_trigger=config.eval_trigger,
                    diagnostic_num_examples=config.eval_diagnostic_num_examples,
                    diagnostic_example_ids=config.eval_diagnostic_example_ids,
                )
                run_train_panel_eval(
                    train_eval_prompts,
                    ctx,
                    step=global_step,
                    epoch=current_epoch,
                    run_dir=config.run_dir,
                    use_system_prompt=config.use_system_prompt,
                    eval_trigger=config.eval_trigger,
                    diagnostic_num_examples=config.train_diagnostic_num_examples,
                    diagnostic_example_ids=config.train_diagnostic_example_ids,
                )
                run_benchmark_evals(
                    config.benchmark_evals,
                    ctx,
                    step=global_step,
                    epoch=current_epoch,
                    total_epochs=completed_epochs if stopped_early else config.num_epochs,
                    run_dir=config.run_dir,
                    use_system_prompt=config.use_system_prompt,
                    eval_trigger=config.eval_trigger,
                )
            update_run_status(
                config.run_dir,
                "running",
                backend="local",
                algorithm=config.algorithm,
                step=global_step,
                epoch=current_epoch,
            )

            if stopped_early:
                break

        config.completed_epochs = completed_epochs
        save_local_checkpoint(ctx, config, global_step=global_step, epoch=completed_epochs)
        return global_step
    finally:
        ctx.metrics_logger.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
