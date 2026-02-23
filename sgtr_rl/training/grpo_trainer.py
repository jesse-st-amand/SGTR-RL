"""GRPO trainers for SGTR-RL."""

import json
import logging
import time
from pathlib import Path

from sgtr_rl.training.train_config import TrainingConfig
from sgtr_rl.training.reward import sgtr_binary_reward

logger = logging.getLogger(__name__)


class LocalGRPOTrainer:
    """GRPO trainer using TRL on a local GPU."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None

    def _load_model_and_tokenizer(self):
        """Load the base HF model and apply LoRA via peft."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _load_prompt_dataset(self):
        """Load JSONL prompt dataset into an HF Dataset."""
        from datasets import Dataset

        records = []
        with open(self.config.train_file, "r") as f:
            for line in f:
                record = json.loads(line)
                records.append({
                    "prompt": record["prompt"],
                    "target": record["target"],
                })

        self.train_dataset = Dataset.from_list(records)
        print(f"Loaded {len(self.train_dataset)} training prompts")

    def _build_reward_fn(self):
        """Return a reward function compatible with TRL's GRPOTrainer.

        TRL calls the reward function with (completions, prompts, ...) where
        completions is a list of strings.  We look up the target for each
        prompt from the dataset and delegate to ``sgtr_binary_reward``.
        """
        # Build a lookup from prompt text to target
        prompt_to_target: dict[str, str] = {}
        with open(self.config.train_file, "r") as f:
            for line in f:
                record = json.loads(line)
                prompt_to_target[record["prompt"]] = record["target"]

        def reward_fn(completions, **kwargs):
            prompts = kwargs.get("prompts", [])
            targets = [prompt_to_target.get(p, "") for p in prompts]
            return sgtr_binary_reward(completions, targets)

        return reward_fn

    def train(self, resume_from_checkpoint: str | None = None):
        """Run GRPO training loop."""
        from trl import GRPOTrainer, GRPOConfig

        self._load_model_and_tokenizer()
        self._load_prompt_dataset()
        reward_fn = self._build_reward_fn()

        output_dir = (
            self.config.output_dir or f"data/checkpoints/{self.config.experiment_name}"
        )

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            num_generations=self.config.num_rollouts_per_prompt,
            max_completion_length=self.config.max_completion_length,
            save_steps=self.config.save_steps,
            bf16=self.config.bf16,
            seed=self.config.seed,
            logging_steps=10,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=[reward_fn],
            args=grpo_config,
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final checkpoint
        final_dir = Path(output_dir) / "checkpoint-final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        print(f"Saved final checkpoint to {final_dir}")


class TinkerRLTrainer:
    """GRPO trainer using the Tinker managed GPU platform.

    Implements GRPO via Tinker's ``importance_sampling`` loss with
    group-based rollouts and per-group advantage centering, following the
    pattern from the tinker-cookbook ``rl_loop.py``.

    The training loop runs on your local CPU and issues API calls to
    Tinker's managed GPU cluster for sampling and gradient updates.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

    def _load_prompts(self) -> list[dict]:
        """Load prompt dataset from JSONL."""
        prompts = []
        with open(self.config.train_file, "r") as f:
            for line in f:
                prompts.append(json.loads(line))
        logger.info(f"Loaded {len(prompts)} training prompts")
        return prompts

    @staticmethod
    def _get_reward(completion_text: str, target: str) -> float:
        """Score a single completion against its target."""
        return sgtr_binary_reward([completion_text], [target])[0]

    def _log_example_prompt(self, prompts: list[dict]) -> None:
        """Log an example prompt so the user can verify data looks right."""
        example = prompts[0]
        prompt_text = example["prompt"]
        # Truncate long prompts for readability
        if len(prompt_text) > 1000:
            display = prompt_text[:500] + "\n  [...truncated...]\n" + prompt_text[-200:]
        else:
            display = prompt_text
        logger.info(
            f"Example training prompt (target={example['target']}):\n"
            f"  ---\n  {display}\n  ---"
        )

    def _log_example_output(
        self, renderer, sequence, target: str, reward: float, is_first: bool
    ) -> None:
        """Log the first model output of training so the user can see what it generates."""
        if not is_first:
            return
        from tinker_cookbook import renderers as r

        parsed_msg, _ = renderer.parse_response(sequence.tokens)
        content = r.get_text_content(parsed_msg)
        logger.info(
            f"Example model output (first sample of training):\n"
            f"  completion: {content!r}\n"
            f"  target: {target!r}, reward: {reward}"
        )

    def train(self, resume_from_checkpoint: str | None = None):
        """Run GRPO training via Tinker API.

        For each batch of prompts:
        1. Sample ``num_rollouts_per_prompt`` completions per prompt
        2. Score each completion with the binary SGTR reward
        3. Compute GRPO advantages (per-group mean centering)
        4. Build Tinker Datum objects and run forward_backward + optim_step
        """
        import tinker
        from tinker import types
        from tinker.types.tensor_data import TensorData
        import torch
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        cfg = self.config
        prompts = self._load_prompts()

        # Log example prompt
        self._log_example_prompt(prompts)

        # Tinker setup
        logger.info(f"Connecting to Tinker (model={cfg.model_name}, lora_rank={cfg.lora_rank})...")
        service_client = tinker.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=cfg.model_name, rank=cfg.lora_rank
        )
        logger.info("Tinker training client created")

        tokenizer = get_tokenizer(cfg.model_name)
        renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        logger.info(f"Using renderer: {renderer_name}")

        sampling_params = types.SamplingParams(
            max_tokens=cfg.max_completion_length,
            stop=renderer.get_stop_sequences(),
        )
        adam_params = types.AdamParams(
            learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )

        group_size = cfg.num_rollouts_per_prompt
        batch_size = cfg.per_device_train_batch_size
        n_batches = len(prompts) // batch_size
        n_epochs = cfg.num_epochs

        log_dir = cfg.output_dir or f"data/checkpoints/{cfg.experiment_name}"
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Training: {n_epochs} epochs, {n_batches} batches/epoch, "
            f"batch_size={batch_size}, group_size={group_size}, "
            f"total_steps={n_batches * n_epochs}"
        )

        global_step = 0
        logged_first_output = False
        cumulative_correct = 0
        cumulative_total = 0

        for epoch in range(n_epochs):
            epoch_rewards: list[float] = []

            for batch_idx in range(n_batches):
                t_start = time.time()
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(prompts))
                batch = prompts[batch_start:batch_end]

                # Get fresh sampling client from current weights
                sampling_client = training_client.save_weights_and_get_sampling_client()

                # Build model inputs and fire off sampling requests
                futures = []
                model_inputs = []
                for item in batch:
                    convo = [{"role": "user", "content": item["prompt"]}]
                    model_input = renderer.build_generation_prompt(convo)
                    future = sampling_client.sample(
                        prompt=model_input,
                        num_samples=group_size,
                        sampling_params=sampling_params,
                    )
                    futures.append(future)
                    model_inputs.append(model_input)

                # Collect results, compute rewards and advantages, build datums
                datums: list[types.Datum] = []
                batch_rewards: list[float] = []
                batch_correct = 0
                batch_total = 0

                for future, prompt_input, item in zip(futures, model_inputs, batch):
                    sample_result = future.result()
                    target = item["target"]

                    rewards_G: list[float] = []
                    sampled_tokens_G: list[list[int]] = []
                    logprobs_G: list[list[float]] = []

                    for sequence in sample_result.sequences:
                        sampled_tokens_G.append(sequence.tokens)
                        assert sequence.logprobs is not None
                        logprobs_G.append(sequence.logprobs)

                        parsed_msg, _ = renderer.parse_response(sequence.tokens)
                        content = renderers.get_text_content(parsed_msg)
                        reward = self._get_reward(content, target)
                        rewards_G.append(reward)

                        # Log the very first output
                        if not logged_first_output:
                            self._log_example_output(
                                renderer, sequence, target, reward, True
                            )
                            logged_first_output = True

                    batch_correct += sum(int(r == 1.0) for r in rewards_G)
                    batch_total += len(rewards_G)

                    # GRPO: center advantages within the group
                    mean_reward = sum(rewards_G) / len(rewards_G)
                    advantages_G = [r - mean_reward for r in rewards_G]
                    batch_rewards.append(mean_reward)

                    # Skip groups where all rewards are identical (no signal)
                    if all(a == 0.0 for a in advantages_G):
                        continue

                    # Build training datums
                    ob_len = prompt_input.length - 1
                    for tokens, logprobs, advantage in zip(
                        sampled_tokens_G, logprobs_G, advantages_G
                    ):
                        model_input = prompt_input.append(
                            types.EncodedTextChunk(tokens=tokens[:-1])
                        )
                        target_tokens = [0] * ob_len + tokens
                        padded_logprobs = [0.0] * ob_len + logprobs
                        padded_advantages = (
                            [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                        )
                        assert (
                            model_input.length
                            == len(target_tokens)
                            == len(padded_logprobs)
                            == len(padded_advantages)
                        )
                        datums.append(
                            types.Datum(
                                model_input=model_input,
                                loss_fn_inputs={
                                    "target_tokens": TensorData.from_torch(
                                        torch.tensor(target_tokens)
                                    ),
                                    "logprobs": TensorData.from_torch(
                                        torch.tensor(padded_logprobs)
                                    ),
                                    "advantages": TensorData.from_torch(
                                        torch.tensor(padded_advantages)
                                    ),
                                },
                            )
                        )

                # Training step
                if datums:
                    fwd_bwd_future = training_client.forward_backward(
                        datums, loss_fn="importance_sampling"
                    )
                    optim_future = training_client.optim_step(adam_params)
                    fwd_bwd_future.result()
                    optim_future.result()

                avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
                epoch_rewards.extend(batch_rewards)
                cumulative_correct += batch_correct
                cumulative_total += batch_total
                elapsed = time.time() - t_start
                batch_acc = batch_correct / batch_total if batch_total else 0.0
                running_acc = cumulative_correct / cumulative_total if cumulative_total else 0.0

                logger.info(
                    f"[epoch {epoch+1}/{n_epochs}] batch {batch_idx+1}/{n_batches} | "
                    f"reward={avg_reward:.3f} | acc={batch_acc:.1%} (running={running_acc:.1%}) | "
                    f"datums={len(datums)} | {elapsed:.1f}s"
                )
                global_step += 1

            epoch_avg = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            logger.info(
                f"Epoch {epoch+1} complete | avg reward={epoch_avg:.3f} | "
                f"running acc={cumulative_correct}/{cumulative_total} "
                f"= {cumulative_correct/cumulative_total:.1%}" if cumulative_total else
                f"Epoch {epoch+1} complete | avg reward={epoch_avg:.3f}"
            )

        # Save final checkpoint
        from tinker_cookbook import checkpoint_utils

        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=log_dir,
            kind="both",
            loop_state={"batch": global_step},
        )
        logger.info(f"Training complete. Checkpoint saved to {log_dir}")
        logger.info(
            f"Final stats: {global_step} steps, "
            f"accuracy={cumulative_correct}/{cumulative_total} "
            f"= {cumulative_correct/cumulative_total:.1%}" if cumulative_total else
            f"Final stats: {global_step} steps"
        )
