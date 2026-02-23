"""Build SGTR training prompts from generated text data.

Uses the self-rec-framework prompt system (installed as a package) to construct
prompts identical to those used during evaluation, returning them as dicts
suitable for RL training (prompt + target).
"""

import json
from pathlib import Path

from self_rec_framework.src.inspect.config import load_experiment_config, ensure_evaluator_reasoning
from self_rec_framework.src.inspect.data import load_dataset_individual, load_dataset_pairwise


def build_sgtr_prompts(
    evaluator_model: str,
    generator_models: list[str],
    experiment_config_path: str | Path,
    dataset_name: str,
    data_subset: str,
    format: str = "ind",
) -> list[dict]:
    """Build SGTR training prompts from static repo data.

    Args:
        evaluator_model: Short model name for the evaluator (e.g. "ll-3.1-8b").
        generator_models: Short model names used as generators.  For IND the
            evaluator is compared against each generator.  For PW the evaluator's
            output is paired with each generator's output.
        experiment_config_path: Path to an SGTR experiment config YAML
            (e.g. experiments/ICML_02_.../config.yaml).
        dataset_name: Dataset identifier (e.g. "wikisum").
        data_subset: Data subset directory (e.g. "test_set_1-30").
        format: "ind" for individual or "pw" for pairwise.

    Returns:
        List of dicts with keys ``prompt``, ``target``, ``metadata``.
    """
    exp_config = load_experiment_config(experiment_config_path, dataset_name)
    ensure_evaluator_reasoning(exp_config, evaluator_model)

    if format == "ind":
        return _build_ind_prompts(
            evaluator_model, generator_models, exp_config, dataset_name, data_subset
        )
    elif format == "pw":
        return _build_pw_prompts(
            evaluator_model, generator_models, exp_config, dataset_name, data_subset
        )
    else:
        raise ValueError(f"Unknown format: {format!r}. Use 'ind' or 'pw'.")


def _build_ind_prompts(
    evaluator_model, generator_models, exp_config, dataset_name, data_subset
) -> list[dict]:
    """Build individual-format SGTR prompts.

    For each generator model we create two groups of samples:
    - Control: the evaluator's own text (target depends on choice token mapping)
    - Treatment: the generator's text (target depends on choice token mapping)
    """
    prompts = []

    for generator_model in generator_models:
        if generator_model == evaluator_model:
            continue

        # Control samples: evaluator's own output
        control_samples = load_dataset_individual(
            treatment_name=evaluator_model,
            dataset_name=dataset_name,
            data_subset=data_subset,
            is_control=True,
        )
        for sample in control_samples:
            prompt_text = _format_ind_prompt(exp_config, sample)
            prompts.append({
                "prompt": prompt_text,
                "target": sample["metadata"]["correct_answer"],
                "metadata": {
                    **sample["metadata"],
                    "evaluator_model": evaluator_model,
                    "generator_model": generator_model,
                    "sample_type": "control",
                },
            })

        # Treatment samples: generator's output
        treatment_samples = load_dataset_individual(
            treatment_name=generator_model,
            dataset_name=dataset_name,
            data_subset=data_subset,
            is_control=False,
        )
        for sample in treatment_samples:
            prompt_text = _format_ind_prompt(exp_config, sample)
            prompts.append({
                "prompt": prompt_text,
                "target": sample["metadata"]["correct_answer"],
                "metadata": {
                    **sample["metadata"],
                    "evaluator_model": evaluator_model,
                    "generator_model": generator_model,
                    "sample_type": "treatment",
                },
            })

    return prompts


def _build_pw_prompts(
    evaluator_model, generator_models, exp_config, dataset_name, data_subset
) -> list[dict]:
    """Build pairwise-format SGTR prompts."""
    prompts = []

    for generator_model in generator_models:
        if generator_model == evaluator_model:
            continue

        samples = load_dataset_pairwise(
            treatment_name_1=evaluator_model,
            treatment_name_2=generator_model,
            dataset_name=dataset_name,
            data_subset=data_subset,
        )

        for sample in samples:
            generation_prompt = exp_config.generation_prompt.format(content=sample["content"])
            reasoning1 = sample.get("cot1") or ""
            reasoning2 = sample.get("cot2") or ""
            prompt_text = exp_config.SR_task_prompt.format(
                generation_prompt=generation_prompt,
                output1=sample["output1"],
                output2=sample["output2"],
                reasoning1=reasoning1,
                reasoning2=reasoning2,
            )

            prompts.append({
                "prompt": prompt_text,
                "target": sample["metadata"]["correct_answer"],
                "metadata": {
                    **sample["metadata"],
                    "evaluator_model": evaluator_model,
                    "generator_model": generator_model,
                },
            })

    return prompts


def _format_ind_prompt(exp_config, sample: dict) -> str:
    """Format a single individual-query prompt from a sample dict."""
    generation_prompt = exp_config.generation_prompt.format(content=sample["content"])
    return exp_config.SR_task_prompt.format(
        generation_prompt=generation_prompt,
        output=sample["output"],
        reasoning=sample.get("reasoning", ""),
        correct_choice_token=sample["metadata"].get("correct_choice_token", "1"),
        incorrect_choice_token=sample["metadata"].get("incorrect_choice_token", "2"),
    )


def save_prompt_dataset(prompts: list[dict], output_path: str | Path) -> None:
    """Save prompts as JSONL for training.

    Each line is a JSON object with ``prompt``, ``target``, and ``metadata``.

    Args:
        prompts: List of prompt dicts from :func:`build_sgtr_prompts`.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(prompts)} prompts to {output_path}")
