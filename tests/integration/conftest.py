"""Shared fixtures for integration tests of Tinker-based trainers.

Provides mock module injection for tinker/tinker_cookbook and tiny data fixtures.
"""

import json
import sys
import types as builtin_types
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest


def _pw_record(id: str, target: str, **extra) -> dict:
    return {
        "prompt": f"Which response did you write? (id={id})",
        "target": target,
        "id": id,
        "format": "pw",
        **extra,
    }


def _write_jsonl(path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@contextmanager
def patch_tinker_modules(
    answer_text="1",
    num_sequences=1,
    logprob_value=-0.1,
):
    """Inject fake tinker/tinker_cookbook modules into sys.modules.

    Args:
        answer_text: What the mock model "generates" (parsed by extract_answer).
        num_sequences: Sequences per sample() call (= group_size for GRPO).
        logprob_value: Logprob value returned by forward_backward (for SFT
            accuracy testing; values > log(0.5) ~ -0.693 count as correct).
    """
    module_names = [
        "tinker",
        "tinker.types",
        "tinker.types.tensor_data",
        "tinker_cookbook",
        "tinker_cookbook.checkpoint_utils",
        "tinker_cookbook.model_info",
        "tinker_cookbook.renderers",
        "tinker_cookbook.supervised",
        "tinker_cookbook.supervised.common",
        "tinker_cookbook.supervised.data",
        "tinker_cookbook.tokenizer_utils",
        "tinker_cookbook.utils",
        "tinker_cookbook.utils.ml_log",
        "tinker_cookbook.types",
    ]

    modules = {}
    for name in module_names:
        mod = builtin_types.ModuleType(name)
        modules[name] = mod

    tinker = modules["tinker"]
    tinker_types = modules["tinker.types"]
    tensor_data_mod = modules["tinker.types.tensor_data"]
    cookbook = modules["tinker_cookbook"]
    checkpoint_utils = modules["tinker_cookbook.checkpoint_utils"]
    model_info = modules["tinker_cookbook.model_info"]
    renderers_mod = modules["tinker_cookbook.renderers"]
    supervised = modules["tinker_cookbook.supervised"]
    supervised_common = modules["tinker_cookbook.supervised.common"]
    supervised_data = modules["tinker_cookbook.supervised.data"]
    tokenizer_utils = modules["tinker_cookbook.tokenizer_utils"]
    utils_mod = modules["tinker_cookbook.utils"]
    ml_log_mod = modules["tinker_cookbook.utils.ml_log"]
    cookbook_types = modules["tinker_cookbook.types"]

    tinker.types = tinker_types
    tinker_types.tensor_data = tensor_data_mod
    cookbook.checkpoint_utils = checkpoint_utils
    cookbook.model_info = model_info
    cookbook.renderers = renderers_mod
    cookbook.supervised = supervised
    supervised.common = supervised_common
    supervised.data = supervised_data
    cookbook.tokenizer_utils = tokenizer_utils
    cookbook.utils = utils_mod
    utils_mod.ml_log = ml_log_mod
    cookbook.types = cookbook_types

    mocks = {}

    # Renderer
    renderer = MagicMock(name="renderer")
    renderer.get_stop_sequences.return_value = ["<|end|>"]

    base_prompt_length = 10
    mock_model_input = MagicMock(name="model_input")
    mock_model_input.length = base_prompt_length

    def mock_append(chunk):
        appended = MagicMock(name="appended_model_input")
        appended.length = base_prompt_length + len(chunk.tokens)
        return appended

    mock_model_input.append = MagicMock(side_effect=mock_append)
    renderer.build_generation_prompt.return_value = mock_model_input

    parsed_msg = MagicMock(name="parsed_msg")
    renderer.parse_response.return_value = (parsed_msg, None)
    mocks["renderer"] = renderer

    renderers_mod.get_renderer = MagicMock(return_value=renderer)
    renderers_mod.get_text_content = MagicMock(return_value=answer_text)
    renderers_mod.TrainOnWhat = MagicMock()
    renderers_mod.TrainOnWhat.LAST_ASSISTANT_MESSAGE = "LAST_ASSISTANT_MESSAGE"
    mocks["renderers_mod"] = renderers_mod

    model_info.get_recommended_renderer_name = MagicMock(return_value="test_renderer")
    tokenizer_utils.get_tokenizer = MagicMock(return_value=MagicMock(name="tokenizer"))

    # Types
    tinker_types.AdamParams = MagicMock(name="AdamParams")
    tinker_types.SamplingParams = MagicMock(name="SamplingParams")
    tinker_types.Datum = MagicMock(name="Datum")
    tinker.Datum = tinker_types.Datum

    class FakeEncodedTextChunk:
        def __init__(self, tokens):
            self.tokens = tokens

    tinker_types.EncodedTextChunk = FakeEncodedTextChunk

    tensor_data_mock = MagicMock(name="TensorData")
    tensor_data_mock.from_torch = MagicMock(side_effect=lambda x: x)
    tensor_data_mod.TensorData = tensor_data_mock

    cookbook_types.SamplingParams = MagicMock(name="CookbookSamplingParams")

    # ServiceClient -> training_client
    training_client = MagicMock(name="training_client")
    service_client = MagicMock(name="service_client")
    service_client.create_lora_training_client.return_value = training_client
    tinker.ServiceClient = MagicMock(return_value=service_client)
    mocks["service_client"] = service_client
    mocks["training_client"] = training_client

    # forward_backward
    def make_fwd_bwd_result(datums, loss_fn):
        logprob_mocks = []
        for _ in datums:
            lp_mock = MagicMock(name="logprob_entry")
            lp_mock.to_torch.return_value = np.array(
                [0.0] * 9 + [logprob_value]
            )
            logprob_mocks.append({"logprobs": lp_mock})

        result = MagicMock(name="fwd_bwd_result")
        result.loss_fn_outputs = logprob_mocks
        future = MagicMock(name="fwd_bwd_future")
        future.result.return_value = result
        return future

    training_client.forward_backward = MagicMock(
        side_effect=make_fwd_bwd_result
    )

    # optim_step
    optim_future = MagicMock(name="optim_future")
    optim_future.result.return_value = None
    training_client.optim_step = MagicMock(return_value=optim_future)

    # save_weights_and_get_sampling_client -> sampling_client
    sampling_client = MagicMock(name="sampling_client")

    def make_sample_result(prompt, num_samples, sampling_params):
        n = num_samples if num_samples else num_sequences
        sequences = []
        for _ in range(n):
            seq = MagicMock(name="sequence")
            seq.tokens = [101, 102, 103]
            seq.logprobs = [-0.1, -0.2, -0.3]
            sequences.append(seq)
        result = MagicMock(name="sample_result")
        result.sequences = sequences
        future = MagicMock(name="sample_future")
        future.result.return_value = result
        return future

    sampling_client.sample = MagicMock(side_effect=make_sample_result)
    training_client.save_weights_and_get_sampling_client = MagicMock(
        return_value=sampling_client
    )
    mocks["sampling_client"] = sampling_client

    # conversation_to_datum
    def make_datum(convo, renderer_, tokenizer_, train_on_what):
        datum = MagicMock(name="datum")
        datum.loss_fn_inputs = {
            "weights": MagicMock(
                to_torch=MagicMock(
                    return_value=np.array([0.0] * 9 + [1.0])
                )
            )
        }
        return datum

    supervised_data.conversation_to_datum = MagicMock(side_effect=make_datum)
    mocks["conversation_to_datum"] = supervised_data.conversation_to_datum

    supervised_common.compute_mean_nll = MagicMock(return_value=0.5)
    mocks["compute_mean_nll"] = supervised_common.compute_mean_nll

    ml_logger = MagicMock(name="ml_logger")
    ml_log_mod.setup_logging = MagicMock(return_value=ml_logger)
    mocks["ml_logger"] = ml_logger

    checkpoint_utils.save_checkpoint = MagicMock()
    mocks["checkpoint_utils"] = checkpoint_utils

    saved = {name: sys.modules.get(name) for name in module_names}
    try:
        for name, mod in modules.items():
            sys.modules[name] = mod
        yield mocks
    finally:
        for name in module_names:
            prev = saved[name]
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _build_ctx(mocks):
    """Build a TinkerContext from the mock objects returned by patch_tinker_modules."""
    from sgtr_rl.tinker import TinkerContext

    return TinkerContext(
        training_client=mocks["training_client"],
        renderer=mocks["renderer"],
        tokenizer=MagicMock(name="tokenizer"),
        eval_params=MagicMock(name="eval_params"),
        adam_params=MagicMock(name="adam_params"),
        ml_logger=mocks["ml_logger"],
    )


@pytest.fixture()
def tiny_train_val_files(tmp_path):
    """Create tiny PW train/val JSONL files (flat schema)."""
    train_records = [
        _pw_record("train-id-1", "1"),
        _pw_record("train-id-1", "2"),
        _pw_record("train-id-2", "1"),
        _pw_record("train-id-2", "2"),
    ]
    val_records = [
        _pw_record("val-id-1", "1"),
        _pw_record("val-id-1", "2"),
    ]
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)
    return str(train_path), str(val_path)


@pytest.fixture()
def tiny_prompts(tiny_train_val_files):
    """Load tiny train/val prompts as lists of dicts."""
    from sgtr_rl.data import load_jsonl

    train_path, val_path = tiny_train_val_files
    return load_jsonl(train_path), load_jsonl(val_path)


@pytest.fixture()
def sft_config(tmp_path, tiny_train_val_files):
    """TrainingConfig for SFT with tiny data."""
    from sgtr_rl.config import TrainingConfig

    train_file, val_file = tiny_train_val_files
    run_dir = str(tmp_path / "run")
    return TrainingConfig(
        algorithm="sft",
        experiment_name="test_sft",
        model_name="test-model",
        num_epochs=2,
        batch_size=2,
        train_file=train_file,
        val_file=val_file,
        run_dir=run_dir,
    )


@pytest.fixture()
def grpo_config(tmp_path, tiny_train_val_files):
    """TrainingConfig for GRPO with tiny data."""
    from sgtr_rl.config import TrainingConfig

    train_file, val_file = tiny_train_val_files
    run_dir = str(tmp_path / "run")
    return TrainingConfig(
        algorithm="grpo",
        experiment_name="test_grpo",
        model_name="test-model",
        num_epochs=2,
        batch_size=2,
        num_rollouts_per_prompt=2,
        train_file=train_file,
        val_file=val_file,
        run_dir=run_dir,
    )
