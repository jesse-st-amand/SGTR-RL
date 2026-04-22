"""Tests for sgtr_rl.tinker_eval."""

import sys
from types import ModuleType, SimpleNamespace

from sgtr_rl.tinker_eval import compute_val_nll


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _FakeTrainingClient:
    def __init__(self):
        self.forward_calls = []
        self.forward_backward_called = False
        self.optim_step_called = False

    def forward(self, datums, loss_fn):
        self.forward_calls.append((datums, loss_fn))
        return _FakeFuture(SimpleNamespace(loss_fn_outputs=[{"logprobs": "LP"}]))

    def forward_backward(self, *args, **kwargs):
        self.forward_backward_called = True
        raise AssertionError("compute_val_nll should not call forward_backward")

    def optim_step(self, *args, **kwargs):
        self.optim_step_called = True
        raise AssertionError("compute_val_nll should not call optim_step")


def test_compute_val_nll_uses_forward_without_optimizer_step(monkeypatch):
    fake_renderers = ModuleType("tinker_cookbook.renderers")

    class _TrainOnWhat:
        LAST_ASSISTANT_MESSAGE = "last-assistant-message"

    fake_renderers.TrainOnWhat = _TrainOnWhat

    fake_common = ModuleType("tinker_cookbook.supervised.common")

    def fake_compute_mean_nll(logprobs_list, weights_list):
        assert logprobs_list == ["LP"]
        assert weights_list == ["W"]
        return 0.123

    fake_common.compute_mean_nll = fake_compute_mean_nll

    fake_data = ModuleType("tinker_cookbook.supervised.data")

    def fake_conversation_to_datum(convo, renderer, max_length, train_on_what):
        assert convo[-1] == {"role": "assistant", "content": "1"}
        assert train_on_what == _TrainOnWhat.LAST_ASSISTANT_MESSAGE
        return SimpleNamespace(loss_fn_inputs={"weights": "W"})

    fake_data.conversation_to_datum = fake_conversation_to_datum

    fake_supervised = ModuleType("tinker_cookbook.supervised")
    fake_supervised.common = fake_common
    fake_supervised.data = fake_data

    fake_root = ModuleType("tinker_cookbook")
    fake_root.renderers = fake_renderers
    fake_root.supervised = fake_supervised

    monkeypatch.setitem(sys.modules, "tinker_cookbook", fake_root)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.renderers", fake_renderers)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.supervised", fake_supervised)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.supervised.common", fake_common)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.supervised.data", fake_data)

    training_client = _FakeTrainingClient()
    ctx = SimpleNamespace(training_client=training_client, renderer=object())
    val_prompts = [{"prompt": "hello", "target": "1", "id": "a"}]

    result = compute_val_nll(val_prompts, ctx)

    assert result == 0.123
    assert training_client.forward_calls
    assert training_client.forward_calls[0][1] == "cross_entropy"
    assert training_client.forward_backward_called is False
    assert training_client.optim_step_called is False
