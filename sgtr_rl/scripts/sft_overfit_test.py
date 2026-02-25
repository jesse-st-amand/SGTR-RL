"""Quick script to test SFT overfitting on a mixed batch with train-set eval.

Usage:
    uv run python sgtr_rl/scripts/sft_overfit_test.py --lr 1e-4 --epochs 30
"""

import argparse
import json

from dotenv import load_dotenv
load_dotenv()

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.common import compute_mean_nll
from sgtr_rl.training.reward import _extract_answer


def run(lr: float, n_epochs: int, eval_every: int, rank: int = 32):
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=model_name, rank=rank)
    adam_params = types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    with open("data/training_data/pw_debug_mixed/train.jsonl") as f:
        samples = [json.loads(l) for l in f]

    print(f"lr={lr}, rank={rank}, epochs={n_epochs}, samples={len(samples)}, targets={[s['target'] for s in samples]}")

    for epoch in range(n_epochs):
        datums = []
        for item in samples:
            convo = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["target"]},
            ]
            datum = conversation_to_datum(convo, renderer, None, TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            datums.append(datum)

        fwd_bwd_future = training_client.forward_backward(datums, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [d.loss_fn_inputs["weights"] for d in datums]
        nll = compute_mean_nll(logprobs, weights)

        if (epoch + 1) % eval_every == 0:
            sampling_client = training_client.save_weights_and_get_sampling_client()
            eval_params = types.SamplingParams(
                max_tokens=16, stop=renderer.get_stop_sequences(), temperature=0.0
            )
            correct = 0
            preds = []
            for item in samples:
                convo = [{"role": "user", "content": item["prompt"]}]
                model_input = renderer.build_generation_prompt(convo)
                future = sampling_client.sample(
                    prompt=model_input, num_samples=1, sampling_params=eval_params
                )
                result = future.result()
                seq = result.sequences[0]
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(parsed_msg)
                answer = _extract_answer(content)
                target = item["target"]
                if answer == target:
                    correct += 1
                preds.append(f"{answer}({'T' if answer == target else 'F'})")
            print(
                f"Epoch {epoch+1}: nll={nll:.4f}, "
                f"train_acc={correct}/{len(samples)} "
                f"preds={preds}"
            )
        else:
            print(f"Epoch {epoch+1}: nll={nll:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    run(args.lr, args.epochs, args.eval_every, args.rank)
