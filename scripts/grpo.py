"""Train OLMo on GSM8K with the on-policy GRPO algorithm from section 4.

Example:
    uv run --extra gpu python scripts/grpo.py --output-dir runs/grpo-seed-0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.grpo import compute_rollout_rewards, grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_gsm8k(path: Path, limit: int, seed: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as source:
        examples = [json.loads(line) for line in source if line.strip()]
    if not 0 < limit <= len(examples):
        raise ValueError(f"Requested {limit} examples from {path}, which has {len(examples)}.")
    random.Random(seed).shuffle(examples)
    return examples[:limit]


def ground_truth(example: dict[str, str]) -> str:
    return example["answer"].split("####")[-1].strip()


def make_sampling_params(
    temperature: float, max_tokens: int, use_answer_stop: bool, top_p: float = 1.0
) -> dict:
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": None,
    }
    if use_answer_stop:
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


def evaluate(
    server: VLLMServer,
    examples: list[dict[str, str]],
    prompt_template: str,
    reward_fn,
    sampling_params: dict,
    generation_batch_size: int,
) -> tuple[dict[str, float], list[dict[str, str | float]]]:
    prompts = [prompt_template.format(question=example["question"]) for example in examples]
    completions = server.generate_completions(
        prompts, sampling_params, batch_size=generation_batch_size
    )
    responses = [completion.text for completion in completions]
    truths = [ground_truth(example) for example in examples]
    _, reward_metadata = compute_rollout_rewards(reward_fn, responses, truths)
    metrics = {
        "val_reward": reward_metadata["train_reward"],
        "val_format_reward": reward_metadata["train_format_reward"],
        "val_answer_reward": reward_metadata["train_answer_reward"],
        "val_response_length": sum(len(completion.token_ids) for completion in completions)
        / len(completions),
    }
    samples = [
        {
            "question": example["question"],
            "ground_truth": truth,
            "response": response,
            "reward": reward_fn(response, truth)["reward"],
        }
        for example, response, truth in zip(examples[:8], responses[:8], truths[:8])
    ]
    return metrics, samples


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default="allenai/OLMo-2-0425-1B")
    parser.add_argument(
        "--prompt-file", type=Path, default=REPO_ROOT / "cs336_alignment/prompts/r1_zero.prompt"
    )
    parser.add_argument("--train-file", type=Path, default=REPO_ROOT / "data/gsm8k/train.jsonl")
    parser.add_argument("--val-file", type=Path, default=REPO_ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--wandb-project", default=None)
    return parser


def main() -> None:
    args = make_parser().parse_args() #获取输入的参数
    if torch.cuda.device_count() < 2:
        raise RuntimeError("GRPO training needs GPU 0 for the policy and GPU 1 for vLLM.")
    if args.group_size < 2 or args.rollout_batch_size < 1:
        raise ValueError("group-size must be at least 2 and rollout-batch-size must be positive.")
    if args.rollout_batch_size % args.group_size:
        raise ValueError("rollout-batch-size must be divisible by group-size.")
    if args.eval_every < 1 or args.log_rollouts_every < 1:
        raise ValueError("Logging intervals must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )
    train_examples = read_gsm8k(args.train_file, args.n_train_examples, args.seed)
    val_examples = read_gsm8k(args.val_file, args.n_val_examples, args.seed)
    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    use_answer_stop = "<answer>" in prompt_template  #布尔值
    from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

    reward_fn = r1_zero_reward_fn if use_answer_stop else question_only_reward_fn
    sampling_params = make_sampling_params(
        args.temperature, args.max_tokens, use_answer_stop, args.top_p
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    policy, tokenizer = get_model_and_tokenizer(args.model_name_or_path, "cuda:0")
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    tracker = None
    if args.wandb_project:
        import wandb

        tracker = wandb.init(
            project=args.wandb_project,
            config=json.loads(json.dumps(vars(args), default=str)),
        )

    server = VLLMServer(model_id=args.model_name_or_path, gpu=1, seed=args.seed, startup_timeout=1800)
    metrics_path = args.output_dir / "metrics.jsonl"
    samples_path = args.output_dir / "rollouts.jsonl"
    try:
        server.start()
        server.init_weight_sync("cuda:0")
        server.sync_policy_weights(policy)
        val_metrics, samples = evaluate(
            server, val_examples, prompt_template, reward_fn, sampling_params,
            args.generation_batch_size,
        )
        append_jsonl(metrics_path, {"step": 0, **val_metrics})
        append_jsonl(samples_path, {"step": 0, "split": "val", "samples": samples})
        if tracker is not None:
            tracker.log(val_metrics, step=0)

        prompts_per_batch = args.rollout_batch_size // args.group_size
        for step in range(1, args.num_rollout_steps + 1):
            batch = random.sample(train_examples, prompts_per_batch)
            prompts = [prompt_template.format(question=example["question"]) for example in batch]
            repeated_prompts = [prompt for prompt in prompts for _ in range(args.group_size)]
            repeated_truths = [ground_truth(example) for example in batch for _ in range(args.group_size)]
            server.sync_policy_weights(policy)
            completions = server.generate_completions(
                repeated_prompts, sampling_params, batch_size=args.generation_batch_size
            )
            responses = [completion.text for completion in completions]
            loss, train_metrics = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=responses,
                repeated_ground_truths=repeated_truths,
                group_size=args.group_size,
            )
            metrics = {"step": step, **train_metrics}
            if step % args.eval_every == 0 or step == args.num_rollout_steps:
                server.sync_policy_weights(policy)
                val_metrics, samples = evaluate(
                    server, val_examples, prompt_template, reward_fn, sampling_params,
                    args.generation_batch_size,
                )
                metrics.update(val_metrics)
                append_jsonl(samples_path, {"step": step, "split": "val", "samples": samples})
            if step % args.log_rollouts_every == 0:
                samples = [
                    {
                        "question": example["question"],
                        "ground_truth": ground_truth(example),
                        "responses": responses[index * args.group_size:(index + 1) * args.group_size],
                    }
                    for index, example in enumerate(batch[:8])
                ]
                append_jsonl(samples_path, {"step": step, "split": "train", "samples": samples})
            append_jsonl(metrics_path, metrics)
            if tracker is not None:
                tracker.log({key: value for key, value in metrics.items() if key != "step"}, step=step)
            print(json.dumps(metrics), flush=True)

        checkpoint_dir = args.output_dir / "final_model"
        policy.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
    finally:
        server.stop()
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
