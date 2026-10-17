# CS336 Spring 2026 Assignment 5: Alignment

[中文说明 / Chinese README](README.zh-CN.md)

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Section 4: on-policy GRPO

The reusable GRPO components are in `cs336_alignment/grpo.py`. Run the
section 4 checks with:

```sh
uv run pytest tests/test_grpo.py -k 'tokenize_prompt_and_output or get_response_log_probs or compute_rollout_rewards or compute_group_normalized_rewards_grpo or compute_policy_gradient_loss_on_policy or aggregate_loss_across_microbatch_sequence or grpo_train_step_standard_on_policy'
```

The training script uses GPU 0 for the policy and GPU 1 for vLLM. Its defaults
match the suggested section 4 settings, including the zero-shot `r1_zero`
prompt and 200 rollout steps:

```sh
uv run --extra gpu python scripts/grpo.py --output-dir runs/grpo-seed-0 --seed 0
```

Each run writes `metrics.jsonl`, sampled `rollouts.jsonl`, and a final model
checkpoint under the output directory. Add `--wandb-project NAME` to log the
same metrics to Weights & Biases.

