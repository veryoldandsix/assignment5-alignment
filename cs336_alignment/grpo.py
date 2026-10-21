"""On-policy Group Relative Policy Optimization components."""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    if len(prompt_strs) != len(output_strs) or not prompt_strs:
        raise ValueError("Prompts and responses must be nonempty, equally sized batches.")

    prompt_ids = [tokenizer.encode(text, add_special_tokens=False) for text in prompt_strs]
    output_ids = [tokenizer.encode(text, add_special_tokens=False) for text in output_strs]
    combined = [prompt + output for prompt, output in zip(prompt_ids, output_ids)]
    if any(len(ids) < 2 for ids in combined):
        raise ValueError("Each prompt and response must contain at least two tokens together.")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("The tokenizer needs a pad or EOS token for batch padding.")

    padded_length = max(len(ids) for ids in combined)
    input_ids, labels, response_mask = [], [], []
    for prompt, output, ids in zip(prompt_ids, output_ids, combined):
        padding = padded_length - len(ids)
        padded_ids = ids + [pad_id] * padding
        input_ids.append(padded_ids[:-1]) #去掉最后一个元素
        labels.append(padded_ids[1:]) #去掉第一个元素
        token_mask = [False] * len(prompt) + [True] * len(output) + [False] * padding #对应掩码
        response_mask.append(token_mask[1:]) #生成掩码矩阵

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    logits = model(input_ids).logits.float()
    all_log_probs = F.log_softmax(logits, dim=-1) #按照最后一维进行log_softmax 然后再取log 如何呢
    selected_log_probs = all_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": selected_log_probs} #返回的是取到正确预测词的概率  [对应问题 ，生成的概率]
    if return_token_entropy:
        result["token_entropy"] = -(all_log_probs.exp() * all_log_probs).sum(dim=-1)
    return result


def compute_rollout_rewards( #
    reward_fn: Callable[[str, str], dict[str, float]], #这个是接收“模型回答、标准答案”，返回奖励字典
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    if len(rollout_responses) != len(repeated_ground_truths) or not rollout_responses:
        raise ValueError("Responses and ground truths must be nonempty, equally sized batches.")
    #scores是一个字典 保存相关得分类型和分数
    scores = [reward_fn(response, truth) for response, truth in zip(rollout_responses, repeated_ground_truths)] #zip是把rollout_responses repeated_ground_truths两两配对
    rewards = torch.tensor([score["reward"] for score in scores], dtype=torch.float32) #将reward展成一位张量
    metadata = {
        "train_reward": rewards.mean().item(),
        "train_format_reward": sum(score["format_reward"] for score in scores) / len(scores),
        "train_answer_reward": sum(score["answer_reward"] for score in scores) / len(scores),
    }
    return rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, dict[str, float]]:
    if (
        raw_rewards.ndim != 1 #输入一个一维张量
        or group_size < 2
        or not raw_rewards.numel()
        or raw_rewards.numel() % group_size
    ):
        raise ValueError("Rewards must be a flat batch of complete groups.")
    if baseline != "mean" or advantage_normalizer != "std":
        raise NotImplementedError("Section 4 implements standard on-policy GRPO only.")
    grouped = raw_rewards.reshape(-1, group_size) #让 PyTorch 根据元素总数自动计算行数，最多只能有一个维度写 -1。
    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True)
    advantages = ((grouped - group_mean) / (group_std + advantage_eps)).reshape(-1) #重新变成一个一维张量
    return advantages, {
        "reward_mean": raw_rewards.mean().item(),
        "group_reward_std": group_std.mean().item(),
    }


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if importance_reweighting_method != "none":
        raise NotImplementedError("Section 4 implements on-policy GRPO only.")
    if raw_rewards_or_advantages.numel() != policy_log_probs.shape[0]:
        raise ValueError("One reward or advantage is required per response.")
    advantages = raw_rewards_or_advantages.reshape(-1, 1) #把advantages展开成二维的tensor 
    return -advantages * policy_log_probs, {} #矩阵乘法


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    if loss_normalization != "sequence":
        raise NotImplementedError("Section 4 implements sequence normalization only.")
    if per_token_policy_gradient_loss.shape != mask.shape:
        raise ValueError("Loss and response mask must have the same shape.")
    token_count = mask.sum(dim=1).clamp_min(1)
    sequence_loss = (per_token_policy_gradient_loss * mask).sum(dim=1) / token_count
    return sequence_loss.mean()


def grpo_train_step(
    model: torch.nn.Module, #输入模型
    tokenizer: PreTrainedTokenizerBase, 
    optimizer: torch.optim.Optimizer,# 优化器
    gradient_accumulation_steps: int, #梯度积累的个数
    max_grad_norm: float | None, 
    reward_fn: Callable[[str, str], dict[str, float]], #woc我知道了 这个是一个函数 callable 对应输入一个str str得到一个对应评分的字典
    repeated_prompts: list[str], #重复的提示词 为什么会重复因为希望提示词和回答一一对应
    rollout_responses: list[str], #同一个问题生成的不同回答
    repeated_ground_truths: list[str], #重复的真正答案
    group_size: int, #同一道题生成了多少条回答，用于组内比较
    baseline: Literal["mean", "none"] = "mean", #选择baseline
    advantage_eps: float = 1e-6, #
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    batch_size = len(rollout_responses) #本次的训练批次数量
    if len(repeated_prompts) != batch_size or len(repeated_ground_truths) != batch_size:
        raise ValueError("Prompts, responses, and ground truths must have the same length.")
    if not 1 <= gradient_accumulation_steps <= batch_size:
        raise ValueError("Gradient accumulation steps must fit within the batch.")
    if importance_reweighting_method != "none" or loss_normalization != "sequence":
        raise NotImplementedError("Section 4 implements standard on-policy GRPO only.")



    raw_rewards, reward_metadata = compute_rollout_rewards( #得到了得分的张量
        reward_fn, rollout_responses, repeated_ground_truths
    )   
    advantages, advantage_metadata = compute_group_normalized_rewards( #根据reward生成对应的 advantages这个是[问题，不同答案的得分]
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    tokens = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer) #按照提示词生成 label mask question的tensor [question，question_num*group_size]
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    response_token_count = 0

    for microbatch_index in range(gradient_accumulation_steps):
        start = microbatch_index * batch_size // gradient_accumulation_steps #这是一个小size 的开端
        end = (microbatch_index + 1) * batch_size // gradient_accumulation_steps
        input_ids = tokens["input_ids"][start:end].to(device) #这个是一个什么
        labels = tokens["labels"][start:end].to(device)
        response_mask = tokens["response_mask"][start:end].to(device)
        scored = get_response_log_probs(model, input_ids, labels, return_token_entropy=True) #
        per_token_loss, _ = compute_policy_gradient_loss(
            advantages[start:end].to(device), scored["log_probs"] #advantage对一个回答只有一个
        )
        microbatch_loss = aggregate_loss_across_microbatch(per_token_loss, response_mask)
        scaled_loss = microbatch_loss * ((end - start) / batch_size)
        scaled_loss.backward()  #
        total_loss += scaled_loss.detach()
        entropy_sum += (scored["token_entropy"].detach() * response_mask).sum()
        response_token_count += response_mask.sum().item()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm if max_grad_norm is not None else float("inf")
    )
    optimizer.step() 
    optimizer.zero_grad(set_to_none=True)
    metadata: dict[str, Tensor | float] = {
        "loss": total_loss.item(),
        "gradient_norm": gradient_norm.item(),
        "token_entropy": (entropy_sum / max(response_token_count, 1)).item(),
        **reward_metadata,
        **advantage_metadata,
    }
    return total_loss, metadata
