# CS336 作业 5：第 4 节 GRPO 中文说明

本文说明仓库中已实现的第 4 节 **on-policy Group Relative Policy Optimization（GRPO）** 代码。任务定义见[作业 PDF](cs336_spring2026_assignment5_alignment.pdf)。这里的实现覆盖第 4.2 节的组件和训练步骤，以及第 4.3 节要求的训练脚本；第 5、6 节的算法变体不在本文范围内。

## 1. 任务与代码位置

| 对应任务 | 代码位置 | 完成的工作 |
| --- | --- | --- |
| `tokenize_prompt_and_output` | [`cs336_alignment/grpo.py`](cs336_alignment/grpo.py) 中的同名函数 | 分别编码提示和回答，拼接、填充并构造只标记回答 token 的 `response_mask`。 |
| `get_response_log_probs` | [`cs336_alignment/grpo.py`](cs336_alignment/grpo.py) 中的同名函数 | 从语言模型 logits 提取实际回答 token 的对数概率，并可选计算每个位置的预测熵。 |
| `compute_rollout_rewards` | [`cs336_alignment/grpo.py`](cs336_alignment/grpo.py) 中的同名函数 | 用评分函数给每条生成结果打分，统计总奖励、格式奖励和答案奖励。 |
| `compute_group_normalized_rewards_grpo` | [`cs336_alignment/grpo.py`](cs336_alignment/grpo.py) 中的 `compute_group_normalized_rewards` | 对同一问题的多条回答按组减去平均奖励，再除以组标准差，得到优势值。 |
| `compute_policy_gradient_loss_on_policy` | [`cs336_alignment/grpo.py`](cs336_alignment/grpo.py) 中的 `compute_policy_gradient_loss` | 计算每个回答 token 的 `-优势值 × 对数概率`。 |
| `aggregate_loss_across_microbatch_sequence` | [`cs336_alignment/grpo.py`](cs336_alignment/grpo.py) 中的 `aggregate_loss_across_microbatch` | 按 `response_mask` 先对每条回答的 token 损失求平均，再对批次求平均。 |
| `grpo_train_step_standard_on_policy` | [`cs336_alignment/grpo.py`](cs336_alignment/grpo.py) 中的 `grpo_train_step` | 把奖励、优势值、模型前向计算和损失组合起来，完成梯度累积、裁剪与一次参数更新。 |
| `grpo_experiments_standard_on_policy` 的脚本部分 | [`scripts/grpo.py`](scripts/grpo.py) | 加载 GSM8K，调用 vLLM 采样，循环训练、验证、记录指标并保存模型。 |

测试入口在 [`tests/adapters.py`](tests/adapters.py)：其中 `run_*` 函数把作业测试连接到 `cs336_alignment/grpo.py` 的实现。对应测试在 [`tests/test_grpo.py`](tests/test_grpo.py)。

训练脚本还使用以下现有组件：[`checkpoint.py`](cs336_alignment/checkpoint.py) 加载模型和 tokenizer；[`vllm_utils.py`](cs336_alignment/vllm_utils.py) 启动推理服务、生成回答和同步权重；[`drgrpo_grader.py`](cs336_alignment/drgrpo_grader.py) 评判数学答案；[`prompts/`](cs336_alignment/prompts/) 存放提示模板。GSM8K 数据已位于 [`data/gsm8k/`](data/gsm8k/)。

## 2. 代码如何完成 GRPO

### 2.1 从文本到可训练的 token

`tokenize_prompt_and_output` 对提示和回答分别调用 tokenizer，且不添加特殊 token，然后直接拼接。填充后的序列错开一位形成 `input_ids` 和 `labels`：模型读入前面的 token，预测下一个 token。`response_mask` 与 `labels` 对齐，仅回答部分为 `True`，因此提示和填充位置不会进入训练损失。

`get_response_log_probs` 对模型 logits 做 `log_softmax`，再按 `labels` 取出实际生成 token 的对数概率。开启 `return_token_entropy` 时，还会对每个位置的完整词表分布计算熵，供训练日志观察模型的不确定性。

### 2.2 从回答奖励到 GRPO 优势值

`compute_rollout_rewards` 把每条回答及其标准答案传给奖励函数，得到原始奖励和批次统计。`scripts/grpo.py` 会从 GSM8K 的 `answer` 字段中按 `####` 提取最终答案，并根据提示模板选择 `r1_zero_reward_fn` 或 `question_only_reward_fn`。

同一问题连续生成 `group_size` 条回答。`compute_group_normalized_rewards` 对每组计算平均奖励和标准差，按下式得到优势值：

```text
优势值 = (该回答奖励 - 该组平均奖励) / (该组标准差 + advantage_eps)
```

代码使用 PyTorch 默认的 `torch.std`，并在分母加入默认值 `1e-6`，避免一组回答奖励相同时除以零。高于组平均奖励的回答得到正优势值，低于平均奖励的回答得到负优势值。

### 2.3 从 token 损失到参数更新

`compute_policy_gradient_loss` 对每个 token 计算 `-优势值 × log P(token)`；负号使优化器的梯度下降对应提高高奖励回答的概率。`aggregate_loss_across_microbatch` 仅保留 `response_mask=True` 的位置，先按每条回答的有效 token 数取平均，再对回答取平均，这就是第 4 节使用的序列长度归一化。

`grpo_train_step` 将一批回答拆成多个 microbatch。每个 microbatch 的损失按照它在整批中的回答数占比缩放，然后反向传播并累积梯度；全部反向传播完成后，记录裁剪前梯度范数、按 `max_grad_norm` 裁剪梯度、调用一次 `optimizer.step()`，最后清空梯度。返回的日志包含损失、梯度范数、回答 token 熵及训练奖励。

### 2.4 完整训练循环

[`scripts/grpo.py`](scripts/grpo.py) 的主要流程是：

1. 读取训练/验证 JSONL、提示模板与模型；将标准答案从 `####` 后提取出来。
2. 把训练模型放在 GPU 0，vLLM 推理服务放在 GPU 1；先同步模型权重。
3. 每轮抽取问题，每题重复 `group_size` 次，由 vLLM 生成回答，再调用 `grpo_train_step` 更新一次模型。
4. 定期同步新权重并在验证集上评估；记录总奖励、格式奖励、答案奖励和平均回答长度。
5. 写入 `config.json`、`metrics.jsonl`、`rollouts.jsonl`，最后保存 `final_model/`。传入 `--wandb-project` 时，还会把指标同步到 Weights & Biases。

默认设置与作业第 4 节建议值一致：`rollout_batch_size=256`、`group_size=8`、`gradient_accumulation_steps=32`、`learning_rate=1e-5`、`num_rollout_steps=200`、`temperature=1.0`、`top_p=1.0`、`max_tokens=512`。默认提示为 `r1_zero.prompt`；该提示会在生成 `</answer>` 后停止，并保留停止字符串以供评分。

## 3. 环境依赖与安装

### 3.1 运行平台

项目的 [`pyproject.toml`](pyproject.toml) 要求 **Python 3.12**（`>=3.12,<3.13`），并用 `uv.lock` 固定依赖。完整 GPU 训练建议在 **Linux x86_64** 上运行，需要两张可用的 NVIDIA GPU：GPU 0 用于训练，GPU 1 用于 vLLM 推理及权重同步。Linux GPU 依赖使用 CUDA 12.9 版 PyTorch，因此驱动也需兼容该运行环境。FlashAttention wheel 针对 Linux x86_64、Python 3.12 和 CUDA 12 构建；当前项目配置不面向原生 Windows GPU 训练。

先在 Linux 机器上确认驱动和两张 GPU 可见：

```sh
nvidia-smi
```

### 3.2 安装 uv、Python 和项目依赖

按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 `uv`。例如在 Linux 上：

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装完成后按安装器提示更新 `PATH` 或重新打开终端，再运行 `uv --version` 确认安装成功。

进入仓库目录，安装 Python 3.12。`.python-version` 已指定版本，`uv` 会据此创建项目虚拟环境：

```sh
cd assignment5-alignment
uv python install 3.12
```

在项目声明支持的 Linux x86_64 或 macOS arm64 环境中，只运行第 4 节的 CPU 单元测试时，安装基础依赖即可：

```sh
uv sync
```

运行双 GPU 训练时，安装 `gpu` 额外依赖。项目原有说明建议先安装其余依赖，再安装 FlashAttention：

```sh
uv sync --extra gpu --no-install-package flash-attn
uv sync --extra gpu
uv run --extra gpu python -c "import torch; print(torch.cuda.device_count())"
```

最后一条命令应输出至少 `2`。基础依赖包含 PyTorch、Transformers、pytest 和数学评分相关库；`gpu` 额外依赖包含 vLLM、FlashAttention 和 Wandb。模型与 tokenizer 需要远程下载时，代码通过 `HF_ENDPOINT=https://hf-mirror.com` 使用 HF-Mirror；GSM8K 数据已在仓库中。Python 包仍按 `pyproject.toml` 中的 PyPI、PyTorch 和 GitHub 源安装，HF-Mirror 配置只作用于 Hugging Face 模型下载。

### 3.3 验证第 4 节实现

下面只选择第 4 节的测试。`tests/test_grpo.py` 还包含第 5、6 节的变体测试，直接运行整个文件会测试超出本文范围的功能。

```sh
uv run --extra gpu pytest tests/test_grpo.py -k 'tokenize_prompt_and_output or get_response_log_probs or compute_rollout_rewards or compute_group_normalized_rewards_grpo or compute_policy_gradient_loss_on_policy or aggregate_loss_across_microbatch_sequence or grpo_train_step_standard_on_policy'
```

如果只安装了基础依赖，将命令中的 `--extra gpu` 去掉；若已安装 GPU 依赖，保留它可避免 `uv run` 重新同步为基础环境。

### 3.4 启动训练

在双 GPU 的 Linux 环境中运行：

```sh
uv run --extra gpu python scripts/grpo.py --output-dir runs/grpo-seed-0 --seed 0
```

脚本会自动选择空闲端口启动 vLLM，并同时检查 `/health` 和权重同步接口。
如果日志显示 `/get_world_size` 返回 404，通常是连到了其他服务，或目标 vLLM
没有开启开发接口；仅 `/health` 成功并不能证明目标是 vLLM。
手动连接已有服务时，需用 `VLLMServer(launch_server=False, port=实际端口)`，并确保
该服务启动时设置了 `VLLM_SERVER_DEV_MODE=1` 和
`--weight-transfer-config '{"backend": "nccl"}'`。
默认 vLLM 显存比例为 `0.8`，给 NCCL 权重同步留出空间；可通过
`--vllm-gpu-memory-utilization` 调整。权重传输使用 64 MiB 的分包目标，避免
vLLM 默认的 GiB 级传输缓冲与推理缓存一起耗尽显存。

可以用 `--model-name-or-path` 指定 HF 模型 ID 或本地模型目录，用 `--prompt-file`、`--train-file`、`--val-file` 替换输入。先查看全部可调参数：

```sh
uv run --extra gpu python scripts/grpo.py --help
```

如需 Wandb 记录，添加 `--wandb-project 项目名`。不同随机种子应使用不同的 `--output-dir`，避免日志相互追加。当前仓库尚未提供完整双 GPU 运行产生的曲线或最终准确率；文档中的命令用于在具备相应硬件的环境中完成实验。
