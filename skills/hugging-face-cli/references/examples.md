# HF CLI 常见工作流与示例

Hugging Face Hub 常见任务的实用示例。

## 目录
- [浏览 Hub](#浏览-hub)
- [模型工作流](#模型工作流)
- [数据集工作流](#数据集工作流)
- [Space 工作流](#space-工作流)
- [推理端点](#推理端点)
- [缓存管理](#缓存管理)
- [自动化模式](#自动化模式)

---

## 浏览 Hub

### 发现模型

```bash
# 查找流行的文本生成模型
hf models ls --filter "text-generation" --sort downloads --limit 10

# 搜索特定模型架构
hf models ls --search "MiniMax" --author MiniMaxAI

# 查找带有扩展信息的模型
hf models ls --search "MiniMax" --expand downloads,likes,pipeline_tag

# 获取模型的详细信息
hf models info MiniMaxAI/MiniMax-M2.1
hf models info MiniMaxAI/MiniMax-M2.1 --expand downloads,likes,tags,config
```

### 发现数据集

```bash
# 查找流行数据集
hf datasets ls --sort downloads --limit 10

# 按主题搜索数据集
hf datasets ls --search "finepdfs" --author HuggingFaceFW

# 获取数据集的详细信息
hf datasets info HuggingFaceFW/finepdfs
hf datasets info HuggingFaceFW/finepdfs --expand downloads,likes,description
```

### 发现 Spaces

```bash
# 列出热门 Spaces
hf spaces ls --limit 10

# 按 3D 建模 Spaces 过滤
hf spaces ls --filter "3d" --limit 10

# 按作者查找 Spaces
hf spaces ls --author enzostvs --limit 20

# 获取 Space 的详细信息
hf spaces info enzostvs/deepsite
hf spaces info enzostvs/deepsite --expand sdk,runtime,likes
```

---

## 模型工作流

### 下载模型用于本地推理

```bash
# 下载整个模型到缓存（推荐）
hf download meta-llama/Llama-3.2-1B-Instruct

# 仅下载特定文件
hf download meta-llama/Llama-3.2-1B-Instruct config.json tokenizer.json

# 仅下载 safetensors（跳过 pytorch .bin 文件）
hf download meta-llama/Llama-3.2-1B-Instruct --include "*.safetensors" --exclude "*.bin"

# 下载到特定目录用于部署
hf download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama
```

### 发布微调模型

```bash
# 创建仓库
hf repo create my-username/my-finetuned-model --private

# 上传模型文件
hf upload my-username/my-finetuned-model ./output . \
  --commit-message="Initial model upload after SFT training"

# 添加版本标签
hf repo tag create my-username/my-finetuned-model v1.0

# 列出标签以验证
hf repo tag list my-username/my-finetuned-model
```

### 下载特定模型版本

```bash
# 从特定分支下载
hf download stabilityai/stable-diffusion-xl-base-1.0 --revision fp16

# 从特定提交下载
hf download gpt2 --revision 11c5a3d5811f50298f278a704980280950aedb10

# 从 PR 下载
hf download bigcode/starcoder2 --revision refs/pr/42
```

---

## 数据集工作流

### 下载数据集

```bash
# 完整数据集到缓存
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# 特定拆分/文件
hf download HuggingFaceH4/ultrachat_200k data/train.parquet --repo-type dataset

# 到本地目录用于处理
hf download tatsu-lab/alpaca --repo-type dataset --local-dir ./data/alpaca
```

### 上传数据集

```bash
# 创建数据集仓库
hf repo create my-username/my-dataset --repo-type dataset --private

# 上传数据文件夹
hf upload my-username/my-dataset ./data . --repo-type dataset \
  --commit-message="Add training data"

# 上传结构化路径
hf upload my-username/my-dataset ./train_data /train --repo-type dataset
hf upload my-username/my-dataset ./test_data /test --repo-type dataset
```

### 贡献到现有数据集

```bash
# 创建包含新数据的 PR
hf upload community/shared-dataset ./my_contribution /contributed \
  --repo-type dataset --create-pr \
  --commit-message="Add 1000 new samples for domain X"
```

---

## Space 工作流

### 下载 Space 用于本地开发

```bash
hf download HuggingFaceH4/zephyr-chat --repo-type space --local-dir ./my-space
```

### 部署/更新 Space

```bash
# 创建 space
hf repo create my-username/my-app --repo-type space --space_sdk gradio

# 上传应用文件
hf upload my-username/my-app . . --repo-type space \
  --exclude="__pycache__/*" --exclude=".git/*" --exclude="*.pyc"

# 开发期间的持续部署（每 5 分钟上传一次）
hf upload my-username/my-app . . --repo-type space --every=5
```

### 同步本地更改

```bash
# 上传更改并从远程删除已移除的文件
hf upload my-username/my-app . . --repo-type space \
  --exclude="/logs/*" --exclude="*.tmp" \
  --delete="*" \
  --commit-message="Sync local with Hub"
```

---

## 推理端点

### 列出端点

```bash
hf endpoints ls
hf endpoints ls --namespace my-org
```

### 部署端点

```bash
hf endpoints deploy my-endpoint \
  --repo openai/gpt-oss-120b \
  --framework vllm \
  --accelerator gpu \
  --instance-size x4 \
  --instance-type nvidia-a10g \
  --region us-east-1 \
  --vendor aws
```

### 操作端点

```bash
hf endpoints describe my-endpoint
hf endpoints pause my-endpoint
hf endpoints resume my-endpoint
hf endpoints scale-to-zero my-endpoint
```

---

## 缓存管理

### 检查缓存使用情况

```bash
# 所有缓存仓库的概览
hf cache ls

# 包含版本
hf cache ls --revisions

# 自定义缓存位置
hf cache ls --cache-dir /path/to/custom/cache
```

### 清理磁盘空间

```bash
# 从缓存中移除特定仓库
hf cache rm model/gpt2

# 移除分离的版本
hf cache prune

# 非交互式模式（用于脚本）
hf cache rm model/gpt2 --yes
```

---

## 自动化模式

### 脚本化认证

```bash
# 用于 CI/CD 的非交互式登录
hf auth login --token $HF_TOKEN --add-to-git-credential

# 验证认证
hf auth whoami

# 列出存储的令牌
hf auth list
```

### 脚本的安静模式

```bash
# 获取缓存路径用于进一步处理
MODEL_PATH=$(hf download gpt2 --quiet)
echo "Model downloaded to: $MODEL_PATH"

# 获取上传 URL
UPLOAD_URL=$(hf upload my-model ./output . --quiet)
echo "Uploaded to: $UPLOAD_URL"
```

### 批量下载多个模型

```bash
#!/bin/bash
MODELS=(
  "meta-llama/Llama-3.2-1B-Instruct"
  "microsoft/phi-2"
  "google/gemma-2b"
)

for model in "${MODELS[@]}"; do
  echo "Downloading $model..."
  hf download "$model" --quiet
done
```

### CI/CD 模型发布

```bash
#!/bin/bash
# 模型发布的典型 CI 工作流

# 认证
hf auth login --token $HF_TOKEN

# 创建仓库（如果需要 - 如果存在相同设置则会成功）
hf repo create $ORG/$MODEL_NAME --private || true

# 上传模型工件
hf upload $ORG/$MODEL_NAME ./model_output . \
  --commit-message="Release v${VERSION}" \
  --commit-description="Training metrics: loss=${LOSS}, accuracy=${ACC}"

# 标记发布
hf repo tag create $ORG/$MODEL_NAME "v${VERSION}"
```

### 运行 GPU 训练作业

```bash
# 在 A100 上运行训练脚本
hf jobs run --flavor a100-large \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  --secrets HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_KEY \
  python train.py --epochs 10 --batch-size 32

# 监控作业
hf jobs ps
hf jobs logs <job_id>
```

### 计划数据管道

```bash
# 每天午夜运行数据处理
hf jobs scheduled run "0 0 * * *" python:3.12 \
  --secrets HF_TOKEN \
  python -c "
import huggingface_hub
# Your daily data pipeline code
"

# 列出计划作业
hf jobs scheduled ps
```

---

## 快速参考模式

| 任务 | 命令 |
|------|---------|
| 下载模型 | `hf download <repo_id>` |
| 下载到文件夹 | `hf download <repo_id> --local-dir ./path` |
| 上传文件夹 | `hf upload <repo_id> . .` |
| 创建模型仓库 | `hf repo create <name>` |
| 创建数据集仓库 | `hf repo create <name> --repo-type dataset` |
| 创建私有仓库 | `hf repo create <name> --private` |
| 创建 space | `hf repo create <name> --repo-type space --space_sdk gradio` |
| 标记发布 | `hf repo tag create <repo_id> v1.0` |
| 删除文件 | `hf repo-files delete <repo_id> <files>` |
| 列出模型 | `hf models ls` |
| 获取模型信息 | `hf models info <model_id>` |
| 列出数据集 | `hf datasets ls` |
| 获取数据集信息 | `hf datasets info <dataset_id>` |
| 列出 spaces | `hf spaces ls` |
| 获取 space 信息 | `hf spaces info <space_id>` |
| 检查缓存 | `hf cache ls` |
| 清理缓存 | `hf cache prune` |
| 在 GPU 上运行 | `hf jobs run --flavor a10g-small <image> <cmd>` |
| 获取环境信息 | `hf env` |
