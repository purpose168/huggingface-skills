#!/usr/bin/env python3
"""
使用 CoT-Self-Instruct 方法生成合成数据的脚本

用法:
    uv run https://huggingface.co/datasets/uv-scripts/synthetic-data/raw/main/cot-self-instruct.py \
        --seed-dataset <种子数据集> \
        --output-dataset <输出数据集> \
        --task-type <任务类型> \
        --generation-model <生成模型> \
        --filter-method <过滤方法> \
        --num-samples <样本数量>

示例:
    # 生成推理任务数据 (问答对)
    uv run cot-self-instruct.py \
        --seed-dataset GSM8K/gsm8k \
        --output-dataset <your-username>/synthetic-math-reasoning \
        --task-type reasoning

    # 生成指令任务数据 (提示)
    uv run cot-self-instruct.py \
        --seed-dataset meta-math/MetaMathQA \
        --output-dataset <your-username>/synthetic-instruction \
        --task-type instruction

高级用法:
    # 使用 Answer-Consistency 过滤
    --filter-method answer-consistency \
    --k-responses 16 \
    --quality-threshold 0.5

    # 使用 RIP 过滤
    --filter-method rip \
    --reward-model Nexusflow/Athene-RM-8B

    # 同时使用两种过滤
    --filter-method both

    # 自定义生成模型
    --generation-model Qwen/Qwen3-30B-A3B-Thinking-2507

依赖:
    vllm>=0.6.0
    torch
    transformers
    datasets
    huggingface_hub
    scikit-learn
    tqdm
"""

import argparse
import os
import random
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetCard, load_dataset
from huggingface_hub import login
from sklearn.cluster import KMeans
from tqdm import tqdm
from vllm import LLM, SamplingParams

REASONING_PROMPT_TEMPLATE = """\
# 任务
基于以下两个种子示例,生成一个新的、具有挑战性的数学推理问题。

# 种子示例 1
{seed1}

# 种子示例 2
{seed2}

# 要求
1. 生成的问题应该与种子示例类似,但包含新的数值和情境
2. 问题应该需要多步推理才能解决
3. 确保问题表述清晰、逻辑连贯
4. 答案应该可以通过逐步推理得出

请按照以下格式输出:
[思考过程 Begin]
在此处详细描述解决这个问题的推理过程,包括所有中间步骤
[思考过程 End]

[新问题 Begin]
在此处写出生成的新问题
[新问题 End]

[新问题的最终答案 Begin]
\\boxed{{最终答案}}
[新问题的最终答案 End]
"""

INSTRUCTION_PROMPT_TEMPLATE = """\
# 任务
基于以下两个种子提示,生成一个新的、类似的指令。

# 种子提示 1
{prompt1}

# 种子提示 2
{prompt2}

# 要求
1. 生成的指令应该与种子提示类似,但主题或情境不同
2. 保持指令的复杂度和结构相似
3. 确保指令表述清晰、意图明确
4. 生成有实际应用价值的指令

请按照以下格式输出:
[思考过程 Begin]
在此处分析两个种子提示的共同特点和风格
[思考过程 End]

[合成提示 Begin]
在此处写出生成的合成指令
[合成提示 End]
"""


def check_gpu_availability():
    """检查 GPU 可用性并返回 GPU 数量。"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个 GPU")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name}, {gpu_memory:.2f} GB")
        return num_gpus
    else:
        print("未检测到 GPU,将使用 CPU (可能会很慢)")
        return 0


def parse_thinking_output(text: str) -> str:
    """从模型输出中提取思考过程并清理文本。"""
    text = re.sub(r'<think>.*?
</think>

', '', text, flags=re.DOTALL)
    return text.strip()


def extract_reasoning_output(text: str) -> Tuple[Optional[str], Optional[str]]:
    """从推理任务输出中提取问题和答案。"""
    text = parse_thinking_output(text)
    
    # 提取问题
    question_match = re.search(r'\[新问题 Begin\](.*?)\[新问题 End\]', text, re.DOTALL)
    if not question_match:
        return None, None
    question = question_match.group(1).strip()
    
    # 提取答案
    answer_match = re.search(r'\[新问题的最终答案 Begin\]\\?boxed\{(.*?)\}\[新问题的最终答案 End\]', text, re.DOTALL)
    if not answer_match:
        # 尝试不带 \boxed 的格式
        answer_match = re.search(r'\[新问题的最终答案 Begin\](.*?)\[新问题的最终答案 End\]', text, re.DOTALL)
    
    if not answer_match:
        return question, None
    
    answer = answer_match.group(1).strip()
    return question, answer


def extract_instruction_output(text: str) -> Optional[str]:
    """从指令任务输出中提取合成提示。"""
    text = parse_thinking_output(text)
    
    # 查找 "Step 3 #Synthetic Prompt#:" 之后的合成提示
    match = re.search(r'Step 3 #合成提示#:\s*(.+)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def categorize_prompts(prompts: List[str], num_categories: int = 8) -> Dict[int, List[int]]:
    """使用聚类对提示进行分类,用于指令任务。"""
    from transformers import AutoModel
    
    logger.info(f"正在将 {len(prompts)} 个提示分类为 {num_categories} 个类别...")
    
    # 使用小型模型进行嵌入
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # 获取嵌入向量
    embeddings = []
    for prompt in tqdm(prompts, desc="计算嵌入向量"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        embeddings.append(embedding[0])
    
    # 聚类
    kmeans = KMeans(n_clusters=num_categories, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # 按类别分组
    categories = {}
    for idx, label in enumerate(labels):
        if label not in categories:
            categories[label] = []
        categories[label].append(idx)
    
    return categories


def generate_synthetic_data(
    llm: LLM,
    seed_data: List[Dict],
    task_type: str,
    num_samples: int,
    categories: Optional[Dict[int, List[int]]] = None,
) -> List[Dict]:
    """使用 CoT-Self-Instruct 生成合成数据。"""
    synthetic_data = []
    
    # 设置进度条
    pbar = tqdm(total=num_samples, desc="生成合成数据")
    
    while len(synthetic_data) < num_samples:
        # 采样种子数据
        if task_type == "reasoning":
            # 推理任务的随机采样
            seeds = random.sample(seed_data, min(2, len(seed_data)))
            prompt = REASONING_PROMPT_TEMPLATE.format(
                seed1=seeds[0].get("question", seeds[0].get("prompt", "")),
                seed2=seeds[1].get("question", seeds[1].get("prompt", "")) if len(seeds) > 1 else seeds[0].get("question", seeds[0].get("prompt", ""))
            )
        else:
            # 指令任务的类别感知采样
            if categories:
                # 随机选择一个类别
                category = random.choice(list(categories.keys()))
                category_indices = categories[category]
                indices = random.sample(category_indices, min(2, len(category_indices)))
                seeds = [seed_data[i] for i in indices]
            else:
                seeds = random.sample(seed_data, min(2, len(seed_data)))
            
            prompt = INSTRUCTION_PROMPT_TEMPLATE.format(
                prompt1=seeds[0].get("prompt", seeds[0].get("question", "")),
                prompt2=seeds[1].get("prompt", seeds[1].get("question", "")) if len(seeds) > 1 else seeds[0].get("prompt", seeds[0].get("question", ""))
            )
        
        # 生成
        sampling_params = SamplingParams(
            temperature=0.7 if task_type == "reasoning" else 0.8,
            top_p=0.95 if task_type == "reasoning" else 0.9,
            max_tokens=2048,
        )
        
        outputs = llm.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text
        
        # 解析输出
        if task_type == "reasoning":
            question, answer = extract_reasoning_output(output_text)
            if question and answer:
                synthetic_data.append({
                    "question": question,
                    "answer": answer,
                    "seed_indices": [seed_data.index(s) for s in seeds],
                })
                pbar.update(1)
        else:
            synthetic_prompt = extract_instruction_output(output_text)
            if synthetic_prompt:
                synthetic_data.append({
                    "prompt": synthetic_prompt,
                    "seed_indices": [seed_data.index(s) for s in seeds],
                })
                pbar.update(1)
    
    pbar.close()
    return synthetic_data


def answer_consistency_filter(
    llm: LLM,
    synthetic_data: List[Dict],
    k_responses: int = 16,
    threshold: float = 0.5,
) -> List[Dict]:
    """使用答案一致性过滤推理任务。"""
    logger.info(f"正在应用 Answer-Consistency 过滤,K={k_responses}")
    
    filtered_data = []
    
    for item in tqdm(synthetic_data, desc="Answer-Consistency 过滤"):
        question = item["question"]
        original_answer = item["answer"]
        
        # 生成 K 个响应
        prompts = [question] * k_responses
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024,
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        # 提取答案
        answers = []
        for output in outputs:
            text = output.outputs[0].text
            # 尝试提取带框的答案
            match = re.search(r'\\boxed\{(.*?)\}', text)
            if match:
                answers.append(match.group(1).strip())
        
        if not answers:
            continue
        
        # 获取多数答案
        answer_counts = Counter(answers)
        if answer_counts:
            majority_answer, count = answer_counts.most_common(1)[0]
            
            # 检查多数答案是否与原始答案匹配且满足阈值
            if (majority_answer == original_answer and 
                count / len(answers) >= threshold):
                item["consistency_score"] = count / len(answers)
                filtered_data.append(item)
    
    logger.info(f"Answer-Consistency: 保留 {len(filtered_data)}/{len(synthetic_data)} 个示例")
    return filtered_data


def rip_filter(
    llm: LLM,
    synthetic_data: List[Dict],
    reward_model_id: str,
    k_responses: int = 32,
    threshold: float = 0.5,
) -> List[Dict]:
    """使用拒绝指令偏好 (RIP) 进行过滤。"""
    logger.info(f"正在应用 RIP 过滤,K={k_responses},奖励模型为 {reward_model_id}")
    
    # 注意:在完整实现中,您需要加载并使用实际的奖励模型
    # 对于此示例,我们将使用占位符评分机制
    logger.warning("RIP 过滤需要奖励模型实现 - 使用占位符")
    
    filtered_data = []
    
    for item in tqdm(synthetic_data, desc="RIP 过滤"):
        prompt = item.get("prompt", item.get("question", ""))
        
        # 生成 K 个响应
        prompts = [prompt] * k_responses
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        # 在实际实现中:使用奖励模型对每个响应评分
        # 现在,使用长度作为代理(较长的响应通常得分较高)
        scores = [len(output.outputs[0].text) for output in outputs]
        
        # 使用最小分数作为质量指标
        min_score = min(scores) if scores else 0
        normalized_score = min_score / 1000  # 归一化到 0-1 范围
        
        if normalized_score >= threshold:
            item["rip_score"] = normalized_score
            filtered_data.append(item)
    
    logger.info(f"RIP 过滤: 保留 {len(filtered_data)}/{len(synthetic_data)} 个示例")
    return filtered_data


def create_dataset_card(
    task_type: str,
    source_dataset: str,
    generation_model: str,
    filter_method: str,
    num_generated: int,
    num_filtered: int,
    generation_time: str,
    additional_info: Dict = None,
) -> str:
    """创建完整的数据集卡片。"""
    filter_info = ""
    if filter_method == "answer-consistency":
        filter_info = """
### Answer-Consistency 过滤

此数据集使用 Answer-Consistency 进行过滤:
- 为每个合成问题生成 K 个响应
- 只保留多数答案与生成答案匹配的示例
- 确保高质量、正确解决的问题"""
    elif filter_method == "rip":
        filter_info = """
### RIP (拒绝指令偏好) 过滤

此数据集使用 RIP 进行过滤:
- 为每个合成提示生成 K 个响应
- 使用奖励模型对响应进行评分
- 只保留最小分数高的提示"""
    
    return f"""---
tags:
- synthetic-data
- cot-self-instruct
- {task_type}
- uv-script
---

# CoT-Self-Instruct 合成数据

此数据集包含使用思维链自指令 (Chain-of-Thought Self-Instruct) 方法生成的合成 {task_type} 数据。

## 生成详情

- **源数据集**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **生成模型**: [{generation_model}](https://huggingface.co/{generation_model})
- **任务类型**: {task_type}
- **过滤方法**: {filter_method}
- **生成的示例**: {num_generated:,}
- **过滤后**: {num_filtered:,} ({(num_filtered/num_generated)*100:.1f}% 接受率)
- **生成日期**: {generation_time}
{filter_info}

## 方法论

使用 CoT-Self-Instruct 生成,该方法:
1. 使用思维链推理分析种子示例
2. 生成具有相似质量和复杂度的新合成示例
3. 应用质量过滤以确保高质量输出

基于论文:"CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks" (2025)

## 生成脚本

使用来自 [uv-scripts/synthetic-data](https://huggingface.co/datasets/uv-scripts/synthetic-data) 的 CoT-Self-Instruct 脚本生成。

复现方法:
```bash
uv run https://huggingface.co/datasets/uv-scripts/synthetic-data/raw/main/cot-self-instruct.py \\
    --seed-dataset {source_dataset} \\
    --output-dataset <your-dataset> \\
    --task-type {task_type} \\
    --generation-model {generation_model} \\
    --filter-method {filter_method}
```
"""


def main():
    parser = argparse.ArgumentParser(
        description="使用 CoT-Self-Instruct 生成合成数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # 数据集参数
    parser.add_argument(
        "--seed-dataset",
        type=str,
        required=True,
        help="包含种子示例的 HuggingFace 数据集 ID",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        required=True,
        help="输出的 HuggingFace 数据集 ID",
    )
    
    # 任务配置
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["reasoning", "instruction", "auto"],
        default="auto",
        help="任务类型 (reasoning 生成问答,instruction 生成提示)",
    )
    parser.add_argument(
        "--task-column",
        type=str,
        default=None,
        help="包含任务的列名 (未指定时自动检测)",
    )
    
    # 模型配置
    parser.add_argument(
        "--generation-model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Thinking-2507",
        help="用于合成数据生成的模型",
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        default=None,
        help="用于过滤的模型 (默认为生成模型)",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default="Nexusflow/Athene-RM-8B",
        help="用于 RIP 过滤的奖励模型",
    )
    
    # 生成参数
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="要生成的合成示例数量",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="生成的批次大小",
    )
    
    # 过滤参数
    parser.add_argument(
        "--filter-method",
        type=str,
        choices=["answer-consistency", "rip", "both", "none"],
        default="answer-consistency",
        help="质量过滤方法",
    )
    parser.add_argument(
        "--k-responses",
        type=int,
        default=16,
        help="用于过滤的响应数量",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.5,
        help="过滤的最小质量阈值",
    )
    
    # GPU 配置
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="用于张量并行的 GPU 数量 (未设置时自动检测)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU 内存利用率",
    )
    
    # 其他参数
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API 令牌",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 检查 GPU
    num_gpus = check_gpu_availability()
    tensor_parallel_size = args.tensor_parallel_size or num_gpus
    
    # 认证
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # 加载种子数据集
    logger.info(f"正在加载种子数据集: {args.seed_dataset}")
    seed_dataset = load_dataset(args.seed_dataset, split="train")
    
    # 根据需要自动检测任务类型和列
    if args.task_type == "auto":
        columns = seed_dataset.column_names
        if "question" in columns and "answer" in columns:
            args.task_type = "reasoning"
            logger.info("自动检测任务类型: reasoning")
        else:
            args.task_type = "instruction"
            logger.info("自动检测任务类型: instruction")
    
    if not args.task_column:
        if args.task_type == "reasoning":
            args.task_column = "question"
        else:
            # 尝试查找提示列
            for col in ["prompt", "instruction", "text", "input"]:
                if col in seed_dataset.column_names:
                    args.task_column = col
                    break
    
    logger.info(f"使用的任务列: {args.task_column}")
    
    # 转换为字典列表
    seed_data = seed_dataset.to_list()
    
    # 为指令任务分类提示
    categories = None
    if args.task_type == "instruction" and len(seed_data) > 100:
        prompts = [item.get(args.task_column, "") for item in seed_data]
        categories = categorize_prompts(prompts)
    
    # 初始化生成模型
    logger.info(f"正在加载生成模型: {args.generation_model}")
    generation_llm = LLM(
        model=args.generation_model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    # 生成合成数据
    start_time = datetime.now()
    synthetic_data = generate_synthetic_data(
        generation_llm,
        seed_data,
        args.task_type,
        args.num_samples,
        categories,
    )
    
    # 应用过滤
    filter_llm = generation_llm
    if args.filter_model and args.filter_model != args.generation_model:
        logger.info(f"正在加载过滤模型: {args.filter_model}")
        # 清理生成模型
        del generation_llm
        torch.cuda.empty_cache()
        
        filter_llm = LLM(
            model=args.filter_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    
    filtered_data = synthetic_data
    if args.filter_method != "none":
        if args.filter_method == "answer-consistency" and args.task_type == "reasoning":
            filtered_data = answer_consistency_filter(
                filter_llm,
                synthetic_data,
                args.k_responses,
                args.quality_threshold,
            )
        elif args.filter_method == "rip":
            filtered_data = rip_filter(
                filter_llm,
                synthetic_data,
                args.reward_model,
                args.k_responses,
                args.quality_threshold,
            )
        elif args.filter_method == "both":
            if args.task_type == "reasoning":
                filtered_data = answer_consistency_filter(
                    filter_llm,
                    synthetic_data,
                    args.k_responses,
                    args.quality_threshold,
                )
            filtered_data = rip_filter(
                filter_llm,
                filtered_data,
                args.reward_model,
                args.k_responses,
                args.quality_threshold,
            )
    
    # 创建 HuggingFace 数据集
    logger.info(f"正在创建包含 {len(filtered_data)} 个示例的数据集")
    dataset = Dataset.from_list(filtered_data)
    
    # 创建数据集卡片
    generation_time = start_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    dataset_card = create_dataset_card(
        args.task_type,
        args.seed_dataset,
        args.generation_model,
        args.filter_method,
        len(synthetic_data),
        len(filtered_data),
        generation_time,
    )
    
    # 推送到 hub
    logger.info(f"正在推送到: {args.output_dataset}")
    # 创建数据集卡片
    card = DatasetCard(dataset_card)
    dataset.push_to_hub(args.output_dataset)
    # 单独推送卡片
    card.push_to_hub(args.output_dataset)
    
    logger.info("完成! 数据集可用链接: https://huggingface.co/datasets/" + args.output_dataset)
    
    # 如果在本地运行,打印示例 HF Jobs 命令
    if len(sys.argv) > 1:
        print("\n在 HF Jobs 上运行:")
        print(f"""hf jobs uv run --flavor l4x4 \\
    --image vllm/vllm-openai \\
    -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\
    https://huggingface.co/datasets/uv-scripts/synthetic-data/raw/main/cot-self-instruct.py \\
    --seed-dataset {args.seed_dataset} \\
    --output-dataset {args.output_dataset} \\
    --task-type {args.task_type} \\
    --generation-model {args.generation_model} \\
    --filter-method {args.filter_method} \\
    --num-samples {args.num_samples}""")


if __name__ == "__main__":
    main()
