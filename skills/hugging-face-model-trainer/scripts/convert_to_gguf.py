#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.36.0",
#     "peft>=0.7.0",
#     "torch>=2.0.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "sentencepiece>=0.1.99",
#     "protobuf>=3.20.0",
#     "numpy",
#     "gguf",
# ]
# ///

"""
GGUF 转换脚本 - 生产就绪版本

本脚本将 LoRA 微调模型转换为 GGUF 格式，以便在以下工具中使用：
- llama.cpp
- Ollama
- LM Studio
- 其他兼容 GGUF 的工具

前置条件（请先安装这些）：
- Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y build-essential cmake
- RHEL/CentOS: sudo yum groupinstall -y "Development Tools" && sudo yum install -y cmake
- macOS: xcode-select --install && brew install cmake

使用方法：
    设置环境变量：
    - ADAPTER_MODEL: 您的微调模型（例如："username/my-finetuned-model"）
    - BASE_MODEL: 用于微调的基础模型（例如："Qwen/Qwen2.5-0.5B"）
    - OUTPUT_REPO: 上传 GGUF 文件的位置（例如："username/my-model-gguf"）
    - HF_USERNAME: 您的 Hugging Face 用户名（可选，用于 README）

依赖项：所有必需的包已在上述 PEP 723 头部声明。
"""

# 导入必要的 Python 标准库和第三方库
import os  # 操作系统接口，用于环境变量和文件操作
import sys  # 系统相关的参数和函数
import torch  # PyTorch 深度学习框架
from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face Transformers 模型加载器
from peft import PeftModel  # PEFT（参数高效微调）库，用于加载 LoRA 适配器
from huggingface_hub import HfApi  # Hugging Face Hub API，用于上传模型
import subprocess  # 子进程管理，用于执行系统命令


def check_system_dependencies():
    """
    检查必需的系统包是否可用

    Returns:
        bool: 如果所有依赖都存在则返回 True，否则返回 False
    """
    print("🔍 正在检查系统依赖...")

    # 检查 git 是否已安装
    # subprocess.run 用于执行系统命令，capture_output=True 捕获输出
    if subprocess.run(["which", "git"], capture_output=True).returncode != 0:
        print("  ❌ git 未安装。请安装它：")
        print("     Ubuntu/Debian: sudo apt-get install git")
        print("     RHEL/CentOS: sudo yum install git")
        print("     macOS: brew install git")
        return False

    # 检查 make 或 cmake 是否已安装
    # returncode == 0 表示命令执行成功
    has_make = subprocess.run(["which", "make"], capture_output=True).returncode == 0
    has_cmake = subprocess.run(["which", "cmake"], capture_output=True).returncode == 0

    if not has_make and not has_cmake:
        print("  ❌ 未找到 make 或 cmake。请安装构建工具：")
        print("     Ubuntu/Debian: sudo apt-get install build-essential cmake")
        print("     RHEL/CentOS: sudo yum groupinstall 'Development Tools' && sudo yum install cmake")
        print("     macOS: xcode-select --install && brew install cmake")
        return False

    print("  ✅ 系统依赖已找到")
    return True


def run_command(cmd, description):
    """
    执行命令并进行错误处理

    Args:
        cmd (list): 要执行的命令及其参数列表
        description (str): 命令的描述信息

    Returns:
        bool: 命令执行成功返回 True，失败返回 False
    """
    print(f"   {description}...")
    try:
        # subprocess.run 执行命令，check=True 会在命令失败时抛出异常
        # capture_output=True 捕获标准输出和错误输出
        # text=True 将输出作为字符串而非字节返回
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        # 如果有标准输出，显示前 200 个字符
        if result.stdout:
            print(f"   {result.stdout[:200]}")  # 显示前 200 个字符
        return True
    except subprocess.CalledProcessError as e:
        # 处理命令执行失败的情况
        print(f"   ❌ 命令失败: {' '.join(cmd)}")
        if e.stdout:
            print(f"   标准输出: {e.stdout[:500]}")
        if e.stderr:
            print(f"   标准错误: {e.stderr[:500]}")
        return False
    except FileNotFoundError:
        # 处理命令未找到的情况
        print(f"   ❌ 未找到命令: {cmd[0]}")
        return False


# 打印脚本标题
print("🔄 GGUF 转换脚本")
print("=" * 60)

# 首先检查系统依赖
if not check_system_dependencies():
    print("\n❌ 请安装缺失的系统依赖后重试。")
    sys.exit(1)

# 从环境变量获取配置
# os.environ.get() 获取环境变量，第二个参数是默认值
ADAPTER_MODEL = os.environ.get("ADAPTER_MODEL", "evalstate/qwen-capybara-medium")  # LoRA 适配器模型
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B")  # 基础模型
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "evalstate/qwen-capybara-medium-gguf")  # 输出仓库
username = os.environ.get("HF_USERNAME", ADAPTER_MODEL.split('/')[0])  # Hugging Face 用户名

print(f"\n📦 配置信息:")
print(f"   基础模型: {BASE_MODEL}")
print(f"   适配器模型: {ADAPTER_MODEL}")
print(f"   输出仓库: {OUTPUT_REPO}")

# 步骤 1: 加载基础模型和适配器
print("\n🔧 步骤 1: 正在加载基础模型和 LoRA 适配器...")
print("   (这可能需要几分钟)")

try:
    # 加载基础因果语言模型
    # dtype=torch.float16 使用半精度浮点数以减少内存占用
    # device_map="auto" 自动将模型分配到可用的设备（CPU/GPU）
    # trust_remote_code=True 允许执行模型仓库中的自定义代码
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("   ✅ 基础模型已加载")
except Exception as e:
    print(f"   ❌ 加载基础模型失败: {e}")
    sys.exit(1)

try:
    # 加载并合并适配器
    print("   正在加载 LoRA 适配器...")
    # PeftModel.from_pretrained 加载 LoRA 适配器并应用到基础模型
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
    print("   ✅ 适配器已加载")

    print("   正在将适配器与基础模型合并...")
    # merge_and_unload() 将 LoRA 权重合并到基础模型中并卸载适配器
    merged_model = model.merge_and_unload()
    print("   ✅ 模型已合并！")
except Exception as e:
    print(f"   ❌ 合并模型失败: {e}")
    sys.exit(1)

try:
    # 加载分词器（tokenizer）
    # 分词器用于将文本转换为模型可理解的 token 序列
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL, trust_remote_code=True)
    print("   ✅ 分词器已加载")
except Exception as e:
    print(f"   ❌ 加载分词器失败: {e}")
    sys.exit(1)

# 步骤 2: 临时保存合并后的模型
print("\n💾 步骤 2: 正在保存合并后的模型...")
merged_dir = "/tmp/merged_model"  # 临时保存目录
try:
    # save_pretrained 保存模型权重和配置
    # safe_serialization=True 使用安全序列化格式（safetensors）
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"   ✅ 合并后的模型已保存到 {merged_dir}")
except Exception as e:
    print(f"   ❌ 保存合并后的模型失败: {e}")
    sys.exit(1)

# 步骤 3: 安装 llama.cpp 用于转换
print("\n📥 步骤 3: 正在设置 llama.cpp 用于 GGUF 转换...")

# 克隆 llama.cpp 仓库
if not run_command(
    ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"],
    "正在克隆 llama.cpp 仓库"
):
    print("   正在尝试备用克隆方法...")
    # 尝试浅克隆（只克隆最新版本，不包含历史记录）
    if not run_command(
        ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"],
        "正在克隆 llama.cpp（浅克隆）"
    ):
        sys.exit(1)

# 安装 Python 依赖
print("   正在安装 Python 依赖...")
if not run_command(
    ["pip", "install", "-r", "/tmp/llama.cpp/requirements.txt"],
    "正在安装 llama.cpp 依赖"
):
    print("   ⚠️  某些依赖可能已经安装")

if not run_command(
    ["pip", "install", "sentencepiece", "protobuf"],
    "正在安装分词器依赖"
):
    print("   ⚠️  分词器依赖可能已经安装")

# 步骤 4: 转换为 GGUF（FP16 格式）
print("\n🔄 步骤 4: 正在转换为 GGUF 格式（FP16）...")
gguf_output_dir = "/tmp/gguf_output"  # GGUF 输出目录
os.makedirs(gguf_output_dir, exist_ok=True)  # 创建输出目录，如果已存在则不报错

convert_script = "/tmp/llama.cpp/convert_hf_to_gguf.py"  # 转换脚本路径
model_name = ADAPTER_MODEL.split('/')[-1]  # 从模型路径中提取模型名称
gguf_file = f"{gguf_output_dir}/{model_name}-f16.gguf"  # 输出文件路径

print(f"   正在运行转换...")
if not run_command(
    [
        sys.executable, convert_script,  # 使用当前 Python 解释器执行转换脚本
        merged_dir,  # 输入模型目录
        "--outfile", gguf_file,  # 输出文件路径
        "--outtype", "f16"  # 输出类型为 FP16（半精度浮点）
    ],
    f"正在转换为 FP16"
):
    print("   ❌ 转换失败！")
    sys.exit(1)

print(f"   ✅ FP16 GGUF 已创建: {gguf_file}")

# 步骤 5: 量化为不同格式
print("\n⚙️  步骤 5: 正在创建量化版本...")

# 使用 CMake 构建量化工具（比 make 更可靠）
print("   正在使用 CMake 构建量化工具...")
os.makedirs("/tmp/llama.cpp/build", exist_ok=True)  # 创建构建目录

# 使用 CMake 配置
# -B 指定构建目录，-S 指定源代码目录
# -DGGML_CUDA=OFF 禁用 CUDA 支持（如果需要 GPU 加速可设置为 ON）
if not run_command(
    ["cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp",
     "-DGGML_CUDA=OFF"],
    "正在使用 CMake 配置"
):
    print("   ❌ CMake 配置失败")
    sys.exit(1)

# 只构建量化工具
# --target 指定构建目标，-j 4 使用 4 个并行任务
if not run_command(
    ["cmake", "--build", "/tmp/llama.cpp/build", "--target", "llama-quantize", "-j", "4"],
    "正在构建 llama-quantize"
):
    print("   ❌ 构建失败！")
    sys.exit(1)

print("   ✅ 量化工具已构建")

# 使用 CMake 构建输出路径
quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"  # 量化工具可执行文件

# 常用量化格式
# 每个元组包含：(量化类型, 描述)
quant_formats = [
    ("Q4_K_M", "4 位，中等质量（推荐）"),
    ("Q5_K_M", "5 位，更高质量"),
    ("Q8_0", "8 位，非常高质量"),
]

quantized_files = []  # 存储生成的量化文件列表
for quant_type, description in quant_formats:
    print(f"   正在创建 {quant_type} 量化（{description}）...")
    quant_file = f"{gguf_output_dir}/{model_name}-{quant_type.lower()}.gguf"  # 量化文件路径

    if not run_command(
        [quantize_bin, gguf_file, quant_file, quant_type],  # 执行量化命令
        f"正在量化为 {quant_type}"
    ):
        print(f"   ⚠️  由于错误跳过 {quant_type}")
        continue

    quantized_files.append((quant_file, quant_type))  # 添加到已生成文件列表

    # 获取文件大小
    size_mb = os.path.getsize(quant_file) / (1024 * 1024)  # 转换为 MB
    print(f"   ✅ {quant_type}: {size_mb:.1f} MB")

if not quantized_files:
    print("   ❌ 没有成功创建任何量化版本")
    sys.exit(1)

# 步骤 6: 上传到 Hub
print("\n☁️  步骤 6: 正在上传到 Hugging Face Hub...")
api = HfApi()  # 创建 Hugging Face API 实例

# 创建仓库
print(f"   正在创建仓库: {OUTPUT_REPO}")
try:
    # create_repo 创建新仓库，exist_ok=True 如果仓库已存在则不报错
    api.create_repo(repo_id=OUTPUT_REPO, repo_type="model", exist_ok=True)
    print("   ✅ 仓库已就绪")
except Exception as e:
    print(f"   ℹ️  仓库可能已存在: {e}")

# 上传 FP16 版本
print("   正在上传 FP16 GGUF...")
try:
    # upload_file 上传文件到仓库
    api.upload_file(
        path_or_fileobj=gguf_file,  # 本地文件路径
        path_in_repo=f"{model_name}-f16.gguf",  # 仓库中的文件名
        repo_id=OUTPUT_REPO,  # 仓库 ID
    )
    print("   ✅ FP16 已上传")
except Exception as e:
    print(f"   ❌ 上传失败: {e}")
    sys.exit(1)

# 上传量化版本
for quant_file, quant_type in quantized_files:
    print(f"   正在上传 {quant_type}...")
    try:
        api.upload_file(
            path_or_fileobj=quant_file,
            path_in_repo=f"{model_name}-{quant_type.lower()}.gguf",
            repo_id=OUTPUT_REPO,
        )
        print(f"   ✅ {quant_type} 已上传")
    except Exception as e:
        print(f"   ❌ {quant_type} 上传失败: {e}")
        continue

# 创建 README
print("\n📝 正在创建 README...")
readme_content = f"""---
base_model: {BASE_MODEL}
tags:
- gguf
- llama.cpp
- quantized
- trl
- sft
---

# {OUTPUT_REPO.split('/')[-1]}

这是 [{ADAPTER_MODEL}](https://huggingface.co/{ADAPTER_MODEL}) 的 GGUF 转换版本，它是 [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) 的 LoRA 微调版本。

## 模型详情

- **基础模型：** {BASE_MODEL}
- **微调模型：** {ADAPTER_MODEL}
- **训练：** 使用 TRL 进行监督微调（SFT）
- **格式：** GGUF（用于 llama.cpp、Ollama、LM Studio 等）

## 可用量化版本

| 文件 | 量化 | 大小 | 描述 | 使用场景 |
|------|-------|------|-------------|----------|
| {model_name}-f16.gguf | F16 | ~1GB | 全精度 | 最佳质量，速度较慢 |
| {model_name}-q8_0.gguf | Q8_0 | ~500MB | 8 位 | 高质量 |
| {model_name}-q5_k_m.gguf | Q5_K_M | ~350MB | 5 位中等 | 良好质量，更小 |
| {model_name}-q4_k_m.gguf | Q4_K_M | ~300MB | 4 位中等 | 推荐 - 良好平衡 |

## 使用方法

### 使用 llama.cpp

```bash
# 下载模型
huggingface-cli download {OUTPUT_REPO} {model_name}-q4_k_m.gguf

# 使用 llama.cpp 运行
./llama-cli -m {model_name}-q4_k_m.gguf -p "在此输入您的提示词"
```

### 使用 Ollama

1. 创建一个 `Modelfile`：
```
FROM ./{model_name}-q4_k_m.gguf
```

2. 创建模型：
```bash
ollama create my-model -f Modelfile
ollama run my-model
```

### 使用 LM Studio

1. 下载 `.gguf` 文件
2. 导入到 LM Studio
3. 开始聊天！

## 许可证

继承自基础模型的许可证：{BASE_MODEL}

## 引用

```bibtex
@misc{{{OUTPUT_REPO.split('/')[-1].replace('-', '_')},
  author = {{{username}}},
  title = {{{OUTPUT_REPO.split('/')[-1]}}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{OUTPUT_REPO}}}
}}
```

---

*使用 llama.cpp 转换为 GGUF 格式*
"""

try:
    # 上传 README 文件
    # encode() 将字符串编码为字节
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=OUTPUT_REPO,
    )
    print("   ✅ README 已上传")
except Exception as e:
    print(f"   ❌ README 上传失败: {e}")

print("\n" + "=" * 60)
print("✅ GGUF 转换完成！")
print(f"📦 仓库：https://huggingface.co/{OUTPUT_REPO}")
print(f"\n📥 下载命令：")
print(f"   huggingface-cli download {OUTPUT_REPO} {model_name}-q4_k_m.gguf")
print(f"\n🚀 使用 Ollama：")
print("   1. 下载 GGUF 文件")
print(f"   2. 创建 Modelfile: FROM ./{model_name}-q4_k_m.gguf")
print("   3. ollama create my-model -f Modelfile")
print("   4. ollama run my-model")
print("=" * 60)
