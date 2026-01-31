# GGUF 转换指南

在使用 TRL 在 Hugging Face Jobs 上训练模型后,将其转换为 **GGUF 格式** 以便与 llama.cpp、Ollama、LM Studio 和其他本地推理工具一起使用。

**本指南提供了基于成功转换的生产级、经过测试的代码。** 包含所有关键依赖项和构建步骤。

## 什么是 GGUF?

**GGUF**(GPT-Generated Unified Format,GPT 生成的统一格式):
- 针对 llama.cpp 的 CPU/GPU 推理优化的格式
- 支持量化(4 位、5 位、8 位)以减小模型大小
- 兼容:Ollama、LM Studio、Jan、GPT4All、llama.cpp
- 对于 7B 模型通常为 2-8GB(而未量化版本为 14GB)

## 何时转换为 GGUF

**在以下情况下转换:**
- 使用 Ollama 或 LM Studio 在本地运行模型
- 使用 CPU 优化的推理
- 通过量化减小模型大小
- 部署到边缘设备
- 分享模型以供本地优先使用

## 关键成功因素

基于生产测试,这些对于可靠转换是**必不可少的**:

### 1. ✅ 首先安装构建工具
**在克隆 llama.cpp 之前**,安装构建依赖项:
```python
subprocess.run(["apt-get", "update", "-qq"], check=True, capture_output=True)  # 更新包列表
subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake"], check=True, capture_output=True)  # 安装构建工具
```

**原因:** 量化工具需要 gcc 和 cmake。在克隆后安装没有帮助。

### 2. ✅ 使用 CMake(而不是 Make)
**使用 CMake 构建量化工具:**
```python
# 创建构建目录
os.makedirs("/tmp/llama.cpp/build", exist_ok=True)

# 配置
subprocess.run([
    "cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp",
    "-DGGML_CUDA=OFF"  # 更快的构建,量化不需要 CUDA
], check=True, capture_output=True, text=True)

# 构建
subprocess.run([
    "cmake", "--build", "/tmp/llama.cpp/build",
    "--target", "llama-quantize", "-j", "4"
], check=True, capture_output=True, text=True)

# 二进制文件路径
quantize_bin = "/tmp/llama.cpp/build/bin/llama-quantize"
```

**原因:** CMake 比 `make` 更可靠,并且产生一致的二进制路径。

### 3. ✅ 包含所有依赖项
**PEP 723 头部必须包含:**
```python
# /// script
# dependencies = [
#     "transformers>=4.36.0",
#     "peft>=0.7.0",
#     "torch>=2.0.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "sentencepiece>=0.1.99",  # 分词器必需
#     "protobuf>=3.20.0",        # 分词器必需
#     "numpy",
#     "gguf",
# ]
# ///
```

**原因:** `sentencepiece` 和 `protobuf` 对于分词器转换至关重要。缺少它们会导致静默失败。

### 4. ✅ 使用前验证名称
**始终验证仓库存在:**
```python
# 在提交作业之前,验证:
hub_repo_details([ADAPTER_MODEL], repo_type="model")  # 验证适配器模型
hub_repo_details([BASE_MODEL], repo_type="model")     # 验证基础模型
```

**原因:** 不存在的数据集/模型名称会导致作业失败,而这些错误可以在几秒钟内被捕获。

## 完整转换脚本

请参阅 `scripts/convert_to_gguf.py` 获取完整的、生产就绪的脚本。

**主要特性:**
- ✅ PEP 723 头部中的所有依赖项
- ✅ 自动安装构建工具
- ✅ CMake 构建过程(可靠)
- ✅ 全面的错误处理
- ✅ 环境变量配置
- ✅ 自动生成 README

## 快速转换作业

```python
# 提交前:验证模型存在
hub_repo_details(["username/my-finetuned-model"], repo_type="model")  # 验证微调模型
hub_repo_details(["Qwen/Qwen2.5-0.5B"], repo_type="model")           # 验证基础模型

# 提交转换作业
hf_jobs("uv", {
    "script": open("trl/scripts/convert_to_gguf.py").read(),  # 或内联脚本
    "flavor": "a10g-large",
    "timeout": "45m",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"},
    "env": {
        "ADAPTER_MODEL": "username/my-finetuned-model",
        "BASE_MODEL": "Qwen/Qwen2.5-0.5B",
        "OUTPUT_REPO": "username/my-model-gguf",
        "HF_USERNAME": "username"  # 可选,用于 README
    }
})
```

## 转换过程

脚本执行以下步骤:

1. **加载和合并** - 加载基础模型和 LoRA 适配器,合并它们
2. **安装构建工具** - 安装 gcc、cmake(关键:在克隆 llama.cpp 之前)
3. **设置 llama.cpp** - 克隆仓库,安装 Python 依赖项
4. **转换为 GGUF** - 使用 llama.cpp 转换器创建 FP16 GGUF
5. **构建量化工具** - 使用 CMake 构建 `llama-quantize`
6. **量化** - 创建 Q4_K_M、Q5_K_M、Q8_0 版本
7. **上传** - 上传所有版本 + README 到 Hub

## 量化选项

常见的量化格式(从小到大):

| 格式 | 大小 | 质量 | 用例 |
|--------|------|---------|----------|
| **Q4_K_M** | ~300MB | 良好 | **推荐** - 大小/质量的最佳平衡 |
| **Q5_K_M** | ~350MB | 更好 | 更高质量,稍大 |
| **Q8_0** | ~500MB | 很高 | 接近原始质量 |
| **F16** | ~1GB | 原始 | 全精度,最大文件 |

**建议:** 创建 Q4_K_M、Q5_K_M 和 Q8_0 版本,为用户提供选择。

## 硬件要求

**用于转换:**
- 小型模型(<1B):CPU-basic 可用,但很慢
- 中型模型(1-7B):推荐 a10g-large
- 大型模型(7B+):a10g-large 或 a100-large

**时间估算:**
- 0.5B 模型:在 A10G 上约 15-25 分钟
- 3B 模型:在 A10G 上约 30-45 分钟
- 7B 模型:在 A10G 上约 45-60 分钟

## 使用 GGUF 模型

**GGUF 模型可在 CPU 和 GPU 上运行。** 它们针对 CPU 推理进行了优化,但在可用时也可以利用 GPU 加速。

### 使用 Ollama(自动检测 GPU)
```bash
# 下载 GGUF
huggingface-cli download username/my-model-gguf model-q4_k_m.gguf

# 创建 Modelfile
echo "FROM ./model-q4_k_m.gguf" > Modelfile

# 创建并运行(如果可用则自动使用 GPU)
ollama create my-model -f Modelfile
ollama run my-model
```

### 使用 llama.cpp
```bash
# 仅 CPU
./llama-cli -m model-q4_k_m.gguf -p "Your prompt"

# 使用 GPU 加速(将 32 层卸载到 GPU)
./llama-cli -m model-q4_k_m.gguf -ngl 32 -p "Your prompt"
```

### 使用 LM Studio
1. 下载 `.gguf` 文件
2. 导入到 LM Studio
3. 开始聊天

## 最佳实践

### ✅ 应该做:
1. **在提交作业之前验证仓库存在**(使用 `hub_repo_details`)
2. **在克隆 llama.cpp 之前首先安装构建工具**
3. **使用 CMake** 构建量化工具(而不是 make)
4. **在 PEP 723 头部中包含所有依赖项**(特别是 sentencepiece、protobuf)
5. **创建多个量化版本** - 给用户选择
6. **在生产使用之前在已知模型上测试**
7. **使用 A10G GPU** 以加快转换速度

### ❌ 不应该做:
1. **假设仓库存在** - 始终使用 hub 工具验证
2. **使用 make** 而不是 CMake - 不太可靠
3. **删除依赖项** 以"简化" - 它们都是必需的
4. **跳过构建工具** - 量化将静默失败
5. **使用默认路径** - CMake 将二进制文件放在 build/bin/ 中

## 常见问题

### 合并期间内存不足
**修复:**
- 使用更大的 GPU(a10g-large 或 a100-large)
- 确保 `device_map="auto"` 用于自动放置
- 使用 `dtype=torch.float16` 或 `torch.bfloat16`

### 转换失败并出现架构错误
**修复:**
- 确保 llama.cpp 支持模型架构
- 检查标准架构(Qwen、Llama、Mistral 等)
- 更新 llama.cpp 到最新版本: `git clone --depth 1 https://github.com/ggerganov/llama.cpp.git`
- 检查 llama.cpp 文档以了解模型支持

### 量化失败
**修复:**
- 验证构建工具已安装: `apt-get install build-essential cmake`
- 使用 CMake(而不是 make)构建量化工具
- 检查二进制路径: `/tmp/llama.cpp/build/bin/llama-quantize`
- 在量化之前验证 FP16 GGUF 存在

### 缺少 sentencepiece 错误
**修复:**
- 添加到 PEP 723 头部: `"sentencepiece>=0.1.99", "protobuf>=3.20.0"`
- 不要删除依赖项以"简化" - 都是必需的

### 上传失败或超时
**修复:**
- 大型模型(>2GB)需要更长的超时时间: `"timeout": "1h"`
- 如果需要,分别上传量化版本
- 检查网络/Hub 状态

## 经验教训

这些来自生产测试和实际失败:

### 1. 始终在使用前验证
**教训:** 不要假设仓库/数据集存在。先检查。
```python
# 在提交作业之前
hub_repo_details(["trl-lib/argilla-dpo-mix-7k"], repo_type="dataset")  # 会捕获错误
```
**防止的失败:** 不存在的数据集名称、模型名称中的拼写错误

### 2. 优先考虑可靠性而非性能
**教训:** 默认使用最可能成功的方法。
- 使用 CMake(而不是 make) - 更可靠
- 在构建中禁用 CUDA - 更快,不需要
- 包含所有依赖项 - 不要"简化"

**防止的失败:** 构建失败、缺少二进制文件

### 3. 创建原子化的、自包含的脚本
**教训:** 不要删除依赖项或步骤。脚本应该作为一个单元工作。
- PEP 723 头部中的所有依赖项
- 包含所有构建步骤
- 清晰的错误消息

**防止的失败:** 缺少分词器库、构建工具失败

## 参考

**在此技能中:**
- `scripts/convert_to_gguf.py` - 完整的、生产就绪的脚本

**外部:**
- [llama.cpp 仓库](https://github.com/ggerganov/llama.cpp)
- [GGUF 规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Ollama 文档](https://ollama.ai)
- [LM Studio](https://lmstudio.ai)

## 总结

**GGUF 转换的关键检查清单:**
- [ ] 验证适配器和基础模型在 Hub 上存在
- [ ] 使用 `scripts/convert_to_gguf.py` 中的生产脚本
- [ ] PEP 723 头部中的所有依赖项(包括 sentencepiece、protobuf)
- [ ] 在克隆 llama.cpp 之前安装构建工具
- [ ] 使用 CMake 构建量化工具(而不是 make)
- [ ] 正确的二进制路径: `/tmp/llama.cpp/build/bin/llama-quantize`
- [ ] 选择 A10G GPU 以获得合理的转换时间
- [ ] 超时设置为至少 45m
- [ ] HF_TOKEN 在 secrets 中用于 Hub 上传

**`scripts/convert_to_gguf.py` 中的脚本融合了所有这些经验教训,并在生产中成功测试。**
