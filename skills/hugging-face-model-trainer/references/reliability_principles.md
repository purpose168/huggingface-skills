# 训练任务的可靠性原则

这些原则源自真实的生产环境故障和成功修复方案。遵循这些原则可以防止常见的失败模式，确保任务可靠执行。

## 原则 1：使用前务必验证

**规则：** 永远不要假设代码仓库、数据集或资源存在。先使用工具进行验证。

### 防止的问题

- **不存在的数据集** - 当数据集不存在时任务会立即失败
- **名称拼写错误** - 如 "argilla-dpo-mix-7k" 与 "ultrafeedback_binarized" 这样的简单错误
- **路径错误** - 旧的或已移动的仓库、重命名的文件
- **缺少依赖** - 未记录的要求

### 如何应用

**在提交任何任务之前：**

```python
# 验证数据集是否存在
dataset_search({"query": "dataset-name", "author": "author-name", "limit": 5})
hub_repo_details(["author/dataset-name"], repo_type="dataset")

# 验证模型是否存在
hub_repo_details(["org/model-name"], repo_type="model")

# 检查脚本/文件路径（对于基于 URL 的脚本）
# 使用前验证：https://github.com/user/repo/blob/main/script.py
```

**能够捕获错误的示例：**

```python
# ❌ 错误：假设数据集存在
hf_jobs("uv", {
    "script": """...""",
    "env": {"DATASET": "trl-lib/argilla-dpo-mix-7k"}  # 不存在！
})

# ✅ 正确：先验证
dataset_search({"query": "argilla dpo", "author": "trl-lib"})
# 会显示："trl-lib/ultrafeedback_binarized" 是正确的名称

hub_repo_details(["trl-lib/ultrafeedback_binarized"], repo_type="dataset")
# 使用前确认其存在
```

### 实施检查清单

- [ ] 训练前检查数据集是否存在
- [ ] 微调前验证基础模型是否存在
- [ ] GGUF 转换前确认适配器模型是否存在
- [ ] 提交前测试脚本 URL 是否有效
- [ ] 验证仓库中的文件路径
- [ ] 检查资源的近期更新/重命名情况

**时间成本：** 5-10 秒
**节省时间：** 数小时的失败任务时间 + 调试时间

---

## 原则 2：可靠性优先于性能

**规则：** 默认选择最可能成功的方案，而不是理论上最快的方案。

### 防止的问题

- **硬件不兼容** - 在某些 GPU 上会失败的功能
- **不稳定的优化** - 导致崩溃的加速方法
- **复杂配置** - 更多的失败点
- **构建系统问题** - 不可靠的编译方法

### 如何应用

**选择可靠性：**

```python
# ❌ 有风险：可能失败的激进优化
SFTConfig(
    torch_compile=True,  # 在 T4、A10G GPU 上可能失败
    optim="adamw_bnb_8bit",  # 需要特定设置
    fp16=False,  # 可能导致训练不稳定
    ...
)

# ✅ 安全：经过验证的默认值
SFTConfig(
    # torch_compile=True,  # 已注释并备注："在 H100 上启用可提升 20% 速度"
    optim="adamw_torch",  # 标准，始终有效
    fp16=True,  # 稳定且快速
    ...
)
```

**对于构建过程：**

```python
# ❌ 不可靠：使用 make（依赖平台）
subprocess.run(["make", "-C", "/tmp/llama.cpp", "llama-quantize"], check=True)

# ✅ 可靠：使用 CMake（一致、有文档）
subprocess.run([
    "cmake", "-B", "/tmp/llama.cpp/build", "-S", "/tmp/llama.cpp",
    "-DGGML_CUDA=OFF"  # 禁用 CUDA 以获得更快、更可靠的构建
], check=True)

subprocess.run([
    "cmake", "--build", "/tmp/llama.cpp/build",
    "--target", "llama-quantize", "-j", "4"
], check=True)
```

### 真实案例

**`torch.compile` 失败案例：**
- 为在 H100 上"提升 20% 速度"而添加
- **在 T4-medium 上致命失败**，错误信息晦涩难懂
- 被误诊为数据集问题（耗费数小时）
- **修复：** 默认禁用，作为可选注释添加

**结果：** 可靠性 > 20% 的性能提升

### 实施检查清单

- [ ] 默认使用经过验证的标准配置
- [ ] 用硬件说明注释掉性能优化
- [ ] 使用稳定的构建系统（CMake > make）
- [ ] 生产前在目标硬件上测试
- [ ] 记录已知的不兼容性
- [ ] 需要时提供"安全"和"快速"的变体

**性能损失：** 最佳情况下 10-20%
**可靠性提升：** 95%+ 成功率 vs 60-70%

---

## 原则 3：创建原子化、自包含的脚本

**规则：** 脚本应作为完整的独立单元工作。不要为了"简化"而删除部分内容。

### 防止的问题

- **缺少依赖** - 删除了"不必要"但实际上需要的包
- **不完整的过程** - 跳过了看似多余的步骤
- **环境假设** - 需要预先设置的脚本
- **部分失败** - 部分部分工作，其他部分静默失败

### 如何应用

**完整的依赖规范：**

```python
# ❌ 不完整：通过删除依赖来"简化"
# /// script
# dependencies = [
#     "transformers",
#     "peft",
#     "torch",
# ]
# ///

# ✅ 完整：所有依赖明确列出
# /// script
# dependencies = [
#     "transformers>=4.36.0",
#     "peft>=0.7.0",
#     "torch>=2.0.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "sentencepiece>=0.1.99",  # 分词器需要
#     "protobuf>=3.20.0",        # 分词器需要
#     "numpy",
#     "gguf",
# ]
# ///
```

**完整的构建过程：**

```python
# ❌ 不完整：假设构建工具存在
subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"])
subprocess.run(["make", "-C", "/tmp/llama.cpp", "llama-quantize"])  # 失败：没有 gcc/make

# ✅ 完整：安装所有要求
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential", "cmake"], check=True)
subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama.cpp"])
# ... 然后构建
```

### 真实案例

**`sentencepiece` 失败案例：**
- 原始脚本包含它：工作正常
- "简化"版本删除了它："看起来不必要"
- **GGUF 转换静默失败** - 分词器无法转换
- 难以调试：没有明显的错误消息
- **修复：** 恢复所有原始依赖

**结果：** 不要在没有充分测试的情况下删除依赖

### 实施检查清单

- [ ] 所有依赖在 PEP 723 头部中，带有版本锁定
- [ ] 所有系统包由脚本安装
- [ ] 不假设预先存在的环境
- [ ] 没有实际上需要的"可选"步骤
- [ ] 在干净环境中测试脚本
- [ ] 记录为什么需要每个依赖

**复杂度：** 脚本稍长
**可靠性：** 脚本每次都"正常工作"

---

## 原则 4：提供清晰的错误上下文

**规则：** 当事情失败时，使问题所在和修复方法显而易见。

### 如何应用

**包装子进程调用：**

```python
# ❌ 不清晰：静默失败
subprocess.run([...], check=True, capture_output=True)

# ✅ 清晰：显示失败内容
try:
    result = subprocess.run(
        [...],
        check=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Warnings:", result.stderr)
except subprocess.CalledProcessError as e:
    print(f"❌ Command failed!")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
    raise
```

**验证输入：**

```python
# ❌ 不清晰：稍后失败，错误晦涩
model = load_model(MODEL_NAME)

# ✅ 清晰：快速失败，消息明确
if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable not set!")

print(f"Loading model: {MODEL_NAME}")
try:
    model = load_model(MODEL_NAME)
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {MODEL_NAME}")
    print(f"Error: {e}")
    print("Hint: Check that model exists on Hub")
    raise
```

### 实施检查清单

- [ ] 用 try/except 包装外部调用
- [ ] 失败时打印 stdout/stderr
- [ ] 尽早验证环境变量
- [ ] 添加进度指示器（✅, ❌, 🔄）
- [ ] 包含常见失败的提示
- [ ] 开始时记录配置

---

## 原则 5：在已知良好输入上测试正常路径

**规则：** 在生产中使用新代码之前，先用已知有效的输入进行测试。

### 如何应用

**已知良好的测试输入：**

```python
# 用于训练
TEST_DATASET = "trl-lib/Capybara"  # 小型、格式良好、广泛使用
TEST_MODEL = "Qwen/Qwen2.5-0.5B"  # 小型、快速、可靠

# 用于 GGUF 转换
TEST_ADAPTER = "evalstate/qwen-capybara-medium"  # 已知可用的模型
TEST_BASE = "Qwen/Qwen2.5-0.5B"  # 兼容的基础模型
```

**测试工作流程：**

1. 首先用已知良好的输入测试
2. 如果有效，尝试生产输入
3. 如果生产失败，你知道是输入问题（不是代码问题）
4. 隔离差异

### 实施检查清单

- [ ] 维护已知良好的测试模型/数据集列表
- [ ] 首先用测试输入测试新脚本
- [ ] 记录什么使输入"良好"
- [ ] 保持测试任务低成本（小模型、短超时）
- [ ] 仅在测试成功后才进入生产

**时间成本：** 测试运行 5-10 分钟
**节省调试时间：** 数小时

---

## 总结：可靠性检查清单

在提交任何任务之前：

### 预检检查
- [ ] **已验证** 所有仓库/数据集存在（hub_repo_details）
- [ ] **已测试** 新代码使用已知良好的输入
- [ ] **使用** 经过验证的硬件/配置
- [ ] **已包含** PEP 723 头部中的所有依赖
- [ ] **已安装** 系统要求（构建工具等）
- [ ] **已设置** 适当的超时（不是默认 30 分钟）
- [ ] **已配置** 使用 HF_TOKEN 的 Hub 推送
- [ ] **已添加** 清晰的错误处理

### 脚本质量
- [ ] 自包含（不需要外部设置）
- [ ] 列出完整的依赖
- [ ] 构建工具由脚本安装
- [ ] 包含进度指示器
- [ ] 错误消息清晰
- [ ] 开始时记录配置

### 任务配置
- [ ] 超时 > 预期运行时间 + 30% 缓冲
- [ ] 硬件适合模型大小
- [ ] 密钥包含 HF_TOKEN
- [ ] 环境变量设置正确
- [ ] 成本已估算且可接受

**遵循这些原则可将任务成功率从约 60-70% 提升到约 95%+**

---

## 原则冲突时的选择

有时可靠性和性能会冲突。以下是选择方法：

| 场景 | 选择 | 理由 |
|----------|--------|-----------|
| 演示/测试 | 可靠性 | 快速失败比慢速成功更糟糕 |
| 生产（首次运行） | 可靠性 | 优化前先证明有效 |
| 生产（已验证） | 性能 | 验证后可以安全优化 |
| 时间关键 | 可靠性 | 失败造成的延迟比慢速运行更多 |
| 成本关键 | 平衡 | 用小模型测试，然后优化 |

**一般规则：** 可靠性优先，优化其次。

---

## 延伸阅读

- `troubleshooting.md` - 常见问题和修复
- `training_patterns.md` - 经过验证的训练配置
- `gguf_conversion.md` - 生产 GGUF 工作流程
