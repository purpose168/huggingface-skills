# TRL 训练任务故障排除

在 Hugging Face Jobs 上使用 TRL 进行训练时的常见问题和解决方案。

## 训练卡在"Starting training..."步骤

**问题:** 任务启动但卡在训练步骤 - 从不推进,从不超时,就停在那里。

**根本原因:** 使用了 `eval_strategy="steps"` 或 `eval_strategy="epoch"` 但没有向训练器提供 `eval_dataset`。

**解决方案:**

**选项 A: 提供 eval_dataset (推荐)**
```python
# 创建训练/评估数据集分割
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)  # 将10%的数据作为评估集,使用固定随机种子42保证可复现性

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",  # 使用Qwen 2.5-0.5B模型
    train_dataset=dataset_split["train"],  # 训练集
    eval_dataset=dataset_split["test"],  # ← 当启用eval_strategy时必须提供评估集
    args=SFTConfig(
        eval_strategy="steps",  # 按步骤进行评估
        eval_steps=50,  # 每50步评估一次
        ...
    ),
)
```

**选项 B: 禁用评估**
```python
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",  # 使用Qwen 2.5-0.5B模型
    train_dataset=dataset,  # 仅使用训练集
    # 不提供eval_dataset
    args=SFTConfig(
        eval_strategy="no",  # ← 显式禁用评估
        ...
    ),
)
```

**预防措施:**
- 始终创建训练/评估分割以便更好地监控训练过程
- 使用 `dataset.train_test_split(test_size=0.1, seed=42)` 进行数据集分割
- 检查示例脚本: `scripts/train_sft_example.py` 包含正确的评估设置

## 任务超时

**问题:** 任务在训练完成前终止,所有进度丢失。

**解决方案:**
- 增加超时参数 (例如 `"timeout": "4h"`)  # 设置任务最长运行时间为4小时
- 减少 `num_train_epochs` 或使用更小的数据集切片  # 减少训练轮数或数据量
- 使用更小的模型或启用 LoRA/PEFT 以加快训练速度  # LoRA(低秩适应)和PEFT(参数高效微调)可以显著减少计算量
- 为加载/保存开销增加20-30%的时间缓冲  # 考虑模型加载和保存的额外时间

**预防措施:**
- 始先从快速演示运行开始以估算时间  # 先用小数据集测试
- 使用 `scripts/estimate_cost.py` 获取时间估算  # 运行成本估算脚本
- 通过 Trackio 或日志密切监控首次运行  # 实时监控训练进度

## 模型未保存到 Hub

**问题:** 训练完成但模型未出现在 Hub 上 - 所有工作丢失。

**检查清单:**
- [ ] 训练配置中设置了 `push_to_hub=True`  # 确保启用推送到Hub
- [ ] 使用用户名指定了 `hub_model_id` (例如 `"username/model-name"`)  # 指定模型在Hub上的完整路径
- [ ] 任务提交时设置了 `secrets={"HF_TOKEN": "$HF_TOKEN"}`  # 提供Hugging Face访问令牌
- [ ] 用户对目标仓库有写入权限  # 确保用户有权限写入
- [ ] 令牌具有写入权限 (在 https://huggingface.co/settings/tokens 检查)  # 验证令牌权限
- [ ] 训练脚本在末尾调用了 `trainer.push_to_hub()`  # 确保脚本执行推送操作

**参考:** `references/hub_saving.md` 了解详细的 Hub 身份验证故障排除

## 内存不足 (OOM)

**问题:** 任务因 CUDA 内存不足错误而失败。

**解决方案 (按优先级排序):**
1. **减小批大小:** 降低 `per_device_train_batch_size` (尝试 4 → 2 → 1)  # 减少每个设备的批大小以减少内存占用
2. **增加梯度累积:** 提高 `gradient_accumulation_steps` 以保持有效批大小  # 通过累积梯度来模拟更大的批大小
3. **禁用评估:** 移除 `eval_dataset` 和 `eval_strategy` (节省约40%内存,适合演示)  # 评估会占用额外内存
4. **启用 LoRA/PEFT:** 使用 `peft_config=LoraConfig(r=8, lora_alpha=16)` 仅训练适配器 (更小的秩 = 更少内存)  # LoRA只训练少量参数
5. **使用更大的 GPU:** 从 `t4-small` → `l4x1` → `a10g-large` → `a100-large` 切换  # 更强大的GPU有更多显存
6. **启用梯度检查点:** 在配置中设置 `gradient_checkpointing=True` (较慢但节省内存)  # 以计算换内存
7. **使用更小的模型:** 尝试更小的变体 (例如 0.5B 而不是 3B)  # 模型参数越少,内存占用越小

**内存指南:**
- T4 (16GB): <1B 模型配合 LoRA  # 适合小模型微调
- A10G (24GB): 1-3B 模型配合 LoRA, <1B 完整微调  # 中等规模训练
- A100 (40GB/80GB): 7B+ 模型配合 LoRA, 3B 完整微调  # 大规模训练

## 参数命名问题

**问题:** `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`

**原因:** TRL 配置类使用 `max_length`,而不是 `max_seq_length`。

**解决方案:**
```python
# ✅ 正确 - TRL 使用 max_length
SFTConfig(max_length=512)  # 设置最大序列长度为512
DPOConfig(max_length=512)

# ❌ 错误 - 这会失败
SFTConfig(max_seq_length=512)  # 参数名称错误
```

**注意:** 大多数 TRL 配置不需要显式设置 max_length - 默认值(1024)效果很好。仅在需要特定值时设置。

## 数据集格式错误

**问题:** 训练因数据集格式错误或缺少字段而失败。

**解决方案:**
1. **检查格式文档:**
   ```python
   hf_doc_fetch("https://huggingface.co/docs/trl/dataset_formats")  # 获取TRL数据集格式文档
   ```

2. **训练前验证数据集:**
   ```bash
   uv run https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py \
     --dataset <dataset-name> --split train  # 使用数据集检查工具验证训练集
   ```
   或通过 hf_jobs:
   ```python
   hf_jobs("uv", {  # 在Hugging Face Jobs上运行检查
       "script": "https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py",
       "script_args": ["--dataset", "dataset-name", "--split", "train"]  # 指定数据集名称和分割
   })
   ```

3. **验证字段名称:**
   - **SFT (监督微调):** 需要 "messages" 字段(对话格式),或 "text" 字段,或 "prompt"/"completion"  # 支持多种格式
   - **DPO (直接偏好优化):** 需要 "chosen" 和 "rejected" 字段  # 需要偏好对
   - **GRPO (群体相对策略优化):** 需要仅提示格式  # 只需要提示

4. **检查数据集分割:**
   - 确保分割存在 (例如 `split="train"`)  # 验证数据集是否包含该分割
   - 预览数据集: `load_dataset("name", split="train[:5]")`  # 查看前5条数据

## 导入/模块错误

**问题:** 任务因 "ModuleNotFoundError" 或导入错误而失败。

**解决方案:**
1. **添加带有依赖项的 PEP 723 头部:**
   ```python
   # /// script  # PEP 723脚本依赖声明开始标记
   # dependencies = [  # 依赖包列表
   #     "trl>=0.12.0",  # TRL库,版本>=0.12.0
   #     "peft>=0.7.0",  # PEFT库,版本>=0.7.0
   #     "transformers>=4.36.0",  # Transformers库,版本>=4.36.0
   # ]
   # ///  # PEP 723脚本依赖声明结束标记
   ```

2. **验证确切格式:**
   - 必须有 `# ///` 分隔符 (#后面有空格)  # 格式必须严格正确
   - 依赖项必须是有效的 PyPI 包名称  # 包名必须存在于PyPI
   - 检查拼写和版本约束  # 确保版本号格式正确

3. **先在本地测试:**
   ```bash
   uv run train.py  # 测试依赖项是否正确  # 使用uv运行脚本以验证依赖
   ```

## 身份验证错误

**问题:** 推送到 Hub 时任务因身份验证或权限错误而失败。

**解决方案:**
1. **验证身份验证:**
   ```python
   mcp__huggingface__hf_whoami()  # 检查当前身份验证的用户信息
   ```

2. **检查令牌权限:**
   - 访问 https://huggingface.co/settings/tokens  # 进入令牌管理页面
   - 确保令牌具有 "write" 权限  # 令牌必须有写入权限
   - 令牌不能是 "read-only"  # 只读令牌无法推送

3. **验证任务中的令牌:**
   ```python
   "secrets": {"HF_TOKEN": "$HF_TOKEN"}  # 必须在任务配置中  # 通过环境变量传递令牌
   ```

4. **检查仓库权限:**
   - 用户必须对目标仓库有写入权限  # 验证用户权限
   - 如果是组织仓库,用户必须是具有写入权限的成员  # 组织成员需要相应权限
   - 仓库必须存在或用户必须有创建权限  # 确保可以创建仓库

## 任务卡住或未启动

**问题:** 任务长时间显示 "pending" 或 "starting"。

**解决方案:**
- 检查 Jobs 仪表板的状态: https://huggingface.co/jobs  # 查看任务状态
- 验证硬件可用性 (某些 GPU 类型可能有队列)  # 某些GPU可能需要排队
- 如果某个硬件类型使用率很高,尝试不同的硬件规格  # 切换到其他可用的GPU
- 检查账户计费问题 (Jobs 需要付费计划)  # 确保账户有足够额度

**典型启动时间:**
- CPU 任务: 10-30 秒  # CPU任务启动较快
- GPU 任务: 30-90 秒  # GPU任务需要更多时间初始化
- 如果 >3 分钟: 可能排队或卡住  # 超过3分钟说明有问题

## 训练损失不下降

**问题:** 训练运行但损失保持平坦或不改善。

**解决方案:**
1. **检查学习率:** 可能太低 (尝试 2e-5 到 5e-5) 或太高 (尝试 1e-6)  # 学习率对训练效果影响很大
2. **验证数据集质量:** 检查示例以确保它们合理  # 确保数据质量良好
3. **检查模型大小:** 非常小的模型可能没有足够容量完成任务  # 模型容量不足
4. **增加训练步数:** 可能需要更多轮次或更大的数据集  # 训练不充分
5. **验证数据集格式:** 错误的格式可能导致训练效果下降  # 格式问题影响训练

## 日志未出现

**问题:** 无法看到训练日志或进度。

**解决方案:**
1. **等待 30-60 秒:** 初始日志可能会延迟  # 日志系统有延迟
2. **通过 MCP 工具检查日志:**
   ```python
   hf_jobs("logs", {"job_id": "your-job-id"})  # 获取指定任务的日志
   ```
3. **使用 Trackio 进行实时监控:** 参见 `references/trackio_guide.md`  # 使用Trackio工具
4. **验证任务实际正在运行:**
   ```python
   hf_jobs("inspect", {"job_id": "your-job-id"})  # 检查任务详细状态
   ```

## 检查点/恢复问题

**问题:** 无法从检查点恢复或检查点未保存。

**解决方案:**
1. **启用检查点保存:**
   ```python
   SFTConfig(
       save_strategy="steps",  # 按步骤保存检查点
       save_steps=100,  # 每100步保存一次
       hub_strategy="every_save",  # 每次保存都推送到Hub
   )
   ```

2. **验证检查点已推送到 Hub:** 检查模型仓库中的检查点文件夹  # 确认检查点文件存在

3. **从检查点恢复:**
   ```python
   trainer = SFTTrainer(
       model="username/model-name",  # 可以是检查点路径
       resume_from_checkpoint="username/model-name/checkpoint-1000",  # 从第1000步的检查点恢复
   )
   ```

## 获取帮助

如果问题仍然存在:

1. **检查 TRL 文档:**
   ```python
   hf_doc_search("your issue", product="trl")  # 搜索TRL相关文档
   ```

2. **检查 Jobs 文档:**
   ```python
   hf_doc_fetch("https://huggingface.co/docs/huggingface_hub/guides/jobs")  # 获取Jobs指南
   ```

3. **查看相关指南:**
   - `references/hub_saving.md` - Hub 身份验证问题  # Hub相关配置问题
   - `references/hardware_guide.md` - 硬件选择和规格  # GPU选择指南
   - `references/training_patterns.md` - 评估数据集要求  # 训练模式说明
   - SKILL.md "Working with Scripts" 部分 - 脚本格式和 URL 问题  # 脚本使用指南

4. **在 HF 论坛提问:** https://discuss.huggingface.co/  # 社区支持
