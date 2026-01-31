# 第3周：在Hub上进行监督微调

在Hub上微调并分享模型。获取一个基础模型，在你的数据上训练它，并发布结果供社区使用。

## 为什么这很重要

微调是我们将基础模型适应特定任务的方式。通过分享微调后的模型——连同你的训练方法——你为社区提供了即用型解决方案和可重现的配方，供他们学习。

## 技能

使用`hf-llm-trainer/`完成此任务。关键功能：

- **SFT**（监督微调）—— 标准指令调优
- **DPO**（直接偏好优化）—— 从偏好数据中进行对齐
- **GRPO**（组相对策略优化）—— 在线RL训练
- 在HF Jobs上进行云GPU训练——无需本地设置
- Trackio集成用于实时监控
- GGUF转换用于本地部署

你的编码代理使用`hf_jobs()`将训练脚本直接提交到HF基础设施。

## XP等级

我们将很快公布此任务的XP等级。

## 资源

- [SKILL.md](../../hf-llm-trainer/SKILL.md) — 完整技能文档
- [SFT示例](../../hf-llm-trainer/scripts/train_sft_example.py) — 生产级SFT模板
- [DPO示例](../../hf-llm-trainer/scripts/train_dpo_example.py) — 生产级DPO模板
- [GRPO示例](../../hf-llm-trainer/scripts/train_grpo_example.py) — 生产级GRPO模板
- [训练方法](../../hf-llm-trainer/references/training_methods.md) — 方法选择指南
- [硬件指南](../../hf-llm-trainer/references/hardware_guide.md) — GPU选择