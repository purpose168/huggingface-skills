# 第2周：发布Hub数据集

在Hub上创建并分享高质量数据集。好的数据是好模型的基础——通过贡献其他人可以训练的数据集来帮助社区。

## 为什么这很重要

最好的开源模型建立在公开可用的数据集之上。通过发布文档完善、结构合理的数据集，你直接为下一代模型开发提供了支持。质量比数量更重要。

## 技能

使用`hf-datasets/`完成此任务。关键功能：

- 以适当的结构初始化数据集仓库
- 多格式支持：聊天、分类、问答、补全、表格
- 基于模板的数据质量验证
- 流式上传，无需下载整个数据集

```bash
# 使用模板快速设置
python hf-datasets/scripts/dataset_manager.py quick_setup \
  --repo_id "your-username/dataset-name" --template chat
```

## XP等级

### 🐢 入门 — 50 XP

**上传一个小型、干净的数据集，包含完整的数据集卡片。**

1. 创建一个≤1,000行的数据集
2. 编写数据集卡片，包括：许可证、分割和数据来源
3. 上传到黑客马拉松组织下的Hub（或你自己的账户）

**计算标准：** 干净的数据、清晰的文档、适当的许可。

```bash
python hf-datasets/scripts/dataset_manager.py init \
  --repo_id "hf-skills/your-dataset-name"

python hf-datasets/scripts/dataset_manager.py add_rows \
  --repo_id "hf-skills/your-dataset-name" \
  --template classification \
  --rows_json "$(cat your_data.json)"
```

### 🐕 标准 — 100 XP

**发布一个会话数据集，包含完整的数据集卡片。**

1. 创建一个≤1,000行的数据集
2. 编写数据集卡片，包括：许可证和分割。
3. 上传到黑客马拉松组织下的Hub。

**计算标准：** 干净的数据、清晰的文档、适当的许可。

### 🦁 高级 — 200 XP

**将数据集翻译成多种语言并在Hub上发布。**

1. 在Hub上找到一个数据集
2. 将数据集翻译成多种语言
3. 在黑客马拉松组织下的Hub上发布翻译后的数据集

**计算标准：** 翻译后的数据集和已合并的PR。

## 资源

- [SKILL.md](../../hf-datasets/SKILL.md) — 完整技能文档
- [Templates](../../hf-datasets/templates/) — 每种格式的JSON模板
- [Examples](../../hf-datasets/examples/) — 样本数据和系统提示

---

**下一个任务：** [监督微调](04_sft-finetune-hub.md)