# 评估表格格式示例

本文档展示了可以从模型README文件中提取的各种评估表格格式。

## 格式1：基准测试作为行（最常见）

```markdown
| 基准测试 | 分数 |
|-----------|-------|
| MMLU      | 85.2  |
| HumanEval | 72.5  |
| GSM8K     | 91.3  |
| HellaSwag | 88.9  |
```

## 格式2：多指标列

```markdown
| 基准测试 | 准确率 | F1 分数 |
|-----------|----------|----------|
| MMLU      | 85.2     | 0.84     |
| GSM8K     | 91.3     | 0.91     |
| DROP      | 78.5     | 0.77     |
```

## 格式3：基准测试作为列

```markdown
| MMLU | HumanEval | GSM8K | HellaSwag |
|------|-----------|-------|-----------|
| 85.2 | 72.5      | 91.3  | 88.9      |
```

## 格式4：百分比值

```markdown
| 基准测试     | 分数    |
|---------------|----------|
| MMLU          | 85.2%    |
| HumanEval     | 72.5%    |
| GSM8K         | 91.3%    |
| TruthfulQA    | 68.7%    |
```

## 格式5：带类别的混合格式

```markdown
### 推理

| 基准测试 | 分数 |
|-----------|-------|
| MMLU      | 85.2  |
| BBH       | 82.4  |
| GPQA      | 71.3  |

### 编码

| 基准测试 | 分数 |
|-----------|-------|
| HumanEval | 72.5  |
| MBPP      | 78.9  |

### 数学

| 基准测试 | 分数 |
|-----------|-------|
| GSM8K     | 91.3  |
| MATH      | 65.8  |
```

## 格式6：带附加列

```markdown
| 基准测试 | 分数 | 排名 | 备注              |
|-----------|-------|------|--------------------|
| MMLU      | 85.2  | #5   | 5-shot             |
| HumanEval | 72.5  | #8   | pass@1             |
| GSM8K     | 91.3  | #3   | 8-shot, maj@1      |
```

## 提取器工作原理

脚本将：
1. 找到README中的所有markdown表格
2. 识别哪些表格包含评估结果
3. 解析表格结构（行与列）
4. 提取数值作为分数
5. 转换为model-index YAML格式

## README作者提示

为确保您的评估表格被正确提取：

1. **使用清晰的标题**：包含"基准测试"、"分数"或类似术语
2. **保持简单**：坚持使用基准测试名称+分数列
3. **使用标准格式**：遵循markdown表格语法
4. **包含数值**：确保分数是可解析的数字
5. **保持一致**：在多个表格中使用相同的格式

## 完整README部分示例

```markdown
# MyModel-7B 模型卡片

## 评估结果

我们的模型在多个标准基准测试上进行了评估：

| 基准测试     | 分数 |
|---------------|-------|
| MMLU          | 85.2  |
| HumanEval     | 72.5  |
| GSM8K         | 91.3  |
| HellaSwag     | 88.9  |
| ARC-Challenge | 81.7  |
| TruthfulQA    | 68.7  |

### 详细结果

有关更详细的结果和方法，请参阅我们的[论文](link)。
```

## 运行提取器

```bash
# 从本示例中提取
python scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/your-model" \
  --dry-run

# 应用到您的模型卡片
python scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/your-model" \
  --task-type "text-generation"
```
