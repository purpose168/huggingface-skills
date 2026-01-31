---
name: hugging-face-trackio
description: 使用Trackio跟踪和可视化ML训练实验。在训练期间记录指标（Python API）或检索/分析记录的指标（CLI）时使用。支持实时仪表板可视化、HF Space同步和用于自动化的JSON输出。
---

# Trackio - ML训练实验跟踪

Trackio是一个用于记录和可视化ML训练指标的实验跟踪库。它同步到Hugging Face Spaces以进行实时监控仪表板。

## 两个接口

| 任务 | 接口 | 参考 |
|------|-----------|-----------|
| **记录指标** 在训练期间 | Python API | [references/logging_metrics.md](references/logging_metrics.md) |
| **检索指标** 训练期间/之后 | CLI | [references/retrieving_metrics.md](references/retrieving_metrics.md) |

## 何时使用每个

### Python API → 记录

在训练脚本中使用`import trackio`来记录指标：

- 使用`trackio.init()`初始化跟踪
- 使用`trackio.log()`记录指标或使用TRL的`report_to="trackio"`
- 使用`trackio.finish()`完成

**关键概念**：对于远程/云训练，传递`space_id`——指标同步到Space仪表板，因此在实例终止后它们仍然存在。

→ 有关设置、TRL集成和配置选项，请参阅[references/logging_metrics.md](references/logging_metrics.md)。

### CLI → 检索

使用`trackio`命令查询记录的指标：

- `trackio list projects/runs/metrics` — 发现可用的内容
- `trackio get project/run/metric` — 检索摘要和值
- `trackio show` — 启动仪表板
- `trackio sync` — 同步到HF Space

**关键概念**：添加`--json`以获得适合自动化和LLM代理的程序化输出。

→ 有关所有命令、工作流程和JSON输出格式，请参阅[references/retrieving_metrics.md](references/retrieving_metrics.md)。

## 最小记录设置

```python
import trackio

trackio.init(project="my-project", space_id="username/trackio")
trackio.log({"loss": 0.1, "accuracy": 0.9})
trackio.log({"loss": 0.09, "accuracy": 0.91})
trackio.finish()
```

### 最小检索

```bash
trackio list projects --json
trackio get metric --project my-project --run my-run --metric loss --json
```