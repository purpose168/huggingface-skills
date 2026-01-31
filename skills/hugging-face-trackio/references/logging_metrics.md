# 使用Trackio记录指标

**Trackio** 是来自Hugging Face的轻量级、免费实验跟踪库。它提供了与wandb兼容的API，用于以本地优先设计记录指标。

- **GitHub**：[gradio-app/trackio](https://github.com/gradio-app/trackio)
- **文档**：[huggingface.co/docs/trackio](https://huggingface.co/docs/trackio/index)

## 安装

```bash
pip install trackio
# 或
uv pip install trackio
```

## 核心API

### 基本用法

```python
import trackio

# 初始化一个运行
trackio.init(
    project="my-project",
    config={"learning_rate": 0.001, "epochs": 10}
)

# 在训练期间记录指标
for epoch in range(10):
    loss = train_epoch()
    trackio.log({"loss": loss, "epoch": epoch})

# 完成运行
trackio.finish()
```

### 关键函数

| 函数 | 用途 |
|----------|---------|
| `trackio.init(...)` | 开始新的跟踪运行 |
| `trackio.log(dict)` | 记录指标（在训练期间重复调用） |
| `trackio.finish()` | 完成运行并确保所有指标已保存 |
| `trackio.show()` | 启动本地仪表板 |
| `trackio.sync(...)` | 将本地项目同步到HF Space |

## trackio.init() 参数

```python
trackio.init(
    project="my-project",           # 项目名称（将运行分组在一起）
    name="run-name",                # 可选：此特定运行的名称
    config={...},                   # 要记录的超参数和配置
    space_id="username/trackio",    # 可选：同步到HF Space以获取远程仪表板
    group="experiment-group",       # 可选：将相关运行分组
)
```

## 本地 vs 远程仪表板

### 本地（默认）

默认情况下，trackio将指标存储在本地SQLite数据库中并在本地运行仪表板：

```python
trackio.init(project="my-project")
# ... 训练 ...
trackio.finish()

# 启动本地仪表板
trackio.show()
```

或从终端：
```bash
trackio show --project my-project
```

### 远程（HF Space）

传递`space_id`以将指标同步到Hugging Face Space，以获得持久的、可共享的仪表板：

```python
trackio.init(
    project="my-project",
    space_id="username/trackio"  # 如果Space不存在则自动创建
)
```

⚠️ **对于远程训练**（云GPU、HF Jobs等）：始终使用`space_id`，因为当实例终止时本地存储会丢失。

### 将本地同步到远程

将现有的本地项目同步到Space：

```python
trackio.sync(project="my-project", space_id="username/my-experiments")
```

## wandb兼容性

Trackio与wandb的API兼容。即插即用的替换：

```python
import trackio as wandb

wandb.init(project="my-project")
wandb.log({"loss": 0.5})
wandb.finish()
```

## TRL集成

使用TRL训练器时，设置`report_to="trackio"`以进行自动指标记录：

```python
from trl import SFTConfig, SFTTrainer
import trackio

trackio.init(
    project="sft-training",
    space_id="username/trackio",
    config={"model": "Qwen/Qwen2.5-0.5B", "dataset": "trl-lib/Capybara"}
)

config = SFTConfig(
    output_dir="./output",
    report_to="trackio",  # 自动指标记录
    # ... 其他配置
)

trainer = SFTTrainer(model=model, args=config, ...)
trainer.train()
trackio.finish()
```

## 记录的内容

使用TRL/Transformers集成时，trackio自动捕获：
- 训练损失
- 学习率
- 评估指标
- 训练吞吐量

对于手动记录，记录任何数值指标：

```python
trackio.log({
    "train_loss": 0.5,
    "train_accuracy": 0.85,
    "val_loss": 0.4,
    "val_accuracy": 0.88,
    "epoch": 1
})
```

## 分组运行

使用`group`在仪表板侧边栏中组织相关实验：

```python
# 按实验类型分组
trackio.init(project="my-project", name="baseline-v1", group="baseline")
trackio.init(project="my-project", name="augmented-v1", group="augmented")

# 按超参数分组
trackio.init(project="hyperparam-sweep", name="lr-0.001", group="lr_0.001")
trackio.init(project="hyperparam-sweep", name="lr-0.01", group="lr_0.01")
```

## 配置最佳实践

保持配置最小化——只记录对比较运行有用的内容：

```python
trackio.init(
    project="qwen-sft-capybara",
    name="baseline-lr2e5",
    config={
        "model": "Qwen/Qwen2.5-0.5B",
        "dataset": "trl-lib/Capybara",
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "batch_size": 8,
    }
)
```

## 嵌入仪表板

使用查询参数将Space仪表板嵌入网站：

```html
<iframe 
  src="https://username-trackio.hf.space/?project=my-project&metrics=train_loss,val_loss&sidebar=hidden" 
  style="width:1600px; height:500px; border:0;">
</iframe>
```

查询参数：
- `project`：过滤到特定项目
- `metrics`：要显示的逗号分隔的指标名称
- `sidebar`：`hidden` 或 `collapsed`
- `smoothing`：0-20（平滑滑块值）
- `xmin`, `xmax`：X轴限制