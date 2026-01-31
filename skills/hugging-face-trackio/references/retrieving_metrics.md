# 使用Trackio CLI检索指标

`trackio` CLI提供直接的终端访问，以在本地查询Trackio实验跟踪数据，而无需启动MCP服务器。

## 快速命令参考

| 任务 | 命令 |
|------|---------|
| 列出项目 | `trackio list projects` |
| 列出运行 | `trackio list runs --project <name>` |
| 列出指标 | `trackio list metrics --project <name> --run <name>` |
| 列出系统指标 | `trackio list system-metrics --project <name> --run <name>` |
| 获取项目摘要 | `trackio get project --project <name>` |
| 获取运行摘要 | `trackio get run --project <name> --run <name>` |
| 获取指标值 | `trackio get metric --project <name> --run <name> --metric <name>` |
| 获取系统指标 | `trackio get system-metric --project <name> --run <name>` |
| 显示仪表板 | `trackio show [--project <name>]` |
| 同步到Space | `trackio sync --project <name> --space-id <space_id>` |

## 核心命令

### 列表命令

```bash
trackio list projects                                    # 列出所有项目
trackio list projects --json                            # JSON输出

trackio list runs --project <name>                      # 列出项目中的运行
trackio list runs --project <name> --json               # JSON输出

trackio list metrics --project <name> --run <name>      # 列出运行的指标
trackio list metrics --project <name> --run <name> --json

trackio list system-metrics --project <name> --run <name>  # 列出系统指标
trackio list system-metrics --project <name> --run <name> --json
```

### 获取命令

```bash
trackio get project --project <name>                    # 项目摘要
trackio get project --project <name> --json             # JSON输出

trackio get run --project <name> --run <name>           # 运行摘要
trackio get run --project <name> --run <name> --json

trackio get metric --project <name> --run <name> --metric <name>  # 指标值
trackio get metric --project <name> --run <name> --metric <name> --json

trackio get system-metric --project <name> --run <name>           # 所有系统指标
trackio get system-metric --project <name> --run <name> --metric <name>  # 特定指标
trackio get system-metric --project <name> --run <name> --json
```

### 仪表板命令

```bash
trackio show                                              # 启动仪表板
trackio show --project <name>                           # 加载特定项目
trackio show --theme <theme>                            # 自定义主题
trackio show --mcp-server                                # 启用MCP服务器
trackio show --color-palette "#FF0000,#00FF00"         # 自定义颜色
```

### 同步命令

```bash
trackio sync --project <name> --space-id <space_id>     # 同步到HF Space
trackio sync --project <name> --space-id <space_id> --private  # 私有空间
trackio sync --project <name> --space-id <space_id> --force   # 覆盖
```

## 输出格式

所有`list`和`get`命令支持两种输出格式：

- **人类可读**（默认）：用于终端查看的格式化文本
- **JSON**（带有`--json`标志）：用于程序化使用的结构化JSON

## 常见模式

### 发现项目和运行

```bash
# 列出所有可用项目
trackio list projects

# 列出项目中的运行
trackio list runs --project my-project

# 获取项目概览
trackio get project --project my-project --json
```

### 检查运行详情

```bash
# 获取带有所有指标的运行摘要
trackio get run --project my-project --run my-run --json

# 列出可用指标
trackio list metrics --project my-project --run my-run

# 获取特定指标值
trackio get metric --project my-project --run my-run --metric loss --json
```

### 查询系统指标

```bash
# 列出系统指标（GPU等）
trackio list system-metrics --project my-project --run my-run

# 获取所有系统指标数据
trackio get system-metric --project my-project --run my-run --json

# 获取特定系统指标
trackio get system-metric --project my-project --run my-run --metric gpu_utilization --json
```

### 自动化脚本

```bash
# 提取最新指标值
LATEST_LOSS=$(trackio get metric --project my-project --run my-run --metric loss --json | jq -r '.values[-1].value')

# 将运行摘要导出到文件
trackio get run --project my-project --run my-run --json > run_summary.json

# 使用jq过滤运行
trackio list runs --project my-project --json | jq '.runs[] | select(startswith("train"))'
```

### LLM代理工作流程

```bash
# 1. 发现可用项目
trackio list projects --json

# 2. 探索项目结构
trackio get project --project my-project --json

# 3. 检查特定运行
trackio get run --project my-project --run my-run --json

# 4. 查询指标值
trackio get metric --project my-project --run my-run --metric accuracy --json
```

## 错误处理

命令验证输入并返回清晰的错误：

- 缺少项目：`Error: Project '<name>' not found.`
- 缺少运行：`Error: Run '<name>' not found in project '<project>'.`
- 缺少指标：`Error: Metric '<name>' not found in run '<run>' of project '<project>'.`

所有错误以非零状态码退出并写入stderr。

## 关键选项

- `--project`：项目名称（大多数命令必需）
- `--run`：运行名称（运行特定命令必需）
- `--metric`：指标名称（指标特定命令必需）
- `--json`：以JSON格式输出而非人类可读
- `--theme`：仪表板主题（用于`show`命令）
- `--mcp-server`：启用MCP服务器模式（用于`show`命令）
- `--color-palette`：逗号分隔的十六进制颜色（用于`show`命令）
- `--private`：创建私有空间（用于`sync`命令）
- `--force`：覆盖现有数据库（用于`sync`命令）

## JSON输出结构

### 列出项目
```json
{"projects": ["project1", "project2"]}
```

### 列出运行
```json
{"project": "my-project", "runs": ["run1", "run2"]}
```

### 项目摘要
```json
{
  "project": "my-project",
  "num_runs": 3,
  "runs": ["run1", "run2", "run3"],
  "last_activity": 100
}
```

### 运行摘要
```json
{
  "project": "my-project",
  "run": "my-run",
  "num_logs": 50,
  "metrics": ["loss", "accuracy"],
  "config": {"learning_rate": 0.001},
  "last_step": 49
}
```

### 指标值
```json
{
  "project": "my-project",
  "run": "my-run",
  "metric": "loss",
  "values": [
    {"step": 0, "timestamp": "2024-01-01T00:00:00", "value": 0.5},
    {"step": 1, "timestamp": "2024-01-01T00:01:00", "value": 0.4}
  ]
}
```

## 参考

- **完整的CLI文档**：参见[docs/source/cli_commands.md](docs/source/cli_commands.md)
- **API和MCP服务器**：参见[docs/source/api_mcp_server.md](docs/source/api_mcp_server.md)