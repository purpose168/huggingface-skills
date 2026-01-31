---
title: 黑客排行榜
emoji: 🏆
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# 黑客排行榜

跟踪 [hf-skills](https://huggingface.co/hf-skills) 组织的参与度，用于黑客马拉松排行榜。

## 积分规则

简单公平 - **每项活动 1 分**：

| 活动 | 积分 |
|------|------|
| 💬 开启讨论 | 1 |
| 📝 发表评论 | 1 |
| 🔀 开启 PR | 1 |
| 📦 拥有/创建仓库 | 1 |

## 脚本

### 收集积分

```bash
# 仅收集组织活动
HF_TOKEN=$HF_TOKEN python collect_points.py

# 同时扫描热门仓库中的成员 PR/讨论
HF_TOKEN=$HF_TOKEN python collect_points.py --scan-external

# 仅扫描特定类型的仓库
HF_TOKEN=$HF_TOKEN python collect_points.py --scan-external --repo-type models
HF_TOKEN=$HF_TOKEN python collect_points.py --scan-external --repo-type models datasets

# 推送到 HF 数据集
HF_TOKEN=$HF_TOKEN python collect_points.py --scan-external --push-to-hub

# 自定义输出
python collect_points.py --output my_leaderboard.json --repo-id my-org/my-dataset
```

### 选项

| 标志 | 描述 |
|------|------|
| `--scan-external` | 扫描整个 Hub 上的热门仓库以获取成员活动 |
| `--repo-type` | 过滤外部扫描范围：`models`、`datasets`、`spaces` |
| `--push-to-hub` | 将结果推送到 HF 数据集 |
| `--repo-id` | 目标数据集仓库（默认：`hf-skills/hackers-leaderboard`） |
| `--output` | 本地 JSON 输出路径 |

### 运行应用

```bash
HF_TOKEN=$HF_TOKEN python app.py
```

## API

收集器扫描：
- 组织中的所有模型、数据集和空间
- 这些仓库上的所有讨论和 PR
- 讨论中的所有评论

结果保存为 JSONL 格式，便于数据集使用。

## 输出格式

```json
{
  "username": "user123",
  "total_points": 15,
  "discussions_opened": 3,
  "comments_made": 8,
  "prs_opened": 2,
  "repos_owned": 2
}
```

