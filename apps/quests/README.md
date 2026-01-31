---
title: README
emoji: 🐠
colorFrom: yellow
colorTo: gray
sdk: static
pinned: false
---

# 人类最后的黑客马拉松 (2025)

<img src="https://github.com/huggingface/skills/raw/main/assets/banner.png" alt="人类最后的黑客马拉松 (2025)" width="100%">

欢迎参加我们的黑客马拉松！

无论你是经验丰富的机器学习工程师、传统的NLP开发者，还是充满AGI热情的程序员，这次黑客马拉松都将是一项艰巨的工作！我们将使用最新、最强大的编码代理来提升开源AI的水平。毕竟，**为什么要用十二月来放松和与亲人共度时光，而不是为全人类解决AI问题呢？** 开玩笑归开玩笑，这次黑客马拉松不是从零开始学习技能或将事情分解为最简单的组件。而是关于协作、交付，以及为开源社区做出贡献。

## 我们正在构建什么

在四周的时间里，我们将使用编码代理来提升开源AI生态系统：

- **第1周** — 评估模型并构建分布式排行榜
- **第2周** — 为社区创建高质量数据集  
- **第3周** — 在Hub上微调并分享模型
- **第4周** — 一起冲刺到终点线

每一次贡献都能获得XP。贡献最多的人将登上排行榜。获胜者将获得奖品！

以下是日程安排：

| 日期 | 事件 | 链接 |
|------|-------|------|
| 12月2日（周一） | 第1周任务发布 | [评估Hub模型](02_evaluate-hub-model.md) |
| 12月4日（周三） | 直播1 | [问答1](https://youtube.com/live/rworGSh-Rgk?feature=share) |
| 12月9日（周一） | 第2周任务发布 | [发布Hub数据集](03_publish-hub-dataset.md) |
| 12月11日（周三） | 直播2 | 待定 |
| 12月16日（周一） | 第3周任务发布 | [监督微调](04_sft-finetune-hub.md) |
| 12月18日（周三） | 直播3 | 待定 |
| 12月23日（周一） | 第4周社区冲刺 | 待定 |
| 12月31日（周二） | 黑客马拉松结束 | 待定

## 开始参与

### 1. 加入组织

加入 Hugging Face 上的 [hf-skills](https://huggingface.co/organizations/hf-skills/share/KrqrmBxkETjvevFbfkXeezcyMbgMjjMaOp)。你的贡献将在这里被跟踪并更新到排行榜上。

### 2. 设置你的编码代理

使用你喜欢的任何编码代理：

- **Claude Code** — 在终端中使用 `claude`
- **Codex** — `codex` CLI
- **Gemini CLI** — 在终端中使用 `gemini`
- **Cursor / Windsurf** — 基于IDE的代理
- **开源** — aider, continue 等

此仓库中的技能可与任何能够读取markdown指令并运行Python脚本的代理配合使用。要安装这些技能，请按照 [README](../README.md) 中的说明进行操作。

### 3. 获取你的 HF 令牌

大多数任务需要具有写入权限的 Hugging Face 令牌：

```bash
# mac/linux
curl -LsSf https://hf.co/cli/install.sh | bash

# windows
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# 登录（创建/存储你的令牌）
hf auth login
```

这将设置你的 `HF_TOKEN` 环境变量。

### 4. 克隆技能仓库

```bash
git clone https://github.com/huggingface/skills.git
cd skills
```

将你的编码代理指向相关配置。查看 [README](../README.md) 了解如何将这些技能与你的编码代理一起使用。

## 你的第一个任务

**第1周已开始！** 前往 [02_evaluate-hub-model.md](02_evaluate-hub-model.md) 开始评估模型并攀登排行榜。

<iframe
	src="https://hf-skills-hacker-leaderboard.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

[排行榜](https://hf-skills-hacker-leaderboard.hf.space)

## 获得 XP

每个任务有三个等级：

| 等级 | 要求 | XP |
|------|---------------|-----|
| 🐢 | 完成基础内容 | 50-75 XP |
| 🐕 | 深入学习更多功能 | 100-125 XP |
| 🦁 | 交付令人印象深刻的成果 | 200-225 XP |

你可以完成多个等级，也可以使用不同的模型/数据集/空间多次完成同一个任务。

## 获取帮助

- [Discord](https://discord.com/channels/879548962464493619/1442881667986624554) — 加入 Hugging Face Discord 获取实时帮助
- [直播](https://www.youtube.com/@HuggingFace/streams) — 每周直播，包含演练和问答
- [Issues](https://github.com/huggingface/skills/issues) — 如果你遇到困难，请在本仓库中打开一个issue

要加入黑客马拉松，请在hub上加入组织并设置你的编码代理。

准备好了吗？让我们一起交付一些AI成果吧。 🚀