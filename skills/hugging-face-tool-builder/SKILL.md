---
name: hugging-face-tool-builder
description: 当用户想要构建工具/脚本或完成任务，而使用Hugging Face API的数据会有所帮助时，请使用此技能。这在链接或组合API调用或任务需要重复/自动化时特别有用。此技能创建一个可重用的脚本来获取、丰富或处理数据。
---

# Hugging Face API 工具构建器

您的目的是创建可重用的命令行脚本和实用程序，用于使用Hugging Face API，允许在有帮助的地方进行链接、管道传输和中间处理。您可以直接访问API，也可以使用`hf`命令行工具。可以直接从仓库访问模型和数据集卡片。

## 脚本规则

确保遵循以下规则：
- 脚本必须接受`--help`命令行参数来描述其输入和输出
- 非破坏性脚本应在交给用户之前进行测试
- 优先使用Shell脚本，但如果复杂性或用户需要，可以使用Python或TSX
- 重要：使用`HF_TOKEN`环境变量作为授权标头。例如：`curl -H "Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/api/`。这提供了更高的速率限制和适当的数据访问授权。
- 在确定最终设计之前调查API结果的形状；在可组合性有益的地方利用管道和链接——尽可能选择简单的解决方案。
- 完成后分享使用示例。

如有疑问或需要澄清的地方，请务必确认用户偏好。

## 示例脚本

以下路径相对于此技能目录。

参考示例：
- `references/hf_model_papers_auth.sh` — 自动使用`HF_TOKEN`，并链接趋势 → 模型元数据 → 模型卡片解析及回退；它演示了多步骤API使用以及 gated/私有内容的身份验证卫生。
- `references/find_models_by_paper.sh` — 通过`--token`可选使用`HF_TOKEN`，一致的经过身份验证的搜索，以及当arXiv前缀搜索太窄时的重试路径；它显示了弹性查询策略和清晰的用户面向帮助。
- `references/hf_model_card_frontmatter.sh` — 使用`hf` CLI下载模型卡片，提取YAML前置matter，并发出NDJSON摘要（许可证、pipeline标签、标签、gated提示标志）以便轻松过滤。

基线示例（超简单、最小逻辑、带有`HF_TOKEN`标头的原始JSON输出）：
- `references/baseline_hf_api.sh` — bash
- `references/baseline_hf_api.py` — python
- `references/baseline_hf_api.tsx` — typescript可执行文件

可组合实用程序（stdin → NDJSON）：
- `references/hf_enrich_models.sh` — 从stdin读取模型ID，获取每个ID的元数据，为流式管道每行发出一个JSON对象。

通过管道可组合（对shell友好的JSON输出）：
- `references/baseline_hf_api.sh 25 | jq -r '.[].id' | references/hf_enrich_models.sh | jq -s 'sort_by(.downloads) | reverse | .[:10]'`
- `references/baseline_hf_api.sh 50 | jq '[.[] | {id, downloads}] | sort_by(.downloads) | reverse | .[:10]'`
- `printf '%s\n' openai/gpt-oss-120b meta-llama/Meta-Llama-3.1-8B | references/hf_model_card_frontmatter.sh | jq -s 'map({id, license, has_extra_gated_prompt})'`

## 高级端点

以下是可用的主要API端点，位于`https://huggingface.co`

```
/api/datasets
/api/models
/api/spaces
/api/collections
/api/daily_papers
/api/notifications
/api/settings
/api/whoami-v2
/api/trending
/oauth/userinfo
```

## 访问API

API在`https://huggingface.co/.well-known/openapi.json`以OpenAPI标准记录。

**重要：** 不要尝试直接读取`https://huggingface.co/.well-known/openapi.json`，因为它太大无法处理。

**重要** 使用`jq`查询和提取相关部分。例如，

获取所有160个端点的命令

```bash
curl -s "https://huggingface.co/.well-known/openapi.json" | jq '.paths | keys | sort'
```

模型搜索端点详情

```bash
curl -s "https://huggingface.co/.well-known/openapi.json" | jq '.paths["/api/models"]'
```

您还可以查询端点以查看数据的形状。这样做时将结果限制在较小的数量以使其易于处理，同时具有代表性。

## 使用HF命令行工具

`hf`命令行工具让您进一步访问Hugging Face仓库内容和基础设施。

```bash
❯ hf --help
Usage: hf [OPTIONS] COMMAND [ARGS]...

  Hugging Face Hub CLI

Options:
  --help                Show this message and exit.

Commands:
  auth                 管理身份验证（登录、退出等）。
  cache                管理本地缓存目录。
  download             从Hub下载文件。
  endpoints            管理Hugging Face推理端点。
  env                  打印环境信息。
  jobs                 在Hub上运行和管理作业。
  repo                 管理Hub上的仓库。
  repo-files           管理Hub上仓库中的文件。
  upload               将文件或文件夹上传到Hub。
  upload-large-folder  将大文件夹上传到Hub。
  version              打印hf版本信息。
```

`hf` CLI命令已替换现已弃用的`huggingface_hub` CLI命令。