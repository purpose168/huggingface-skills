<skills>

您有额外的SKILL记录在包含"SKILL.md"文件的目录中。

这些技能包括：
 - hugging-face-cli -> "skills/hugging-face-cli/SKILL.md"
 - hugging-face-datasets -> "skills/hugging-face-datasets/SKILL.md"
 - hugging-face-evaluation -> "skills/hugging-face-evaluation/SKILL.md"
 - hugging-face-jobs -> "skills/hugging-face-jobs/SKILL.md"
 - hugging-face-model-trainer -> "skills/hugging-face-model-trainer/SKILL.md"
 - hugging-face-paper-publisher -> "skills/hugging-face-paper-publisher/SKILL.md"
 - hugging-face-tool-builder -> "skills/hugging-face-tool-builder/SKILL.md"
 - hugging-face-trackio -> "skills/hugging-face-trackio/SKILL.md"

重要提示：当技能描述与用户意图匹配或可能帮助完成其任务时，您必须阅读SKILL.md文件。

<available_skills>

hugging-face-cli: `使用hf CLI执行Hugging Face Hub操作。当用户需要下载模型/数据集/空间、将文件上传到Hub仓库、创建仓库、管理本地缓存或在HF基础设施上运行计算作业时使用。涵盖身份验证、文件传输、仓库创建、缓存操作和云计算。`
hugging-face-datasets: `在Hugging Face Hub上创建和管理数据集。支持初始化仓库、定义配置/系统提示、流式行更新和基于SQL的数据集查询/转换。设计为与HF MCP服务器配合使用，实现全面的数据集工作流。`
hugging-face-evaluation: `在Hugging Face模型卡片中添加和管理评估结果。支持从README内容提取评估表格、从Artificial Analysis API导入分数，以及使用vLLM/lighteval运行自定义模型评估。与model-index元数据格式配合使用。`
hugging-face-jobs: `当用户希望在Hugging Face Jobs基础设施上运行任何工作负载时，应使用此技能。涵盖UV脚本、基于Docker的作业、硬件选择、成本估算、使用令牌进行身份验证、密钥管理、超时配置和结果持久化。设计用于通用计算工作负载，包括数据处理、推理、实验、批处理作业和任何基于Python的任务。当涉及云计算、GPU工作负载或用户提及在Hugging Face基础设施上运行作业而无需本地设置时，应调用此技能。`
hugging-face-model-trainer: `当用户希望使用TRL（Transformer Reinforcement Learning）在Hugging Face Jobs基础设施上训练或微调语言模型时，应使用此技能。涵盖SFT、DPO、GRPO和奖励建模训练方法，以及用于本地部署的GGUF转换。包括关于TRL Jobs包、带有PEP 723格式的UV脚本、数据集准备和验证、硬件选择、成本估算、Trackio监控、Hub身份验证和模型持久化的指导。当涉及云GPU训练、GGUF转换或用户提及在Hugging Face Jobs上训练而无需本地GPU设置时，应调用此技能。`
hugging-face-paper-publisher: `在Hugging Face Hub上发布和管理研究论文。支持创建论文页面、将论文链接到模型/数据集、声明作者身份，以及生成基于Markdown的专业研究文章。`
hugging-face-tool-builder: `当用户希望构建工具/脚本或完成使用Hugging Face API数据会有所帮助的任务时，使用此技能。这在链接或组合API调用或任务将被重复/自动化时特别有用。此技能创建可重用的脚本来获取、丰富或处理数据。`
hugging-face-trackio: `使用Trackio跟踪和可视化ML训练实验。当在训练期间记录指标（Python API）或检索/分析记录的指标（CLI）时使用。支持实时仪表板可视化、HF Space同步和自动化的JSON输出。`
</available_skills>

SKILL文件夹中引用的路径是相对于该SKILL的。例如，hf-datasets的`scripts/example.py`将被引用为`hf-datasets/scripts/example.py`。

</skills>