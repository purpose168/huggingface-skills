<skills>

您可以在包含"SKILL.md"文件的目录中找到其他技能文档。

这些技能包括：
{{#skills}}
 - {{name}} -> "{{path}}/SKILL.md"
{{/skills}}

重要提示：当技能的描述与用户意图匹配或可能帮助完成任务时，您必须阅读SKILL.md文件。 

<available_skills>

{{#skills}}
{{name}}: `{{description}}`

{{/skills}}
</available_skills>

SKILL文件夹中引用的路径是相对于该技能目录的。例如，hf-datasets的`scripts/example.py`应引用为`hf-datasets/scripts/example.py`。 

</skills>
