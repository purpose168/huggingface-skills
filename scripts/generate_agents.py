#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""根据AGENTS_TEMPLATE.md和SKILL.md前言生成AGENTS.md文件。

同时验证marketplace.json是否与发现的技能同步，
并更新README.md中的技能表格。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "scripts" / "AGENTS_TEMPLATE.md"
OUTPUT_PATH = ROOT / "agents" / "AGENTS.md"
MARKETPLACE_PATH = ROOT / ".claude-plugin" / "marketplace.json"
README_PATH = ROOT / "README.md"

# Markers for the auto-generated skills table in README
README_TABLE_START = "<!-- BEGIN_SKILLS_TABLE -->"
README_TABLE_END = "<!-- END_SKILLS_TABLE -->"


def load_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def parse_frontmatter(text: str) -> dict[str, str]:
    """解析最小化的YAML风格前言块，不依赖外部依赖库。"""
    match = re.search(r"^---\s*\n(.*?)\n---\s*", text, re.DOTALL)
    if not match:
        return {}
    data: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def collect_skills() -> list[dict[str, str]]:
    """收集所有技能信息，从每个skills目录下的SKILL.md文件读取元数据。"""
    skills: list[dict[str, str]] = []
    for skill_md in ROOT.glob("skills/*/SKILL.md"):
        meta = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = meta.get("name")
        description = meta.get("description")
        if not name or not description:
            continue
        skills.append(
            {
                "name": name,
                "description": description,
                "path": str(skill_md.parent.relative_to(ROOT)),
            }
        )
    # 保持确定性顺序以获得一致的输出
    return sorted(skills, key=lambda s: s["name"].lower())


def render(template: str, skills: list[dict[str, str]]) -> str:
    """非常简单的Mustache模板渲染器，仅支持单个技能循环。"""
    def repl(match: re.Match[str]) -> str:
        block = match.group(1).strip("\n")
        rendered_blocks = []
        for skill in skills:
            rendered = (
                block.replace("{{name}}", skill["name"])
                .replace("{{description}}", skill["description"])
                .replace("{{path}}", skill["path"])
            )
            rendered_blocks.append(rendered)
        return "\n".join(rendered_blocks)

    # 渲染循环块
    content = re.sub(r"{{#skills}}(.*?){{/skills}}", repl, template, flags=re.DOTALL)
    return content


def load_marketplace() -> dict:
    """加载marketplace.json并返回解析后的结构。"""
    if not MARKETPLACE_PATH.exists():
        raise FileNotFoundError(f"marketplace.json未在{MARKETPLACE_PATH}找到")
    return json.loads(MARKETPLACE_PATH.read_text(encoding="utf-8"))


def generate_readme_table(skills: list[dict[str, str]]) -> str:
    """使用marketplace.json中的名称生成README.md的技能表格。"""
    marketplace = load_marketplace()
    plugins = {p["source"]: p for p in marketplace.get("plugins", [])}

    lines = [
        "| 名称 | 描述 | 文档 |",
        "|------|-------------|---------------|",
    ]

    for skill in skills:
        source = f"./{skill['path']}"
        plugin = plugins.get(source, {})
        name = plugin.get("name", skill["name"])
        description = plugin.get("description", skill["description"])
        doc_link = f"[SKILL.md]({skill['path']}/SKILL.md)"
        lines.append(f"| `{name}` | {description} | {doc_link} |")

    return "\n".join(lines)


def update_readme(skills: list[dict[str, str]]) -> bool:
    """
    更新README.md中标记之间的技能表格。
    如果文件已更新返回True，如果未找到标记返回False。
    """
    if not README_PATH.exists():
        print(f"警告：README.md未在{README_PATH}找到", file=sys.stderr)
        return False

    content = README_PATH.read_text(encoding="utf-8")

    start_idx = content.find(README_TABLE_START)
    end_idx = content.find(README_TABLE_END)

    if start_idx == -1 or end_idx == -1:
        print(
            f"警告：未找到README.md标记。添加{README_TABLE_START}和"
            f"{README_TABLE_END}以启用表格生成。",
            file=sys.stderr,
        )
        return False

    if end_idx < start_idx:
        print("警告：README.md标记顺序错误。", file=sys.stderr)
        return False

    table = generate_readme_table(skills)
    new_content = (
        content[: start_idx + len(README_TABLE_START)]
        + "\n"
        + table
        + "\n"
        + content[end_idx:]
    )

    README_PATH.write_text(new_content, encoding="utf-8")
    return True


def validate_marketplace(skills: list[dict[str, str]]) -> list[str]:
    """
    验证marketplace.json与发现的技能是否一致。
    返回错误消息列表（空列表=通过）。
    """
    errors: list[str] = []
    marketplace = load_marketplace()
    plugins = marketplace.get("plugins", [])

    # 构建查找表（规范化路径：skill使用"skills/x"，marketplace使用"./skills/x"）
    skill_by_source = {f"./{s['path']}": s for s in skills}
    plugin_by_source = {p["source"]: p for p in plugins}

    # 检查：每个技能在marketplace中都有匹配的条目
    for skill in skills:
        expected_source = f"./{skill['path']}"
        if expected_source not in plugin_by_source:
            errors.append(
                f"技能'{skill['name']}'位于'{skill['path']}'，在marketplace.json中缺失"
            )
        elif plugin_by_source[expected_source]["name"] != skill["name"]:
            errors.append(
                f"路径'{expected_source}'名称不匹配："
                f"SKILL.md='{skill['name']}'，marketplace.json='{plugin_by_source[expected_source]['name']}'"
            )

    # 检查：每个marketplace插件都有对应的技能
    for plugin in plugins:
        if plugin["source"] not in skill_by_source:
            errors.append(
                f"Marketplace插件'{plugin['name']}'位于'{plugin['source']}'没有对应的SKILL.md"
            )

    return errors


def main() -> None:
    """主函数：加载模板、收集技能、生成输出并验证。"""
    template = load_template()
    skills = collect_skills()
    output = render(template, skills)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"已写入{OUTPUT_PATH}，包含{len(skills)}个技能。")

    # 验证marketplace.json
    errors = validate_marketplace(skills)
    if errors:
        print("\nMarketplace.json验证错误：", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
    print("Marketplace.json验证通过。")

    # 更新README.md技能表格
    if update_readme(skills):
        print(f"已更新{README_PATH}技能表格。")


if __name__ == "__main__":
    main()
