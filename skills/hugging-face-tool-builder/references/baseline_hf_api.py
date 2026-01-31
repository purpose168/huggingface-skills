#!/usr/bin/env python3
"""
超简单的 Hugging Face API 示例（Python）。

从 HF API 获取少量模型列表并打印原始 JSON 数据。
如果设置了环境变量 HF_TOKEN，则使用它进行身份验证。
"""

from __future__ import annotations

import os
import sys
import urllib.request


def show_help() -> None:
    """
    显示帮助信息。

    打印程序的使用说明、描述和示例。
    """
    print(
        """超简单的 Hugging Face API 示例（Python）

用法：
  baseline_hf_api.py [limit]
  baseline_hf_api.py --help

描述：
  从 HF API 获取少量模型列表并打印原始 JSON 数据。
  如果设置了环境变量 HF_TOKEN，则使用它进行身份验证。

示例：
  baseline_hf_api.py
  baseline_hf_api.py 5
  HF_TOKEN=your_token baseline_hf_api.py 10
"""
    )


def main() -> int:
    """
    主函数。

    解析命令行参数，从 Hugging Face API 获取模型列表并打印结果。

    返回:
        int: 程序退出码，0 表示成功，1 表示错误。

    流程:
        1. 检查是否需要显示帮助信息
        2. 解析 limit 参数（默认为 3）
        3. 获取 HF_TOKEN 环境变量（如果存在）
        4. 构建请求 URL 和请求头
        5. 发送 HTTP GET 请求并打印响应
    """
    # 检查是否请求帮助信息
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
        return 0

    # 解析 limit 参数，默认值为 "3"
    limit = sys.argv[1] if len(sys.argv) > 1 else "3"
    # 验证 limit 是否为有效数字
    if not limit.isdigit():
        print("错误：limit 必须是一个数字", file=sys.stderr)
        return 1

    # 从环境变量获取 Hugging Face 认证令牌（可选）
    token = os.getenv("HF_TOKEN")
    # 如果存在令牌，则添加 Authorization 请求头
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # 构建 API 请求 URL，limit 参数控制返回的模型数量
    url = f"https://huggingface.co/api/models?limit={limit}"

    # 创建 HTTP 请求对象，包含必要的请求头
    req = urllib.request.Request(url, headers=headers)
    # 发送请求并读取响应内容，将 JSON 数据输出到标准输出
    with urllib.request.urlopen(req) as resp:
        sys.stdout.write(resp.read().decode("utf-8"))
    return 0


if __name__ == "__main__":
    # 当脚本直接运行时，执行主函数并使用其返回值作为退出码
    raise SystemExit(main())
