#!/usr/bin/env tsx

/**
 * 超简单的 Hugging Face API 示例（TSX）。
 *
 * 从 HF API 获取少量模型列表并打印原始 JSON 数据。
 * 如果设置了环境变量 HF_TOKEN，则使用该令牌进行身份验证。
 */

// 显示帮助信息的函数
const showHelp = () => {
  console.log(`超简单的 Hugging Face API 示例（TSX）

使用方法:
  baseline_hf_api.tsx [limit]
  baseline_hf_api.tsx --help

描述:
  从 HF API 获取少量模型列表并打印原始 JSON 数据。
  如果设置了环境变量 HF_TOKEN，则使用该令牌进行身份验证。

示例:
  baseline_hf_api.tsx
  baseline_hf_api.tsx 5
  HF_TOKEN=your_token baseline_hf_api.tsx 10
`);
};

// 获取命令行参数（第二个参数）
const arg = process.argv[2];
// 如果参数是 --help，则显示帮助信息并退出
if (arg === "--help") {
  showHelp();
  process.exit(0);
}

// 设置获取模型数量的限制，默认为 3
const limit = arg ?? "3";
// 验证 limit 参数是否为有效数字
if (!/^\d+$/.test(limit)) {
  console.error("错误：limit 必须是一个数字");
  process.exit(1);
}

// 从环境变量中获取 Hugging Face 访问令牌
const token = process.env.HF_TOKEN;
// 如果存在令牌，则添加到请求头中进行身份验证
const headers: Record<string, string> = token
  ? { Authorization: `Bearer ${token}` }
  : {};

// 构建 Hugging Face API 请求 URL
const url = `https://huggingface.co/api/models?limit=${limit}`;

// 异步执行主逻辑
(async () => {
  // 发送 HTTP GET 请求到 Hugging Face API
  const res = await fetch(url, { headers });

  // 检查响应状态码，如果请求失败则输出错误信息并退出
  if (!res.ok) {
    console.error(`错误：${res.status} ${res.statusText}`);
    process.exit(1);
  }

  // 获取响应文本内容
  const text = await res.text();
  // 将原始 JSON 数据输出到标准输出
  process.stdout.write(text);
})();
