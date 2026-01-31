# 故障排除指南

Hugging Face Jobs 常见问题及解决方案。

## 身份验证问题

### 错误：401 Unauthorized

**症状：**
```
401 Client Error: Unauthorized for url: https://huggingface.co/api/...
```

**原因：**
- 作业中缺少令牌
- 令牌无效或已过期
- 令牌传递不正确

**解决方案：**
1. 在作业配置中添加 `secrets={"HF_TOKEN": "$HF_TOKEN"}`
2. 验证本地 `hf_whoami()` 可用
3. 重新登录：`hf auth login`
4. 检查令牌是否已过期

**验证：**
```python
# 在你的脚本中
import os
assert "HF_TOKEN" in os.environ, "HF_TOKEN 缺失！"
```

### 错误：403 Forbidden

**症状：**
```
403 Client Error: Forbidden for url: https://huggingface.co/api/...
```

**原因：**
- 令牌缺少所需权限
- 无访问私有仓库的权限
- 组织权限不足

**解决方案：**
1. 确保令牌具有写入权限
2. 在 https://huggingface.co/settings/tokens 检查令牌类型
3. 验证对目标仓库的访问权限
4. 如需要，使用组织令牌

### 错误：环境中未找到令牌

**症状：**
```
KeyError: 'HF_TOKEN'
ValueError: HF_TOKEN not found
```

**原因：**
- 作业配置中未传递 `secrets`
- 键名错误（应为 `HF_TOKEN`）
- 使用 `env` 而不是 `secrets`

**解决方案：**
1. 使用 `secrets={"HF_TOKEN": "$HF_TOKEN"}`（而非 `env`）
2. 验证键名确实是 `HF_TOKEN`
3. 检查作业配置语法

## 作业执行问题

### 错误：作业超时

**症状：**
- 作业意外停止
- 状态显示 "TIMEOUT"
- 仅获得部分结果

**原因：**
- 超过默认 30 分钟超时时间
- 作业耗时超出预期
- 未指定超时时间

**解决方案：**
1. 检查日志中的实际运行时间
2. 增加超时时间并预留缓冲：`"timeout": "3h"`
3. 优化代码以加快执行速度
4. 分块处理数据
5. 在估计时间上增加 20-30% 的缓冲

**MCP 工具示例：**
```python
hf_jobs("uv", {
    "script": "...",
    "timeout": "2h"  # 设置适当的超时时间
})
```

**Python API 示例：**
```python
from huggingface_hub import run_uv_job, inspect_job, fetch_job_logs

job = run_uv_job("script.py", timeout="4h")

# 检查作业是否失败
job_info = inspect_job(job_id=job.id)
if job_info.status.stage == "ERROR":
    print(f"作业失败：{job_info.status.message}")
    # 检查日志获取详细信息
    for log in fetch_job_logs(job_id=job.id):
        print(log)
```

### 错误：内存不足 (OOM)

**症状：**
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate array
```

**原因：**
- 批量大小过大
- 模型过大，不适合硬件
- GPU 内存不足

**解决方案：**
1. 减小批量大小
2. 分小块处理数据
3. 升级硬件：cpu → t4 → a10g → a100
4. 使用更小的模型或量化
5. 启用梯度检查点（用于训练）

**示例：**
```python
# 减小批量大小
batch_size = 1

# 分块处理
for chunk in chunks:
    process(chunk)
```

### 错误：缺少依赖项

**症状：**
```
ModuleNotFoundError: No module named 'package_name'
ImportError: cannot import name 'X'
```

**原因：**
- 包不在依赖项列表中
- 包名称错误
- 版本不匹配

**解决方案：**
1. 添加到 PEP 723 头部：
   ```python
   # /// script
   # dependencies = ["package-name>=1.0.0"]
   # ///
   ```
2. 检查包名称拼写
3. 如需要，指定版本
4. 检查包是否可用

### 错误：找不到脚本

**症状：**
```
FileNotFoundError: script.py not found
```

**原因：**
- 使用了本地文件路径（不支持）
- URL 不正确
- 脚本不可访问

**解决方案：**
1. 使用内联脚本（推荐）
2. 使用公开可访问的 URL
3. 先将脚本上传到 Hub
4. 检查 URL 是否正确

**正确的方法：**
```python
# ✅ 内联代码
hf_jobs("uv", {"script": "# /// script\n# dependencies = [...]\n# ///\n\n<code>"})

# ✅ 从 URL 获取
hf_jobs("uv", {"script": "https://huggingface.co/user/repo/resolve/main/script.py"})
```

## Hub 推送问题

### 错误：推送失败

**症状：**
```
Error pushing to Hub
Upload failed
```

**原因：**
- 网络问题
- 令牌缺失或无效
- 仓库访问被拒绝
- 文件过大

**解决方案：**
1. 检查令牌：`assert "HF_TOKEN" in os.environ`
2. 验证仓库存在或可以创建
3. 在日志中检查网络连接
4. 重试推送操作
5. 将大文件分块推送

### 错误：仓库未找到

**症状：**
```
404 Client Error: Not Found
Repository not found
```

**原因：**
- 仓库不存在
- 仓库名称错误
- 无访问私有仓库的权限

**解决方案：**
1. 先创建仓库：
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.create_repo("username/repo-name", repo_type="dataset")
   ```
2. 检查仓库名称格式
3. 验证命名空间存在
4. 检查仓库可见性

### 错误：结果未保存

**症状：**
- 作业成功完成
- Hub 上看不到结果
- 文件未持久化

**原因：**
- 脚本中没有持久化代码
- 推送代码未执行
- 推送静默失败

**解决方案：**
1. 在脚本中添加持久化代码
2. 验证推送是否成功执行
3. 检查日志中的推送错误
4. 在推送周围添加错误处理

**示例：**
```python
try:
    dataset.push_to_hub("username/dataset")
    print("✅ 推送成功")
except Exception as e:
    print(f"❌ 推送失败：{e}")
    raise
```

## 硬件问题

### 错误：GPU 不可用

**症状：**
```
CUDA not available
No GPU found
```

**原因：**
- 使用了 CPU 口味而非 GPU
- 未请求 GPU
- 镜像中未安装 CUDA

**解决方案：**
1. 使用 GPU 口味：`"flavor": "a10g-large"`
2. 检查镜像是否支持 CUDA
3. 在日志中验证 GPU 可用性

### 错误：性能缓慢

**症状：**
- 作业耗时超出预期
- GPU 利用率低
- CPU 瓶颈

**原因：**
- 选择了错误的硬件
- 代码效率低
- 数据加载瓶颈

**解决方案：**
1. 升级硬件
2. 优化代码
3. 使用批量处理
4. 分析代码找出瓶颈

## 常见问题

### 错误：作业状态未知

**症状：**
- 无法检查作业状态
- 状态 API 返回错误

**解决方案：**
1. 使用作业 URL：`https://huggingface.co/jobs/username/job-id`
2. 检查日志：`hf_jobs("logs", {"job_id": "..."})`
3. 检查作业：`hf_jobs("inspect", {"job_id": "..."})`

### 错误：日志不可用

**症状：**
- 看不到日志
- 日志延迟

**原因：**
- 作业刚刚启动（日志延迟 30-60 秒）
- 作业在记录日志前失败
- 日志尚未生成

**解决方案：**
1. 作业启动后等待 30-60 秒
2. 先检查作业状态
3. 使用作业 URL 访问网页界面

### 错误：成本意外过高

**症状：**
- 作业成本超出预期
- 运行时间比估计长

**原因：**
- 作业运行时间超过超时时间
- 选择了错误的硬件
- 代码效率低

**解决方案：**
1. 监控作业运行时间
2. 设置适当的超时时间
3. 优化代码
4. 选择正确的硬件
5. 运行前检查成本估算

## 调试技巧

### 1. 添加日志记录

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("开始处理...")
logger.info(f"已处理 {count} 项")
```

### 2. 验证环境

```python
import os
print(f"Python 版本：{os.sys.version}")
print(f"CUDA 可用：{torch.cuda.is_available()}")
print(f"HF_TOKEN 存在：{'HF_TOKEN' in os.environ}")
```

### 3. 先在本地测试

在提交之前先在本地运行脚本，以尽早发现错误：
```bash
python script.py
# 或使用 uv
uv run script.py
```

### 4. 检查作业日志

**MCP 工具：**
```python
# 查看日志
hf_jobs("logs", {"job_id": "your-job-id"})
```

**CLI：**
```bash
hf jobs logs <job-id>
```

**Python API：**
```python
from huggingface_hub import fetch_job_logs
for log in fetch_job_logs(job_id="your-job-id"):
    print(log)
```

**或使用作业 URL：** `https://huggingface.co/jobs/username/job-id`

### 5. 添加错误处理

```python
try:
    # 你的代码
    process_data()
except Exception as e:
    print(f"错误：{e}")
    import traceback
    traceback.print_exc()
    raise
```

### 6. 以编程方式检查作业状态

```python
from huggingface_hub import inspect_job, fetch_job_logs

job_info = inspect_job(job_id="your-job-id")
print(f"状态：{job_info.status.stage}")
print(f"消息：{job_info.status.message}")

if job_info.status.stage == "ERROR":
    print("作业失败！日志：")
    for log in fetch_job_logs(job_id=job.id):
        print(log)
```

## 快速参考

### 常见错误代码

| 代码 | 含义 | 解决方案 |
|------|------|----------|
| 401 | 未授权 | 添加 `secrets={"HF_TOKEN": "$HF_TOKEN"}` |
| 403 | 禁止 | 检查令牌权限 |
| 404 | 未找到 | 验证仓库存在 |
| 500 | 服务器错误 | 重试或联系支持 |

### 提交前检查清单

- [ ] 令牌配置：`secrets={"HF_TOKEN": "$HF_TOKEN"}`
- [ ] 脚本检查令牌：`assert "HF_TOKEN" in os.environ`
- [ ] 超时时间设置适当
- [ ] 硬件选择正确
- [ ] 依赖项列在 PEP 723 头部中
- [ ] 包含持久化代码
- [ ] 添加了错误处理
- [ ] 添加了日志记录用于调试

## 获取帮助

如果问题持续存在：

1. **检查日志** - 大多数错误包含详细消息
2. **查看文档** - 参见主要 SKILL.md
3. **检查 Hub 状态** - https://status.huggingface.co
4. **社区论坛** - https://discuss.huggingface.co
5. **GitHub 问题** - 对于 huggingface_hub 中的 bug

## 关键要点

1. **始终包含令牌** - `secrets={"HF_TOKEN": "$HF_TOKEN"}`
2. **设置适当的超时时间** - 默认 30 分钟可能不足
3. **验证持久化** - 没有代码结果不会持久化
4. **检查日志** - 大多数问题在作业日志中可见
5. **在本地测试** - 提交前捕获错误
6. **添加错误处理** - 更好的调试信息
7. **监控成本** - 设置超时时间以避免意外费用