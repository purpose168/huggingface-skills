#!/usr/bin/env python3
"""执行skills目录下所有Python程序并显示帮助信息的脚本。"""

import subprocess
from pathlib import Path

def find_python_files():
    """查找skills目录下所有Python文件。"""
    python_files = []
    
    # 搜索skills目录并查找Python文件
    for skills_dir in Path('.').rglob('../skills'):
        if skills_dir.is_dir():
            python_files.extend(skills_dir.rglob('*.py'))
    
    return sorted(set(python_files))

def run_with_help(python_file):
    """使用uv run --help运行Python文件。"""
    try:
        print(f"\n{'='*60}")
        print(f"运行中：{python_file}")
        print(f"{'='*60}")
        
        result = subprocess.run(
            ['uv', 'run', str(python_file), '--help'],
            capture_output=True,
            text=True,
            timeout=30  # 30秒超时
        )
        
        if result.returncode == 0:
            print("成功 - 输出：")
            print(result.stdout)
        else:
            print(f"失败 - 返回码：{result.returncode}")
            if result.stderr:
                print("标准错误：")
                print(result.stderr)
            if result.stdout:
                print("标准输出：")
                print(result.stdout)
                
    except subprocess.TimeoutExpired:
        print("超时 - 命令执行时间过长")
    except FileNotFoundError:
        print("错误 - 未找到uv命令。请安装uv。")
        return False
    except Exception as e:
        print(f"错误 - {str(e)}")
    
    return True

def main():
    """主函数：查找并运行所有Python文件。"""
    print("正在查找skills目录下的Python文件...")
    python_files = find_python_files()
    
    if not python_files:
        print("未在skills目录下找到Python文件。")
        return
    
    print(f"找到{len(python_files)}个Python文件")
    
    success_count = 0
    failed_count = 0
    
    for python_file in python_files:
        if run_with_help(python_file):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"摘要：")
    print(f"总文件数：{len(python_files)}")
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
