#!/usr/bin/env python
"""检查语法错误"""

import ast
import sys

def check_file(filename):
    """检查文件语法"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        print(f"✅ {filename}: 语法正确")
        return True
    except SyntaxError as e:
        print(f"❌ {filename}: 语法错误")
        print(f"   行 {e.lineno}: {e.msg}")
        print(f"   {e.text}")
        return False
    except Exception as e:
        print(f"⚠️  {filename}: 其他错误 - {e}")
        return False

if __name__ == "__main__":
    files = [
        "memory_system/core/logger.py",
        "memory_system/vector/reranker.py"
    ]

    all_good = True
    for f in files:
        if not check_file(f):
            all_good = False

    sys.exit(0 if all_good else 1)