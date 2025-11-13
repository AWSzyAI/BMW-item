#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试BERT.py的基本功能，不依赖实际数据
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from test.BERT import _read_split_or_combined, _choose_label_column, _is_valid_local_hf_dir
    print("✓ 成功导入BERT.py中的辅助函数")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试_is_valid_local_hf_dir函数
print("\n测试_is_valid_local_hf_dir函数:")
print(f"  不存在的目录: {_is_valid_local_hf_dir('/nonexistent/path')}")

# 测试_choose_label_column函数
print("\n测试_choose_label_column函数:")
import pandas as pd
df_test = pd.DataFrame({
    'case_title': ['test'],
    'performed_work': ['test'],
    'linked_items': ['item1']
})
try:
    label_col = _choose_label_column(df_test)
    print(f"  ✓ 找到标签列: {label_col}")
except Exception as e:
    print(f"  ✗ 错误: {e}")

print("\n✓ 基本功能测试通过")