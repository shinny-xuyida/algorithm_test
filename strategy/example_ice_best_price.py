# strategy/example_ice_best_price.py
# 冰山策略使用示例

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strategy import IceBestStrategy
from backtest_engine import run_backtest

def basic_example():
    """基本使用示例"""
    # 创建策略
    strategy = IceBestStrategy(
        side="buy",
        total_qty=200,
        slice_qty=5
    )
    
    # 运行回测
    result = run_backtest(
        csv_path=r"C:\Users\justr\Desktop\SHFE.cu2510.0.2025-07-07 00_00_00.2025-07-09 00_00_00.csv",
        strategy=strategy,
        start_time="2025-07-07 14:00:00"
    )
    
    return result

if __name__ == "__main__":
    # 运行示例
    print(">>> 运行冰山策略示例 (买入 200手，每次5手)")
    basic_example() 