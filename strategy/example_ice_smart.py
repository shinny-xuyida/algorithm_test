# strategy/example_ice_smart.py
# 冰山智能策略使用示例

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strategy import IceSmartStrategy
from backtest_engine import run_backtest

def main():
    """冰山智能策略基础示例"""
    print("=== 冰山智能策略回测示例 ===")
    print("策略特点：")
    print("- 根据盘口情况智能选择挂价或对价")
    print("- 盘口偏离且micro-price支持时使用对价")
    print("- 其他时间使用挂价策略")
    print("- 5档优化：自动使用5档数据进行更精确计算（如果可用）")
    print("- 如果没有5档数据，自动回退到1档计算")
    print()
    
    # 创建策略
    strategy = IceSmartStrategy(
        side="buy",              # 买入方向
        total_qty=200,           # 总目标量200手
        slice_qty=5,             # 每次挂单5手
        imbalance_threshold=0.2  # 盘口失衡阈值0.2
    )
    
    # 运行回测
    result = run_backtest(
        csv_path=r"C:\Users\justr\Desktop\SHFE.cu2510.0.2025-07-07 00_00_00.2025-07-09 00_00_00.csv",
        strategy=strategy,
        start_time="2025-07-07 14:00:00"
    )
    
    print("\n智能策略测试完成！")
    return result

if __name__ == "__main__":
    main() 