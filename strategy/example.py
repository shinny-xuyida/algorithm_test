# strategy/example.py
# -------------------------------------------------------------------
# 策略使用示例：展示如何使用各种策略进行回测
# -------------------------------------------------------------------

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strategy import IcebergStrategy
from backtest_engine import run_backtest

def run_iceberg_example():
    """运行冰山策略示例"""
    print("=" * 60)
    print("冰山策略回测示例")
    print("=" * 60)
    
    # CSV文件路径（用户需要根据实际情况修改）
    csv_path = r"C:\Users\justr\Desktop\SHFE.cu2510.0.2025-07-07 00_00_00.2025-07-09 00_00_00.csv"
    
    # 创建冰山策略实例
    strategy = IcebergStrategy(
        side="buy",          # 买入方向
        total_qty=200,       # 总目标量200手
        slice_qty=5          # 每次挂单5手
    )
    
    print(f"策略配置:")
    print(f"- 方向: {strategy.side}")
    print(f"- 总量: {strategy.total}手")
    print(f"- 每次挂单: {strategy.slice}手")
    print(f"- 预计分割次数: {strategy.total // strategy.slice}次")
    print()
    
    # 运行回测
    try:
        result = run_backtest(
            csv_path=csv_path,
            strategy=strategy,
            start_time="2025-07-07 09:30:00"
        )
        
        print("\n" + "=" * 60)
        print("回测结果:")
        print("=" * 60)
        for key, value in result.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: N/A")
        
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {csv_path}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"回测执行出错: {e}")

if __name__ == "__main__":
    run_iceberg_example() 