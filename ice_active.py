# ice_active.py
# -------------------------------------------------------------------
# 冰山策略示例：展示如何使用strategy包
# -------------------------------------------------------------------

from strategy import IcebergStrategy
from backtest_engine import run_backtest

# === CLI 示例 =================================================================

if __name__ == "__main__":
    # CSV文件路径
    csv_path = r"C:\Users\justr\Desktop\SHFE.cu2510.0.2025-07-07 00_00_00.2025-07-09 00_00_00.csv"
    
    # 创建策略实例
    strategy = IcebergStrategy(
        side="buy",
        total_qty=200,
        slice_qty=5
    )
    
    # 运行回测（合约乘数将自动从文件名推断）
    result = run_backtest(
        csv_path=csv_path,
        strategy=strategy,
        start_time="2025-07-07 09:30:00"
    )
    print(result)
