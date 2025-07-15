# ice_active.py
# -------------------------------------------------------------------
# 冰山对价策略模块：主策略逻辑 + 回测循环
# -------------------------------------------------------------------

from typing import Optional
import pandas as pd

# 导入可复用模块
from market_data import Tick, Order
from backtest_engine import run_backtest

# === 冰山对价策略 ============================================================

class IcebergStrategy:
    """最简冰山追单：一次挂 slice_qty，未击中则 next‑tick 撤 → 追挂"""

    def __init__(self, side: str, total_qty: int, slice_qty: int):
        self.side   = side
        self.total  = total_qty         # 目标总量
        self.slice  = slice_qty         # 冰山块大小
        self.left   = total_qty         # 剩余未成交量
        self.next_id= 0                 # 自增订单号
        self.pending: Optional[Order] = None

    # --- 内部：生成新订单 ----------------------------------------------------
    def _new_order(self, price: float, ts: pd.Timestamp) -> Order:
        self.next_id += 1
        qty = min(self.slice, self.left)
        self.pending = Order(self.next_id, self.side, price, qty, ts)
        return self.pending

    # --- 策略接口 -----------------------------------------------------------
    def on_tick(self, tick: Tick) -> Optional[Order]:
        """若当前无挂单则按对手价挂新单"""
        if self.left == 0 or self.pending is not None:
            return None
        price = tick.ask if self.side == "buy" else tick.bid
        return self._new_order(price, tick.ts)

    def on_fill(self):
        """成交后扣减剩余量并清空挂单"""
        self.left -= self.pending.qty
        self.pending = None

    def chase(self, tick: Tick) -> Order:
        """撤单并按最新对手价追挂"""
        self.pending = None
        price = tick.ask if self.side == "buy" else tick.bid
        new_order = self._new_order(price, tick.ts)
        print(f"追单完成: {new_order.side} {new_order.qty}@{new_order.price}")
        return new_order

# === 冰山策略接口说明 ==========================================================
# 本策略实现以下接口，供回测引擎调用：
# - on_tick(tick) -> Optional[Order]  # 处理tick，可能返回新订单
# - on_fill()                         # 处理成交
# - chase(tick) -> Order             # 追单逻辑
# - pending: Optional[Order]         # 当前挂单属性
# - left: int                        # 剩余数量属性
# - side: str                        # 买卖方向属性

# === CLI 示例 =================================================================

if __name__ == "__main__":
    # 创建策略实例
    strategy = IcebergStrategy(
        side="buy",
        total_qty=200,
        slice_qty=5
    )
    
    # 使用通用回测引擎运行回测
    result = run_backtest(
        csv_path=r"C:\Users\justr\Desktop\SHFE.cu2510.0.2025-07-07 00_00_00.2025-07-09 00_00_00.csv",
        strategy=strategy,
        start_time="2025-07-07 09:30:00",  # 修改为CSV数据范围内的时间
        contract_multiplier=5  # 铜期货合约乘数为5吨/手
    )
    print(result)
