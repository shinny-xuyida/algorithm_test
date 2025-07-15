# ice_active.py
# -------------------------------------------------------------------
# 冰山对价策略模块：主策略逻辑 + 回测循环
# -------------------------------------------------------------------

from typing import Optional
import pandas as pd

# 导入可复用模块
from market_data import Tick, Order, tick_reader
from matching_engine import match
from metrics import Metrics

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

# === 主回测循环 ===============================================================

def run_backtest(csv_path: str,
                 side: str,
                 total_qty: int,
                 slice_qty: int,
                 start_time: str,
                 contract_multiplier: int = 1) -> dict:
    """
    核心循环：逐 Tick 驱动撮合 & 指标收集
    * csv_path           : 行情 CSV
    * side               : 'buy' / 'sell'
    * total_qty          : 总目标量
    * slice_qty          : 冰山块大小
    * start_time         : 起始时间 (字符串，可含日期)
    * contract_multiplier: 合约乘数，用于计算市场VWAP，默认为1
    """
    ticks   = tick_reader(csv_path)                   # 全自动 Tick 迭代器
    start_ts= pd.to_datetime(start_time)              # 起始时间
    strat   = IcebergStrategy(side, total_qty, slice_qty)
    metric  = Metrics(contract_multiplier)

    prev_tick: Optional[Tick] = None                  # 上一 Tick

    for tick in ticks:
        metric.on_tick(tick)                         # 统计市场 VWAP
        if tick.ts < start_ts:                       # 未到开始时间
            prev_tick = tick
            continue

        # --- 若有挂单，用 "下一 Tick"(当前 tick) 撮合 --------------------
        if strat.pending and prev_tick is not None:
            fill = match(strat.pending, tick)
            if fill:
                print(f"成交: {fill.qty}@{fill.price} at {fill.ts}")
                metric.on_fill(fill)
                strat.on_fill()                      # 剩余量扣减
            else:
                print(f"未成交，追单: 原价{strat.pending.price} -> 新价{tick.ask if strat.side=='buy' else tick.bid}")
                strat.chase(tick)                    # 未成交 → 追单

        # --- 让策略决定是否挂新单 ---------------------------------------
        new_order = strat.on_tick(tick)
        if new_order:
            print(f"新下单: {new_order.side} {new_order.qty}@{new_order.price} at {new_order.ts}")

        prev_tick = tick                             # 存起来做下轮对手价
        if strat.left == 0:                          # 全部成交 → 提前结束
            break

    return metric.summary(side)

# === CLI 示例 =================================================================

if __name__ == "__main__":
    # 修改为你的 CSV 路径与参数
    result = run_backtest(
        csv_path=r"C:\Users\justr\Desktop\SHFE.cu2510.0.2025-07-07 00_00_00.2025-07-09 00_00_00.csv",
        side="buy",
        total_qty=200,
        slice_qty=5,
        start_time="2025-07-07 09:30:00",  # 修改为CSV数据范围内的时间
        contract_multiplier=5  # 铜期货合约乘数为5吨/手
    )
    print(result)
