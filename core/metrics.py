# metrics.py
# -------------------------------------------------------------------
# 指标收集模块：统一口径的性能评价指标
# -------------------------------------------------------------------

from typing import Optional
import pandas as pd
from .market_data import Tick
from .matching_engine import Fill

# === 指标收集器 ===============================================================

class Metrics:
    """统一口径：avg_fill、VWAP、slippage、duration、trade_count、order_count"""

    def __init__(self, contract_multiplier: int = 1):
        self.multiplier = contract_multiplier  # 合约乘数
        self.first_amt = None          # 起始累计成交额
        self.first_vol = None          # 起始累计成交量
        self.last_amt = 0.0            # 最新累计成交额
        self.last_vol = 0.0            # 最新累计成交量
        self.fill_val = 0.0            # 策略成交额累计
        self.fill_vol = 0              # 策略成交量累计
        self.trades   = 0              # 成交笔数
        self.orders   = 0              # 委托笔数
        self.first_order_ts: Optional[pd.Timestamp] = None  # 首次下单时间
        self.last_fill_ts: Optional[pd.Timestamp] = None    # 最后成交时间

    def on_tick(self, tick: Tick):
        # 记录第一个tick作为基准点
        if self.first_amt is None:
            self.first_amt = tick.amt
            self.first_vol = tick.vol
        # 更新最新值
        self.last_amt = tick.amt
        self.last_vol = tick.vol

    def on_order(self, order_ts: pd.Timestamp = None):
        """记录委托下单"""
        self.orders += 1
        # 记录首次下单时间
        if self.first_order_ts is None and order_ts is not None:
            self.first_order_ts = order_ts

    def on_fill(self, fill: Fill):
        # 更新最后成交时间
        self.last_fill_ts = fill.ts
        self.fill_val += fill.price * fill.qty
        self.fill_vol += fill.qty
        self.trades   += 1

    def summary(self, side: str) -> dict:
        avg  = self.fill_val / self.fill_vol if self.fill_vol else None
        # 计算期间的增量成交额和成交量
        period_amt = self.last_amt - self.first_amt if self.first_amt is not None else 0
        period_vol = self.last_vol - self.first_vol if self.first_vol is not None else 0
        # 期货VWAP计算：需要考虑合约乘数
        # VWAP(元/单位) = 成交额(元) / (成交量(手) * 合约乘数(单位/手))
        vwap = period_amt / (period_vol * self.multiplier) if period_vol > 0 else None
        

        
        slip = (avg - vwap) if (avg and vwap and side == "buy") else \
               (vwap - avg) if (avg and vwap and side == "sell") else None
        # 计算从首次下单到最后成交的市场时间
        dur = (self.last_fill_ts - self.first_order_ts).total_seconds() if (self.first_order_ts and self.last_fill_ts) else None
        
        # 返回中文指标结果
        return {
            "平均成交价格": avg,           # 策略实际成交的平均价格
            "市场VWAP": vwap,             # 同期市场成交量加权平均价格
            "价格滑点": slip,             # 相对于市场VWAP的滑点成本
            "执行时长(秒)": dur,          # 从首次成交到最后成交的时间
            "成交笔数": self.trades,      # 总成交次数
            "委托笔数": self.orders       # 总委托次数
        } 