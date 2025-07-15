# metrics.py
# -------------------------------------------------------------------
# 指标收集模块：统一口径的性能评价指标
# -------------------------------------------------------------------

from typing import Optional
import pandas as pd
from market_data import Fill, Tick

# === 指标收集器 ===============================================================

class Metrics:
    """统一口径：avg_fill、VWAP、slippage、duration、trade_count"""

    def __init__(self, contract_multiplier: int = 1):
        self.multiplier = contract_multiplier  # 合约乘数
        self.first_amt = None          # 起始累计成交额
        self.first_vol = None          # 起始累计成交量
        self.last_amt = 0.0            # 最新累计成交额
        self.last_vol = 0.0            # 最新累计成交量
        self.fill_val = 0.0            # 策略成交额累计
        self.fill_vol = 0              # 策略成交量累计
        self.trades   = 0              # 成交笔数
        self.start_ts: Optional[pd.Timestamp] = None
        self.end_ts  : Optional[pd.Timestamp] = None

    def on_tick(self, tick: Tick):
        # 记录第一个tick作为基准点
        if self.first_amt is None:
            self.first_amt = tick.amt
            self.first_vol = tick.vol
            print(f"首次tick: 价格={tick.last}, 成交量={tick.vol}, 成交额={tick.amt}")
        # 更新最新值
        self.last_amt = tick.amt
        self.last_vol = tick.vol
        
        # 每100个tick打印一次调试信息
        if hasattr(self, '_tick_count'):
            self._tick_count += 1
        else:
            self._tick_count = 1
            


    def on_fill(self, fill: Fill):
        if self.start_ts is None:
            self.start_ts = fill.ts
        self.end_ts  = fill.ts
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
        dur  = (self.end_ts - self.start_ts).total_seconds() if self.start_ts else None
        return dict(avg_fill_price=avg,
                    market_vwap=vwap,
                    slippage=slip,
                    duration_sec=dur,
                    trade_count=self.trades) 