# matching_engine.py
# -------------------------------------------------------------------
# 撮合引擎模块：交易数据类型 + 撮合逻辑
# -------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional
import pandas as pd
from market_data import Tick

# === 交易数据类型 =============================================================

@dataclass
class Order:
    """委托单"""
    id:    int
    side:  str                    # 'buy' or 'sell'
    price: float
    qty:   int
    ts:    pd.Timestamp


@dataclass
class Fill:
    """成交记录"""
    order_id: int
    price:    float
    qty:      int
    ts:       pd.Timestamp

# === 通用撮合器 ===============================================================

def match(order: Order, tick: Tick) -> Optional[Fill]:
    """
    用 "下一 Tick 对手价" 判断整笔成交：
    * 买单对手价 = tick.ask
    * 卖单对手价 = tick.bid
    成交价记录为对手价；未成交返回 None
    """
    contra = tick.ask if order.side == "buy" else tick.bid       # 对手价
    hit = (order.side == "buy"  and contra <= order.price) or \
          (order.side == "sell" and contra >= order.price)
    if hit:
        return Fill(order.id, contra, order.qty, tick.ts)
    return None 