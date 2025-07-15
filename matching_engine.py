# matching_engine.py
# -------------------------------------------------------------------
# 撮合引擎模块：通用撮合逻辑
# -------------------------------------------------------------------

from typing import Optional
from market_data import Order, Fill, Tick

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