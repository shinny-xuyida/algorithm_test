# strategy/ice_hang.py
# 冰山挂价策略：将大单拆分成小块，以己方价格挂单被动等待成交

from typing import Optional
import pandas as pd

# 导入基础模块
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.market_data import Tick
from core.matching_engine import Order
from .base_strategy import BaseStrategy

class IceHangStrategy(BaseStrategy):
    """
    冰山挂价策略：最简冰山挂单
    - 一次挂 slice_qty 数量
    - 买入时以买一价挂单，卖出时以卖一价挂单
    - 未击中则在下一个tick撤单并重新挂单
    """

    def __init__(self, side: str, total_qty: int, slice_qty: int):
        """
        初始化冰山挂价策略
        
        Args:
            side: 买卖方向 ('buy' or 'sell')  
            total_qty: 目标总交易量
            slice_qty: 每次挂单的冰山块大小
        """
        super().__init__(side, total_qty)
        self.slice = slice_qty

    def _new_order(self, price: float, ts: pd.Timestamp) -> Order:
        """生成新订单"""
        self.next_id += 1
        qty = min(self.slice, self.left)
        self.pending = Order(self.next_id, self.side, price, qty, ts)
        return self.pending

    def on_tick(self, tick: Tick) -> Optional[Order]:
        """
        处理新tick：如果当前无挂单则按己方价挂新单
        
        Args:
            tick: 市场数据tick
            
        Returns:
            可能返回的新订单
        """
        if self.left == 0 or self.pending is not None:
            return None
        # 冰山挂价：买入用买一价，卖出用卖一价（与对价策略相反）
        price = tick.bid if self.side == "buy" else tick.ask
        return self._new_order(price, tick.ts)

    def on_fill(self):
        """处理成交：扣减剩余量并清空挂单"""
        if self.pending:
            self.left -= self.pending.qty
            self.pending = None

    def chase(self, tick: Tick) -> Order:
        """
        撤单并按最新己方价重新挂单
        优化：如果新tick的挂单价格与当前挂单价格一致，则不执行撤单重挂
        """
        if not self.pending:
            # 如果没有挂单，直接生成新订单
            price = tick.bid if self.side == "buy" else tick.ask
            new_order = self._new_order(price, tick.ts)
            return new_order
        
        # 计算新的挂单价格
        new_price = tick.bid if self.side == "buy" else tick.ask
        
        # 如果价格没有变化，则不需要撤单重挂，直接返回当前挂单
        if abs(new_price - self.pending.price) < 1e-6:  # 使用浮点数比较的容差
            return self.pending
        
        # 价格发生变化，执行撤单重挂
        self.pending = None
        new_order = self._new_order(new_price, tick.ts)
        return new_order
