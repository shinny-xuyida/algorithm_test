# strategy/ice_smart_only_imbalance.py
# 冰山智能策略：仅基于订单失衡判断挂价或对价

from typing import Optional
import pandas as pd

# 导入基础模块
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from market_data import Tick
from matching_engine import Order
from .base_strategy import BaseStrategy

class IceSmartOnlyImbalanceStrategy(BaseStrategy):
    """
    冰山智能策略（仅订单失衡版本）：根据订单失衡情况选择报价方式
    - 大部分时间采用挂价策略（被动等待成交）
    - 当订单失衡条件满足时，采用对价策略（主动成交）
    
    使用一档数据：
    - 只使用买一价、买一量、卖一价、卖一量进行计算
    - 简化计算逻辑，提高执行效率
    
    判断条件：
    订单失衡：Q = (买一量-卖一量)/(买一量+卖一量)
    
    交易逻辑：
    - 买入时：Q > threshold 表示买盘更强，使用对价主动成交
    - 卖出时：Q < -threshold 表示卖盘更强，使用对价主动成交
    - 其他情况：使用挂价策略等待成交
    """

    def __init__(self, side: str, total_qty: int, slice_qty: int, imbalance_threshold: float = 0.2):
        """
        初始化冰山智能策略（仅订单失衡版本）
        
        Args:
            side: 买卖方向 ('buy' or 'sell')  
            total_qty: 目标总交易量
            slice_qty: 每次挂单的冰山块大小
            imbalance_threshold: 订单失衡阈值，默认0.2
        """
        super().__init__(side, total_qty)
        self.slice = slice_qty
        self.threshold = imbalance_threshold

    def _calculate_imbalance(self, tick: Tick) -> tuple[float, bool]:
        """
        计算订单失衡指标（使用一档数据）
        
        Args:
            tick: 市场数据tick
            
        Returns:
            (Q值, 是否满足失衡条件)
        """
        # 只使用一档数据
        bid_vol = tick.bid_volume
        ask_vol = tick.ask_volume
        
        # 避免除零错误
        if bid_vol + ask_vol == 0:
            return 0.0, False
        
        # 计算订单失衡指标 Q = (B-A)/(B+A)
        Q = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
        # 判断是否满足失衡条件
        if self.side == "buy":
            # 买入：Q > threshold（买盘多，价格可能上涨）
            imbalance_condition = Q > self.threshold
        else:
            # 卖出：Q < -threshold（卖盘多，价格可能下跌）
            imbalance_condition = Q < -self.threshold
        
        return Q, imbalance_condition

    def _should_use_market_price(self, tick: Tick) -> bool:
        """
        判断是否应该使用对价（市价单）
        仅基于订单失衡判断
        
        Args:
            tick: 市场数据tick
            
        Returns:
            True表示使用对价，False表示使用挂价
        """
        Q, imbalance_condition = self._calculate_imbalance(tick)
        
        # 不输出调试信息，直接返回结果
        
        return imbalance_condition

    def _new_order(self, price: float, ts: pd.Timestamp) -> Order:
        """生成新订单"""
        self.next_id += 1
        qty = min(self.slice, self.left)
        self.pending = Order(self.next_id, self.side, price, qty, ts)
        return self.pending

    def on_tick(self, tick: Tick) -> Optional[Order]:
        """
        处理新tick：根据订单失衡判断选择挂价或对价
        
        Args:
            tick: 市场数据tick
            
        Returns:
            可能返回的新订单
        """
        if self.left == 0 or self.pending is not None:
            return None
        
        # 基于订单失衡判断使用哪种价格
        use_market = self._should_use_market_price(tick)
        
        if use_market:
            # 使用对价（主动成交）
            price = tick.ask if self.side == "buy" else tick.bid
        else:
            # 使用挂价（被动等待）
            price = tick.bid if self.side == "buy" else tick.ask
        
        return self._new_order(price, tick.ts)

    def on_fill(self):
        """处理成交：扣减剩余量并清空挂单"""
        if self.pending:
            self.left -= self.pending.qty
            self.pending = None

    def chase(self, tick: Tick) -> Order:
        """
        撤单并根据最新失衡判断重新挂单
        """
        if not self.pending:
            # 如果没有挂单，直接生成新订单
            use_market = self._should_use_market_price(tick)
            price = (tick.ask if self.side == "buy" else tick.bid) if use_market else \
                   (tick.bid if self.side == "buy" else tick.ask)
            new_order = self._new_order(price, tick.ts)
            return new_order
        
        # 重新判断失衡条件
        use_market = self._should_use_market_price(tick)
        new_price = (tick.ask if self.side == "buy" else tick.bid) if use_market else \
                   (tick.bid if self.side == "buy" else tick.ask)
        
        # 如果价格没有变化，则不需要撤单重挂
        if abs(new_price - self.pending.price) < 1e-6:
            return self.pending
        
        # 价格发生变化，执行撤单重挂
        self.pending = None
        new_order = self._new_order(new_price, tick.ts)
        return new_order
