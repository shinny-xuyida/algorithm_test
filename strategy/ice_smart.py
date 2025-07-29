# strategy/ice_smart.py
# 冰山智能策略：根据盘口情况智能选择挂价或对价

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

class IceSmartStrategy(BaseStrategy):
    """
    冰山智能策略：根据盘口情况智能选择报价方式
    - 大部分时间采用挂价策略（被动等待成交）
    - 当盘口偏离和micro-price两个条件同时满足时，采用对价策略（主动成交）
    
    多档优化：
    - 如果数据包含5档信息，则使用5档的价格和量进行计算
    - 如果没有5档数据，则回退到使用1档数据
    - 多档计算能够更准确地反映市场深度和流动性状况
    
    判断条件：
    1. 盘口偏离：Q = (总买量-总卖量)/(总买量+总卖量)
       - 单档：使用买一量和卖一量
       - 5档：使用前5档的总买量和总卖量
       |Q| > 0.2 说明盘口偏离，需要根据交易方向判断是否使用对价
    2. micro-price：量加权的中间价格
       - 单档：(买1价*卖1量+卖1价*买1量)/(买1量+卖1量)
       - 5档：(买方加权价*总卖量+卖方加权价*总买量)/(总买量+总卖量)
       判断加权中价更接近买方还是卖方，判断市场趋势
    """

    def __init__(self, side: str, total_qty: int, slice_qty: int, imbalance_threshold: float = 0.2):
        """
        初始化冰山智能策略
        
        Args:
            side: 买卖方向 ('buy' or 'sell')  
            total_qty: 目标总交易量
            slice_qty: 每次挂单的冰山块大小
            imbalance_threshold: 盘口失衡阈值，默认0.2
        """
        super().__init__(side, total_qty)
        self.slice = slice_qty
        self.threshold = imbalance_threshold

    def _calculate_market_metrics(self, tick: Tick) -> tuple[float, float, bool, bool]:
        """
        计算市场指标（支持5档数据）
        
        Args:
            tick: 市场数据tick
            
        Returns:
            (Q值, micro_price, 是否盘口失衡, micro_price是否支持当前方向)
        """
        # 确定使用的档位数量（优先使用5档，如果没有则使用1档）
        available_levels = min(len(tick.bids), len(tick.asks), len(tick.bid_volumes), len(tick.ask_volumes))
        use_levels = min(5, available_levels) if available_levels >= 5 else 1
        
        # 计算总买量和总卖量
        total_bid_vol = sum(tick.bid_volumes[:use_levels])
        total_ask_vol = sum(tick.ask_volumes[:use_levels])
        
        # 避免除零错误
        if total_bid_vol + total_ask_vol == 0:
            return 0.0, (tick.bid + tick.ask) / 2, False, False
        
        # 1. 计算盘口失衡指标 Q = (B-A)/(B+A)
        Q = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        
        # 2. 计算加权micro-price
        if use_levels == 1:
            # 单档情况：使用原有计算方式
            micro_price = (tick.bid * total_ask_vol + tick.ask * total_bid_vol) / (total_bid_vol + total_ask_vol)
        else:
            # 多档情况：使用量加权的多档价格
            # 买方加权价格：sum(bid_price[i] * bid_volume[i]) / sum(bid_volume[i])
            weighted_bid_price = sum(tick.bids[i] * tick.bid_volumes[i] for i in range(use_levels)) / total_bid_vol
            # 卖方加权价格：sum(ask_price[i] * ask_volume[i]) / sum(ask_volume[i])  
            weighted_ask_price = sum(tick.asks[i] * tick.ask_volumes[i] for i in range(use_levels)) / total_ask_vol
            # micro-price：根据对方量来加权己方价格
            micro_price = (weighted_bid_price * total_ask_vol + weighted_ask_price * total_bid_vol) / (total_bid_vol + total_ask_vol)
        
        # 3. 判断是否盘口失衡
        imbalance_signal = abs(Q) > self.threshold
        
        # 4. 判断micro-price方向信号
        mid_price = (tick.bid + tick.ask) / 2
        if self.side == "buy":
            # 买入：Q > threshold（买盘多，价格可能上涨）且 micro_price更接近卖一价（确认上涨趋势）
            direction_signal = (Q > self.threshold) and (abs(micro_price - tick.ask) < abs(micro_price - tick.bid))
        else:
            # 卖出：Q < -threshold（卖盘多，价格可能下跌）且 micro_price更接近买一价（确认下跌趋势）
            direction_signal = (Q < -self.threshold) and (abs(micro_price - tick.bid) < abs(micro_price - tick.ask))
        
        return Q, micro_price, imbalance_signal, direction_signal

    def _should_use_market_price(self, tick: Tick) -> bool:
        """
        判断是否应该使用对价（市价单）
        
        Args:
            tick: 市场数据tick
            
        Returns:
            True表示使用对价，False表示使用挂价
        """
        Q, micro_price, imbalance_signal, direction_signal = self._calculate_market_metrics(tick)
        
        # 确定使用的档位数量
        available_levels = min(len(tick.bids), len(tick.asks), len(tick.bid_volumes), len(tick.ask_volumes))
        use_levels = min(5, available_levels) if available_levels >= 5 else 1
        
        # 两个条件都满足才使用对价
        use_market = imbalance_signal and direction_signal
        
        return use_market

    def _new_order(self, price: float, ts: pd.Timestamp) -> Order:
        """生成新订单"""
        self.next_id += 1
        qty = min(self.slice, self.left)
        self.pending = Order(self.next_id, self.side, price, qty, ts)
        return self.pending

    def on_tick(self, tick: Tick) -> Optional[Order]:
        """
        处理新tick：根据智能判断选择挂价或对价
        
        Args:
            tick: 市场数据tick
            
        Returns:
            可能返回的新订单
        """
        if self.left == 0 or self.pending is not None:
            return None
        
        # 智能判断使用哪种价格
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
        撤单并根据最新智能判断重新挂单
        """
        if not self.pending:
            # 如果没有挂单，直接生成新订单
            use_market = self._should_use_market_price(tick)
            price = (tick.ask if self.side == "buy" else tick.bid) if use_market else \
                   (tick.bid if self.side == "buy" else tick.ask)
            new_order = self._new_order(price, tick.ts)
            return new_order
        
        # 重新智能判断
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
