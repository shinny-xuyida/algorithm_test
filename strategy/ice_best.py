# strategy/ice_best.py
# 冰山对价策略：将大单拆分成小块逐步执行

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

class IceBestStrategy(BaseStrategy):
    """
    冰山对价策略：最简冰山追单
    
    策略核心逻辑：
    - 将大单拆分成多个小块（slice）逐步执行，避免对市场造成冲击
    - 每次只挂出一个slice_qty数量的订单
    - 如果当前价格不利（未击中），则撤单并按最新对手价重新挂单
    - 持续追击直到全部成交完毕
    """

    def __init__(self, side: str, total_qty: int, slice_qty: int):
        """
        初始化冰山策略
        
        Args:
            side: 买卖方向 ('buy' or 'sell')  
            total_qty: 目标总交易量（大单总量）
            slice_qty: 每次挂单的冰山块大小（单次暴露量）
        """
        super().__init__(side, total_qty)
        self.slice = slice_qty  # 每次挂单的冰山块大小

    def _new_order(self, price: float, ts: pd.Timestamp) -> Order:
        """
        生成新订单
        
        生成逻辑：
        1. 递增订单ID确保唯一性
        2. 计算本次挂单量：取剩余量和冰山块大小的最小值
        3. 创建新订单并设置为当前挂单
        
        Args:
            price: 挂单价格
            ts: 时间戳
            
        Returns:
            新生成的订单对象
        """
        self.next_id += 1
        # 计算本次挂单量：不能超过剩余需要交易的量
        qty = min(self.slice, self.left)
        # 创建新订单并记录为当前挂单
        self.pending = Order(self.next_id, self.side, price, qty, ts)
        return self.pending

    def on_tick(self, tick: Tick) -> Optional[Order]:
        """
        处理新tick：如果当前无挂单则按对手价挂新单
        
        触发条件：
        1. 还有剩余量需要交易 (self.left > 0)
        2. 当前没有挂单在市场上 (self.pending is None)
        
        Args:
            tick: 市场数据tick
            
        Returns:
            可能返回的新订单，如果不满足挂单条件则返回None
        """
        # 检查是否还需要交易且当前无挂单
        if self.left == 0 or self.pending is not None:
            return None
        
        # 根据买卖方向选择对手价：买单取ask价，卖单取bid价
        price = tick.ask if self.side == "buy" else tick.bid
        return self._new_order(price, tick.ts)

    def on_fill(self):
        """
        处理成交：扣减剩余量并清空挂单
        
        成交处理逻辑：
        1. 从剩余量中扣除本次成交量
        2. 清空当前挂单状态，为下次挂单做准备
        """
        if self.pending:
            # 扣减剩余量
            self.left -= self.pending.qty
            # 清空挂单状态
            self.pending = None

    def chase(self, tick: Tick) -> Order:
        """
        撤单并按最新对手价追挂
        
        追单逻辑：
        1. 撤销当前挂单（设置pending为None）
        2. 按最新tick的对手价重新挂单
        3. 输出追单信息用于监控
        
        Args:
            tick: 最新市场数据
            
        Returns:
            新的追单订单
        """
        # 撤销当前挂单
        self.pending = None
        
        # 根据买卖方向获取最新对手价
        price = tick.ask if self.side == "buy" else tick.bid
        
        # 生成新的追单订单
        new_order = self._new_order(price, tick.ts)
        
        # 追单完成，不输出详细信息
        return new_order 