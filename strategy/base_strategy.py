# strategy/base_strategy.py
# -------------------------------------------------------------------
# 策略基类：定义策略的标准接口
# -------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

# 导入数据类型（需要从上级目录导入）
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from market_data import Tick
from matching_engine import Order

class BaseStrategy(ABC):
    """
    策略基类：定义所有交易策略必须实现的接口
    
    所有具体策略都应该继承此类并实现以下方法：
    - on_tick(): 处理新的市场数据tick
    - on_fill(): 处理成交回报
    - chase(): 追价逻辑
    
    以及维护以下属性：
    - pending: 当前挂单
    - left: 剩余数量
    - side: 买卖方向
    """
    
    def __init__(self, side: str, total_qty: int):
        """
        初始化策略基本参数
        
        Args:
            side: 买卖方向 ('buy' or 'sell')
            total_qty: 目标总交易量
        """
        self.side = side
        self.total = total_qty
        self.left = total_qty
        self.pending: Optional[Order] = None
        self.next_id = 0
    
    @abstractmethod
    def on_tick(self, tick: Tick) -> Optional[Order]:
        """
        处理新的市场数据tick
        
        Args:
            tick: 市场数据tick
            
        Returns:
            可能返回的新订单，如果不需要下单则返回None
        """
        pass
    
    @abstractmethod
    def on_fill(self):
        """
        处理成交回报
        策略应该在此方法中更新剩余数量和挂单状态
        """
        pass
    
    @abstractmethod
    def chase(self, tick: Tick) -> Order:
        """
        追价逻辑：当前挂单未成交时的处理
        
        Args:
            tick: 最新的市场数据tick
            
        Returns:
            新的追价订单
        """
        pass
    
    def is_finished(self) -> bool:
        """
        判断策略是否已完成所有交易
        
        Returns:
            True表示已完成，False表示还有剩余数量需要交易
        """
        return self.left <= 0 