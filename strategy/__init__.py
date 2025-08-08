"""
策略模块 - 算法交易策略实现
Strategy module for algorithmic trading strategies

此模块包含多种期货算法交易策略的实现：

🎯 策略架构：
- base_strategy: 策略基类，定义统一的交易接口
- 所有具体策略都继承自BaseStrategy，实现标准化接口

📈 内置策略：
- ice_best: 冰山对价策略 - 主动成交，快速执行
- ice_hang: 冰山挂价策略 - 被动等待，降低成本  
- ice_smart: 冰山智能策略 - 智能切换报价方式
- ice_smart_only_imbalance: 智能失衡策略 - 基于盘口失衡判断

🔧 策略特性：
- 标准化的策略接口（on_tick, on_fill, chase）
- 支持冰山算法的大单拆分
- 智能的市场微观结构分析
- 灵活的参数配置
"""

from .base_strategy import BaseStrategy
from .ice_best import IceBestStrategy
from .ice_hang import IceHangStrategy
from .ice_smart import IceSmartStrategy
from .ice_smart_only_imbalance import IceSmartOnlyImbalanceStrategy

__all__ = [
    'BaseStrategy',
    'IceBestStrategy',
    'IceHangStrategy',
    'IceSmartStrategy',
    'IceSmartOnlyImbalanceStrategy'
] 