"""
核心模块 - 算法交易回测系统
Core modules for algorithmic trading backtest system

此模块包含算法交易回测系统的核心组件：

📊 核心组件：
- backtest_engine: 回测引擎，驱动整个回测流程
- matching_engine: 撮合引擎，基于tick级别的逐笔撮合
- market_data: 市场数据处理，支持多档位数据读取
- metrics: 性能指标统计，计算VWAP、滑点等关键指标

🔧 主要功能：
- 事件驱动的回测框架
- 精确的tick级撮合逻辑
- 全面的性能评估体系
- 支持多种数据格式
"""

from .backtest_engine import run_backtest
from .matching_engine import Order, Fill, match
from .market_data import tick_reader, Tick
from .metrics import Metrics

__all__ = [
    'run_backtest',
    'Order',
    'Fill', 
    'match',
    'tick_reader',
    'Tick',
    'Metrics'
] 