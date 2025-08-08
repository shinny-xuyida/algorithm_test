"""
工具模块 - 算法交易工具集
Tools and utilities for algorithmic trading

此模块包含算法交易系统的各种工具和实用程序：

🔧 核心工具：
- contract_multiplier: 合约乘数配置，管理各品种期货合约参数
- data_evaluation: 数据评估工具，分析盘口失衡信号有效性
- ice_smart_caculator: 智能策略计算器，分析信号概率
- data_evaluation_long_period: 长期数据评估，多交易日稳定性分析

📊 主要功能：
- 合约参数管理（支持中金所、大商所、上期所等）
- 盘口失衡信号分析
- 策略参数优化
- 多进程并行数据处理
- 稳定性和有效性评估

🎯 应用场景：
- 策略参数调优
- 历史数据分析
- 信号有效性验证
- 合约筛选和评估
"""

__all__ = [
    'contract_multiplier',
    'ice_smart_caculator', 
    'data_evaluation'
] 