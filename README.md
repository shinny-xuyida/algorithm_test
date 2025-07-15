# 冰山策略回测框架

这是一个模块化的冰山对价策略回测框架，将原本单一文件的代码拆分为多个可复用的模块。

## 项目结构

```
algorithm_coding/
├── ice_active.py        # 主策略模块：冰山策略 + 回测循环
├── market_data.py       # 市场数据模块：数据类 + CSV读取器  
├── matching_engine.py   # 撮合引擎模块：通用撮合逻辑
├── metrics.py          # 指标收集模块：性能评价指标
└── README.md           # 项目说明文档
```

## 模块说明

### `ice_active.py` - 主策略模块
- **IcebergStrategy类**：冰山对价策略的核心逻辑
- **run_backtest函数**：主回测循环，驱动整个回测流程
- 包含策略的具体实现和主程序入口

### `market_data.py` - 市场数据模块
- **数据类**：Tick、Order、Fill 数据结构定义
- **tick_reader函数**：CSV文件自动解析器，支持多档行情数据
- 提供统一的市场数据接口

### `matching_engine.py` - 撮合引擎模块  
- **match函数**：通用撮合逻辑，模拟市场撮合过程
- 支持不同的撮合规则扩展

### `metrics.py` - 指标收集模块
- **Metrics类**：统一的性能评价指标计算
- 包含平均成交价、市场VWAP、滑点、成交时长等指标

## 使用方法

```python
# 运行主策略
python ice_active.py

# 或者在其他脚本中导入模块
from ice_active import IcebergStrategy, run_backtest
from market_data import tick_reader
from metrics import Metrics
```

## 特性

- **模块化设计**：各模块职责单一，便于维护和扩展
- **自动数据解析**：支持多种CSV格式，自动识别列名前缀和档位数
- **通用撮合引擎**：可扩展的撮合逻辑，支持不同的交易规则
- **完整性能评估**：提供全面的策略性能指标分析 