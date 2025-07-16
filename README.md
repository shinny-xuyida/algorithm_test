# 算法交易回测框架

这是一个模块化的算法交易策略回测框架，将策略代码与基础设施代码进行了清晰的分离，便于管理和扩展。

## 项目结构

```
algorithm_test/
├── strategy/                # 策略包：包含所有交易策略
│   ├── __init__.py         # 策略包初始化文件
│   ├── base_strategy.py    # 策略基类：定义策略标准接口
│   ├── iceberg.py          # 冰山策略：大单拆分执行策略
│   └── example.py          # 策略使用示例
├── backtest_engine.py      # 回测引擎：通用回测框架
├── market_data.py          # 市场数据：数据类 + CSV读取器  
├── matching_engine.py      # 撮合引擎：通用撮合逻辑
├── metrics.py             # 指标收集：性能评价指标
├── contract_multiplier.py  # 合约乘数：品种配置信息
├── ice_active.py          # 冰山策略示例（兼容性文件）
└── README.md              # 项目说明文档
```

## 核心模块说明

### 策略层 (`strategy/`)
- **`base_strategy.py`** - 策略基类，定义所有策略必须实现的接口
- **`iceberg.py`** - 冰山策略实现，将大单拆分成小块逐步执行
- **`example.py`** - 策略使用示例，展示如何配置和运行不同策略

### 基础设施层
- **`backtest_engine.py`** - 通用回测引擎，支持任意策略的回测
- **`market_data.py`** - 市场数据处理，包含Tick/Order/Fill数据类和CSV读取器
- **`matching_engine.py`** - 撮合引擎，模拟市场订单撮合过程
- **`metrics.py`** - 指标收集器，计算策略性能评价指标
- **`contract_multiplier.py`** - 合约配置，包含各品种的合约乘数信息

## 策略接口规范

所有策略都必须继承`BaseStrategy`并实现以下接口：

```python
class YourStrategy(BaseStrategy):
    def on_tick(self, tick: Tick) -> Optional[Order]:
        """处理新的市场数据tick，可能返回新订单"""
        pass
    
    def on_fill(self):
        """处理成交回报，更新策略状态"""
        pass
    
    def chase(self, tick: Tick) -> Order:
        """追价逻辑，处理未成交订单"""
        pass
```

## 使用方法

### 1. 使用现有策略

```python
from strategy import IcebergStrategy
from backtest_engine import run_backtest

# 创建策略实例
strategy = IcebergStrategy(
    side="buy",          # 买入方向
    total_qty=200,       # 总目标量
    slice_qty=5          # 每次挂单量
)

# 运行回测
result = run_backtest(
    csv_path="your_data.csv",
    strategy=strategy,
    start_time="2025-07-07 09:30:00"
)
```

### 2. 开发新策略

```python
from strategy.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, side, total_qty, **kwargs):
        super().__init__(side, total_qty)
        # 初始化你的策略参数
    
    def on_tick(self, tick):
        # 实现你的策略逻辑
        pass
    
    def on_fill(self):
        # 处理成交
        pass
    
    def chase(self, tick):
        # 追价逻辑
        pass
```

### 3. 快速开始

```bash
# 运行冰山策略示例
python ice_active.py

# 或运行详细的策略示例
python strategy/example.py
```

## 特性

- **清晰的架构分层**：策略层与基础设施层分离，便于管理
- **标准化策略接口**：所有策略遵循统一接口，易于扩展
- **自动数据解析**：支持多种CSV格式，自动识别合约信息
- **通用回测引擎**：一次编写，支持所有策略类型
- **完整性能评估**：提供VWAP、滑点、交易时长等关键指标
- **品种自动识别**：自动从文件名提取合约信息和乘数

## 扩展指南

1. **添加新策略**：在`strategy/`文件夹中创建新的策略文件，继承`BaseStrategy`
2. **修改撮合逻辑**：编辑`matching_engine.py`中的`match`函数
3. **增加性能指标**：扩展`metrics.py`中的`Metrics`类
4. **支持新数据格式**：修改`market_data.py`中的`tick_reader`函数 