# 算法交易回测框架

这是一个模块化的算法交易策略回测框架，将策略代码与基础设施代码进行了清晰的分离，便于管理和扩展。支持多档数据处理，提供多种成熟的冰山策略实现，并包含完整的算法对比和概率分析工具。

## 项目结构

```
algorithm_test/
├── strategy/                            # 策略包：包含所有交易策略
│   ├── __init__.py                     # 策略包初始化文件
│   ├── base_strategy.py                # 策略基类：定义策略标准接口
│   ├── ice_best.py                     # 冰山对价策略：主动成交策略
│   ├── ice_hang.py                     # 冰山挂价策略：被动等待策略  
│   ├── ice_smart.py                    # 冰山智能策略：智能切换策略（支持5档优化）
│   └── ice_smart_only_imbalance.py     # 智能失衡策略：仅基于订单失衡判断
├── results/                            # 结果目录：存储对比测试结果
├── algorithm_comparison.py             # 算法对比工具：多策略性能对比
├── ice_smart_caculator.py             # 智能策略概率计算器：信号分析工具
├── backtest_engine.py                 # 回测引擎：通用回测框架
├── market_data.py                     # 市场数据：数据类 + CSV读取器（支持多档）  
├── matching_engine.py                 # 撮合引擎：通用撮合逻辑
├── metrics.py                         # 指标收集：性能评价指标
├── contract_multiplier.py             # 合约乘数：品种配置信息
└── README.md                          # 项目说明文档
```

## 核心策略介绍

### 1. 冰山对价策略 (`ice_best.py`)
- **特点**：主动成交，快速执行
- **价格**：始终使用对手价（买入用卖一价，卖出用买一价）
- **适用场景**：追求快速成交，对价格不敏感的大单执行
- **风险**：可能产生较大滑点，但成交速度快

### 2. 冰山挂价策略 (`ice_hang.py`)
- **特点**：被动等待，成本控制
- **价格**：使用己方价格（买入用买一价，卖出用卖一价）
- **适用场景**：对成交价格敏感，可以等待的场景
- **风险**：可能成交缓慢或无法完全成交

### 3. 冰山智能策略 (`ice_smart.py`) ⭐
- **特点**：根据市场状况智能选择挂价或对价
- **核心算法**：
  - 盘口失衡指标：`Q = (总买量-总卖量)/(总买量+总卖量)`
  - 微观价格：量加权的中间价格计算
  - 双重判断：失衡信号 + 价格趋势信号
- **5档优化**：
  - 自动检测数据档位，优先使用5档进行计算
  - 如果没有5档数据，自动回退到1档计算
  - 多档计算能更准确反映市场深度和流动性
- **适用场景**：平衡成交速度和成本的通用解决方案

### 4. 智能失衡策略 (`ice_smart_only_imbalance.py`) 🆕
- **特点**：仅基于订单失衡判断，简化版智能策略
- **核心算法**：
  - 使用一档数据：买一价、买一量、卖一价、卖一量
  - 订单失衡指标：`Q = (买一量-卖一量)/(买一量+卖一量)`
  - 简化计算逻辑，提高执行效率
- **判断逻辑**：失衡超过阈值时选择对价，否则挂价
- **适用场景**：对计算效率有要求，或数据只有一档的场景

## 核心模块说明

### 策略层 (`strategy/`)
- **`base_strategy.py`** - 策略基类，定义所有策略必须实现的接口
- **`ice_best.py`** - 冰山对价策略，追求快速成交
- **`ice_hang.py`** - 冰山挂价策略，追求成本控制
- **`ice_smart.py`** - 冰山智能策略，智能决策（支持5档）
- **`ice_smart_only_imbalance.py`** - 智能失衡策略，简化版智能策略

### 基础设施层
- **`backtest_engine.py`** - 通用回测引擎，支持任意策略的回测
- **`market_data.py`** - 市场数据处理，支持1-5档数据自动解析
- **`matching_engine.py`** - 撮合引擎，模拟市场订单撮合过程
- **`metrics.py`** - 指标收集器，计算VWAP、滑点、交易时长等关键指标
- **`contract_multiplier.py`** - 合约配置，支持70+期货品种自动识别

### 分析工具层 🆕
- **`algorithm_comparison.py`** - 算法对比工具，一键运行多种策略并对比性能
- **`ice_smart_caculator.py`** - 智能策略概率计算器，分析盘口失衡和价格变动的关系

## 策略接口规范

所有策略都必须继承`BaseStrategy`并实现以下接口：

```python
from strategy.base_strategy import BaseStrategy

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

### 1. 单策略回测

#### 对价策略（快速成交）

```python
from strategy import IceBestStrategy
from backtest_engine import run_backtest

# 创建对价策略实例
strategy = IceBestStrategy(
    side="buy",          # 买入方向
    total_qty=200,       # 总目标量200手
    slice_qty=5          # 每次挂单5手
)

# 运行回测
result = run_backtest(
    csv_path="your_data.csv",
    strategy=strategy,
    start_time="2025-07-07 14:00:00"
)
```

#### 挂价策略（成本控制）

```python
from strategy import IceHangStrategy

strategy = IceHangStrategy(
    side="sell",         # 卖出方向
    total_qty=100,       # 总目标量100手
    slice_qty=3          # 每次挂单3手
)
```

#### 智能策略（推荐⭐）

```python
from strategy import IceSmartStrategy

strategy = IceSmartStrategy(
    side="buy",                    # 买入方向
    total_qty=200,                 # 总目标量200手
    slice_qty=5,                   # 每次挂单5手
    imbalance_threshold=0.2        # 盘口失衡阈值（可选）
)
```

#### 智能失衡策略（轻量版）

```python
from strategy import IceSmartOnlyImbalanceStrategy

strategy = IceSmartOnlyImbalanceStrategy(
    side="buy",                    # 买入方向
    total_qty=200,                 # 总目标量200手
    slice_qty=5,                   # 每次挂单5手
    imbalance_threshold=0.2        # 失衡阈值
)
```

### 2. 多策略对比测试 🆕

```python
from algorithm_comparison import AlgorithmComparison

# 创建对比器
comparison = AlgorithmComparison()

# 运行对比测试
comparison.run_comparison(
    csv_path="your_data.csv",
    side="buy",
    total_qty=200,
    slice_qty=5,
    start_time="2025-07-07 14:00:00"
)

# 自动生成对比报告到results/目录
```

### 3. 概率分析工具 🆕

```python
from ice_smart_caculator import IceSmartCalculator

# 创建分析器
calculator = IceSmartCalculator()

# 分析信号有效性
results = calculator.analyze_signals(
    csv_path="your_data.csv", 
    side="buy",
    time_window_seconds=30
)

# 查看信号统计
stats = calculator.get_signal_stats(results)
print(f"信号准确率: {stats['accuracy']:.2%}")
```

### 4. 快速开始

```bash
# 运行多策略对比测试
python algorithm_comparison.py

# 运行概率分析
python ice_smart_caculator.py
```

## 核心特性

### 🚀 多档数据支持
- **自动档位检测**：智能识别1档到5档数据
- **5档优化算法**：智能策略支持5档深度计算
- **向下兼容**：没有多档数据时自动回退到1档

### 📊 完整性能评估
- **VWAP对比**：相对市场成交量加权平均价格的表现
- **滑点分析**：量化交易成本和市场冲击
- **执行效率**：成交时长、成交率、委托笔数等指标

### 🏗️ 模块化架构
- **策略层分离**：策略逻辑与基础设施完全解耦
- **标准化接口**：统一的策略接口，易于扩展
- **通用回测引擎**：一次编写，支持所有策略类型

### 🎯 智能算法
- **双重判断机制**：盘口失衡 + 价格趋势双重确认
- **自适应决策**：根据市场状况动态选择挂价或对价
- **风险控制**：避免在不利市场条件下的冲动交易

### 🔍 分析工具 🆕
- **多策略对比**：一键运行所有策略并生成对比报告
- **概率分析**：分析交易信号的有效性和准确率
- **结果存储**：自动保存对比结果到results/目录

## 回测结果指标说明

- **平均成交价格**：策略实际成交的量加权平均价格
- **市场VWAP**：同期市场成交量加权平均价格（基准）
- **价格滑点**：相对于市场VWAP的成本差异（负值表示优于市场）
- **执行时长**：从首次成交到最后成交的耗时
- **成交笔数**：总成交次数
- **委托笔数**：总委托下单次数（含追单）

## 扩展指南

### 添加新策略
1. 在`strategy/`文件夹中创建新策略文件
2. 继承`BaseStrategy`类并实现必要接口
3. 在`strategy/__init__.py`中导出新策略
4. 在`algorithm_comparison.py`中添加新策略到对比列表

### 支持新数据格式
- 修改`market_data.py`中的`tick_reader`函数
- 添加新的列名映射规则
- 测试多档数据的解析正确性

### 扩展性能指标
- 在`metrics.py`中的`Metrics`类添加新指标
- 更新`summary`方法返回新的评估维度

### 自定义分析工具
- 参考`ice_smart_caculator.py`的结构
- 实现自定义的信号分析逻辑
- 添加到主要分析流程中

## 算法对比结果示例

```
算法对比测试结果
==================================================

【冰山对价策略】
  平均成交价格: 3126.0
  价格滑点: 2.34
  执行时长(秒): 20.0
  成交笔数: 40

【冰山挂价策略】
  平均成交价格: 3124.9
  价格滑点: 1.08
  执行时长(秒): 2157.5
  成交笔数: 40

【冰山智能策略】
  平均成交价格: 3124.9
  价格滑点: 1.08
  执行时长(秒): 2157.5
  成交笔数: 40

【智能失衡策略】
  平均成交价格: 3125.35
  价格滑点: 1.68
  执行时长(秒): 96.0
  成交笔数: 40
```

## 技术支持

- 支持期货、股票等多种金融工具
- 兼容多种CSV数据格式
- 自动合约乘数识别（70+期货品种）
- 完整的调试信息输出
- 多策略并行测试和对比
- 交易信号概率分析 