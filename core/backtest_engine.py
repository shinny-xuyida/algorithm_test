# backtest_engine.py
# -------------------------------------------------------------------
# 通用回测引擎：支持任意策略的回测框架
# -------------------------------------------------------------------

from typing import Optional, Any
import pandas as pd
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入可复用模块
from .market_data import Tick, tick_reader
from .matching_engine import match
from .metrics import Metrics
from tools.contract_multiplier import get_contract_info_from_file

# === 通用回测引擎 ============================================================

def run_backtest(csv_path: str = None,
                 strategy: Any = None,
                 start_time: str = None,
                 contract_multiplier: Optional[int] = None,
                 ticks_data: Optional[list] = None) -> dict:
    """
    通用回测引擎：逐 Tick 驱动撮合 & 指标收集
    * csv_path           : 行情 CSV 文件路径（当ticks_data为None时必需）
    * strategy           : 策略实例，需要实现以下接口：
                          - on_tick(tick) -> Optional[Order]  # 处理tick，可能返回新订单
                          - on_fill()                         # 处理成交
                          - chase(tick) -> Order             # 追单逻辑
                          - pending: Optional[Order]         # 当前挂单
                          - left: int                        # 剩余数量
                          - side: str                        # 买卖方向
    * start_time         : 起始时间 (字符串，可含日期)
    * contract_multiplier: 合约乘数，用于计算市场VWAP。如果为None则自动从文件名推断
    * ticks_data         : 预读取的tick数据列表（用于优化性能，避免重复读取文件）
    """
    # 获取tick数据
    if ticks_data is not None:
        # 使用预读取的数据
        ticks = ticks_data
        # 如果使用预读取的数据但没有指定合约乘数，需要从csv_path推断
        if contract_multiplier is None and csv_path is not None:
            _, contract_multiplier = get_contract_info_from_file(csv_path, default_multiplier=1, verbose=False)
        elif contract_multiplier is None:
            contract_multiplier = 1  # 默认值
    else:
        # 从文件读取数据（原有方式）
        if csv_path is None:
            raise ValueError("csv_path和ticks_data不能同时为None")
        
        # 自动推断合约乘数（如果用户未指定）
        if contract_multiplier is None:
            _, contract_multiplier = get_contract_info_from_file(csv_path, default_multiplier=1, verbose=True)
        
        ticks = tick_reader(csv_path)                   # 全自动 Tick 迭代器
    start_ts= pd.to_datetime(start_time)              # 起始时间
    metric  = Metrics(contract_multiplier)

    prev_tick: Optional[Tick] = None                  # 上一 Tick

    for tick in ticks:
        metric.on_tick(tick)                         # 统计市场 VWAP
        if tick.ts < start_ts:                       # 未到开始时间
            prev_tick = tick
            continue

        # --- 若有挂单，用 "下一 Tick"(当前 tick) 撮合 --------------------
        if strategy.pending and prev_tick is not None:
            fill = match(strategy.pending, tick)
            if fill:
                metric.on_fill(fill)
                strategy.on_fill()                   # 剩余量扣减
            else:
                # 记录追单前的订单ID，用于判断是否真的产生了新委托
                old_order_id = strategy.pending.id if strategy.pending else None
                new_order = strategy.chase(tick)     # 未成交 → 追单
                # 只有当订单ID发生变化时，才记录新的委托笔数
                if new_order and (not old_order_id or new_order.id != old_order_id):
                    metric.on_order(new_order.ts)   # 记录追单委托笔数

        # --- 让策略决定是否挂新单 ---------------------------------------
        new_order = strategy.on_tick(tick)
        if new_order:
            metric.on_order(new_order.ts)  # 记录委托笔数

        prev_tick = tick                             # 存起来做下轮对手价
        if strategy.left == 0:                      # 全部成交 → 提前结束
            break

    # 生成评测结果
    result = metric.summary(strategy.side)
    
    # 输出评测结果
    print("\n" + "="*50)
    print("回测评测结果")
    print("="*50)
    for key, value in result.items():
        if value is not None:
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: N/A")
    print("="*50 + "\n")
    
    return result 