# backtest_engine.py
# -------------------------------------------------------------------
# 通用回测引擎：支持任意策略的回测框架
# -------------------------------------------------------------------

from typing import Optional, Any
import pandas as pd

# 导入可复用模块
from market_data import Tick, tick_reader
from matching_engine import match
from metrics import Metrics
from contract_multiplier import get_contract_info_from_file

# === 通用回测引擎 ============================================================

def run_backtest(csv_path: str,
                 strategy: Any,
                 start_time: str,
                 contract_multiplier: Optional[int] = None) -> dict:
    """
    通用回测引擎：逐 Tick 驱动撮合 & 指标收集
    * csv_path           : 行情 CSV
    * strategy           : 策略实例，需要实现以下接口：
                          - on_tick(tick) -> Optional[Order]  # 处理tick，可能返回新订单
                          - on_fill()                         # 处理成交
                          - chase(tick) -> Order             # 追单逻辑
                          - pending: Optional[Order]         # 当前挂单
                          - left: int                        # 剩余数量
                          - side: str                        # 买卖方向
    * start_time         : 起始时间 (字符串，可含日期)
    * contract_multiplier: 合约乘数，用于计算市场VWAP。如果为None则自动从文件名推断
    """
    # 自动推断合约乘数（如果用户未指定）
    if contract_multiplier is None:
        _, contract_multiplier = get_contract_info_from_file(csv_path, default_multiplier=1, verbose=True)
    
    ticks   = tick_reader(csv_path)                   # 全自动 Tick 迭代器
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
                print(f"成交: {fill.qty}@{fill.price} at {fill.ts}")
                metric.on_fill(fill)
                strategy.on_fill()                   # 剩余量扣减
            else:
                print(f"未成交，追单: 原价{strategy.pending.price} -> 新价{tick.ask if strategy.side=='buy' else tick.bid}")
                # 记录追单前的订单ID，用于判断是否真的产生了新委托
                old_order_id = strategy.pending.id if strategy.pending else None
                new_order = strategy.chase(tick)     # 未成交 → 追单
                # 只有当订单ID发生变化时，才记录新的委托笔数
                if new_order and (not old_order_id or new_order.id != old_order_id):
                    metric.on_order()                # 记录追单委托笔数

        # --- 让策略决定是否挂新单 ---------------------------------------
        new_order = strategy.on_tick(tick)
        if new_order:
            print(f"新下单: {new_order.side} {new_order.qty}@{new_order.price} at {new_order.ts}")
            metric.on_order()  # 记录委托笔数

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