# market_data.py
# -------------------------------------------------------------------
# 市场数据模块：数据类定义 + CSV读取器
# -------------------------------------------------------------------

import re                                   # 用于正则匹配列名
from dataclasses import dataclass           # 轻量级数据容器
from typing import Iterator                 # 类型提示
import pandas as pd                         # 读取 CSV & 时间戳处理

# === 数据类定义 ===============================================================

@dataclass
class Tick:
    """单条行情快照（支持 1~N 档价格和数量）"""
    ts:   pd.Timestamp            # 时间戳
    bids: list                    # [bid_price1 .. bid_priceN]
    asks: list                    # [ask_price1 .. ask_priceN]
    bid_volumes: list             # [bid_volume1 .. bid_volumeN] - 买档数量
    ask_volumes: list             # [ask_volume1 .. ask_volumeN] - 卖档数量
    last: float                   # 最新价
    vol:  float                   # 本 Tick 成交量
    amt:  float                   # 本 Tick 成交额

    # 便捷属性：一级买 / 卖价
    @property
    def bid(self) -> float:       # 一级买价
        return self.bids[0]

    @property
    def ask(self) -> float:       # 一级卖价
        return self.asks[0]
    
    @property
    def bid_volume(self) -> float:  # 一级买量
        return self.bid_volumes[0] if self.bid_volumes else 0.0
    
    @property
    def ask_volume(self) -> float:  # 一级卖量
        return self.ask_volumes[0] if self.ask_volumes else 0.0


@dataclass
class Order:
    """委托单"""
    id:    int
    side:  str                    # 'buy' or 'sell'
    price: float
    qty:   int
    ts:    pd.Timestamp


@dataclass
class Fill:
    """成交记录"""
    order_id: int
    price:    float
    qty:      int
    ts:       pd.Timestamp

# === CSV → Tick 读取器 =======================================================

def tick_reader(path: str,
                tz_offset: int = 8) -> Iterator[Tick]:
    """
    逐行读取 CSV 并生成 Tick，完全自动：
    1. 自动探测合约前缀（或无前缀）
    2. 自动探测最大档位（N 档）
    3. 只要求至少有 bid_price1 / ask_price1
    4. 自动支持买卖量字段
    """
    df = pd.read_csv(path)                     # 读文件

    # --- 1) 探测前缀 --------------------------------------------------------
    prefix = ""                               # 若列名为 bid_price1 → 无前缀
    for col in df.columns:
        if col.endswith(".bid_price1"):       # 形如 EX.C.bid_price1
            prefix = col.replace("bid_price1", "")
            break

    # 确保至少能找到一档买价列
    if prefix == "" and "bid_price1" not in df.columns:
        raise ValueError("表头里找不到 bid_price1，无法确定报价列")

    # --- 2) 探测最大档位 ---------------------------------------------------
    pat = re.compile(re.escape(prefix) + r"bid_price(\d)")
    levels = max(
        int(m.group(1))
        for c in df.columns
        if (m := pat.match(c))
    )  # 至少能匹配到 1

    bid_cols = [f"{prefix}bid_price{i}" for i in range(1, levels + 1)]
    ask_cols = [f"{prefix}ask_price{i}" for i in range(1, levels + 1)]
    bid_vol_cols = [f"{prefix}bid_volume{i}" for i in range(1, levels + 1)]
    ask_vol_cols = [f"{prefix}ask_volume{i}" for i in range(1, levels + 1)]

    last_col = f"{prefix}last_price"
    vol_col  = f"{prefix}volume"
    amt_col  = f"{prefix}amount"

    # --- 3) 行转 Tick -------------------------------------------------------
    for _, row in df.iterrows():
        # 3.1 时间戳：优先纳秒列
        if pd.notna(row.get("datetime_nano", None)):
            ts = pd.to_datetime(row["datetime_nano"], unit="ns")
        else:
            ts = pd.to_datetime(row["datetime"])
        ts += pd.Timedelta(hours=tz_offset)   # 转本地时区 (+8)

        # 3.2 买/卖档列表（缺列或 NaN 自动跳过）
        bids = [float(v) for c in bid_cols if c in row and pd.notna((v := row[c]))]
        asks = [float(v) for c in ask_cols if c in row and pd.notna((v := row[c]))]
        bid_volumes = [float(v) for c in bid_vol_cols if c in row and pd.notna((v := row[c]))]
        ask_volumes = [float(v) for c in ask_vol_cols if c in row and pd.notna((v := row[c]))]
        
        # 如果没有量数据，用默认值填充
        if not bid_volumes and bids:
            bid_volumes = [1.0] * len(bids)  # 默认每档1手
        if not ask_volumes and asks:
            ask_volumes = [1.0] * len(asks)  # 默认每档1手

        if not bids or not asks:             # 缺少一级买/卖价，跳过该行
            print(f"跳过 {ts}：缺少一级买/卖价")
            continue

        # 3.3 其他字段（允许缺失）
        last = float(row[last_col]) if last_col in row and pd.notna(row[last_col]) \
               else (bids[0] + asks[0]) / 2
        vol  = float(row[vol_col])  if vol_col in row and pd.notna(row[vol_col]) else 0.0
        amt  = float(row[amt_col])  if amt_col in row and pd.notna(row[amt_col]) else last * vol

        yield Tick(ts, bids, asks, bid_volumes, ask_volumes, last, vol, amt) 