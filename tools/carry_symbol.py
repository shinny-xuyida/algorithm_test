# -*- coding: utf-8 -*-
"""
展期收益率（Roll Yield, RY）计算 - 仅用 TqSDK
------------------------------------------------
核心思路：
1) 对每个 product_id（如 "rb", "au", "TA"），找所有未到期合约；
2) 每个交易日挑“最靠近到期”的两张：近月(near) 与 次近月(far)；
3) 用收盘价计算 RY = (ln(near_close) - ln(far_close)) * 365 / Δt，
   其中 Δt 为两张合约“到期日”的天数差（注意符号约定）。
"""

from datetime import datetime, timedelta, date
from typing import List, Literal, Dict, Any
import numpy as np
import pandas as pd

from tqsdk import TqApi, TqAuth
from tqsdk.tafunc import time_to_datetime


def _last_n_trading_dates(api: TqApi, n: int = 10, lookback_days: int = 200) -> List[date]:
    """
    取最近 n 个交易日（date 对象）。
    - lookback_days 只用于取交易日历的观察窗口，够大即可。
    """
    now = datetime.now()
    cal = api.get_trading_calendar(start_dt=now - timedelta(days=lookback_days), end_dt=now)
    if cal.empty:
        return []
    dts = cal[cal["trading"] == True]["date"].to_list()
    return [time_to_datetime(d).date() for d in dts][-n:]


def _pick_near_far_by_expiry(quotes: List[Dict[str, Any]]) -> List[str]:
    """
    用合约的到期日（expire_datetime）+ 次关键字 open_interest 进行排序；
    返回按“越近越前”的 symbol 列表（只包含未到期且有持仓量的）。
    """
    rows = []
    for q in quotes:
        # 过滤掉没有持仓或缺少到期日的
        if q.get("open_interest", 0) <= 0 or q.get("expire_datetime", 0) <= 0:
            continue
        rows.append({
            "symbol": q["instrument_id"],  # e.g. "SHFE.rb2410"
            "expire_date": time_to_datetime(q["expire_datetime"]).date(),
            "open_oi": q["open_interest"],
        })
    if not rows:
        return []

    df = pd.DataFrame(rows).sort_values(by=["expire_date", "open_oi"], ascending=[True, False])
    # 只保留“未交割”的（到期日在今天之后）
    today = datetime.now().date()
    df = df[df["expire_date"] > today]
    return df["symbol"].tolist()


def compute_rolling_yield_history(
    api: TqApi,
    product_ids: List[str],
    n_trading_days: int = 10,
    sign_convention: Literal["original", "backwardation_positive"] = "original",
    klines_window: int = 200,
) -> pd.DataFrame:
    """
    计算“最近 n 个交易日 × 多个品种”的展期收益率明细（近月/远月、收盘价、RY）。

    参数
    ----
    api : 已登录的 TqApi
    product_ids : 例如 ["rb", "au", "TA"]（注意是 product_id，不含交易所前缀）
    n_trading_days : 交易日数量（默认 10）
    sign_convention :
        - "original" : 保持你原脚本的公式与符号（分母为 (t - far) - (t - near)，通常为负）
        - "backwardation_positive" : 将 Backwardation 定义为 RY>0（更常见）
    klines_window : 为了能覆盖 n_trading_days，日线取样长度（默认 200）

    返回
    ----
    DataFrame 列：
        trading_date, product_id, near_symbol, far_symbol,
        near_close, far_close, near_exp_date, far_exp_date, rolling_yield
    """
    trading_dates = _last_n_trading_dates(api, n=n_trading_days, lookback_days=max(200, n_trading_days * 30))
    results = []

    for pid in product_ids:
        # 1) 列出所有未到期合约
        all_syms = api.query_quotes(ins_class="FUTURE", product_id=pid, expired=False)
        if not all_syms:
            continue

        quotes = api.get_quote_list(symbols=all_syms)
        # 2) 按到期日 + 持仓量排序，拿出候选列表
        ordered = _pick_near_far_by_expiry(quotes)
        if len(ordered) < 2:
            continue

        # 3) 取每个候选的日线（一次性拿足）
        day_bars: Dict[str, pd.DataFrame] = {}
        for sym in ordered:
            try:
                df = api.get_kline_serial(symbol=sym, duration_seconds=86400, data_length=klines_window)
            except Exception:
                df = pd.DataFrame()
            day_bars[sym] = df

        # 4) 对每个交易日，找“当日仍未到期”的近/远两张，并计算 RY
        #    注意：到期日取自 quote；收盘价取自当日日线 close。
        qmap = {q["instrument_id"]: q for q in quotes if q.get("expire_datetime", 0) > 0}
        for td in trading_dates:
            # 仅保留“td 当天还没到期”的合约
            valid = []
            for sym in ordered:
                q = qmap.get(sym, None)
                if not q:
                    continue
                exp = time_to_datetime(q["expire_datetime"]).date()
                if exp > td and not day_bars[sym].empty:
                    # 找到该交易日对应的日线记录
                    df = day_bars[sym].copy()
                    df["date"] = df["datetime"].apply(lambda x: time_to_datetime(x).date())
                    row = df[df["date"] == td]
                    if not row.empty:
                        valid.append((sym, exp, float(row.iloc[-1]["close"]), int(row.iloc[-1]["open_oi"])))

            if len(valid) < 2:
                continue

            # 以到期日 + open_oi 排序，取近/远两张
            valid_df = pd.DataFrame(valid, columns=["symbol", "exp", "close", "open_oi"])
            valid_df = valid_df.sort_values(by=["exp", "open_oi"], ascending=[True, True])
            near = valid_df.iloc[0]
            far = valid_df.iloc[1]

            # 分母：两张到期日的“天数差”
            # 原脚本的分母： (td - far_exp).days - (td - near_exp).days = (near_exp - far_exp).days（通常为负）
            denom = ((td - far["exp"]).days - (td - near["exp"]).days)

            if denom == 0:
                continue

            # 计算展期收益率 (Roll Yield)
            # 使用对数差计算相对收益率：log(near_price) - log(far_price) = log(near_price/far_price)
            # 乘以365.0将日收益率年化，除以天数差得到标准化收益率
            # 对数差的好处：1)标准化为相对收益率而非绝对价格差 2)便于年化处理 3)符合金融学连续复利概念
            ry = (np.log(near["close"]) - np.log(far["close"])) * 365.0 / denom

            # 如需“Backwardation 为正”，翻转一下符号（或改用 |far - near| 做正的分母）
            if sign_convention == "backwardation_positive":
                ry = -ry  # 因为 denom 通常为负号，取负号可令 near>far 时 RY>0

            results.append({
                "trading_date": td,
                "product_id": pid,
                "near_symbol": near["symbol"],
                "far_symbol": far["symbol"],
                "near_close": near["close"],
                "far_close": far["close"],
                "near_exp_date": near["exp"],
                "far_exp_date": far["exp"],
                "rolling_yield": float(ry),
            })

    return pd.DataFrame(results)


def rank_products_by_ry(
    api: TqApi,
    product_ids: List[str],
    n_trading_days: int = 10,
    agg: Literal["mean", "median"] = "mean",
    sign_convention: Literal["original", "backwardation_positive"] = "original",
) -> pd.DataFrame:
    """
    将每个品种最近 n 个交易日的 RY 聚合后排序，便于挑选“高/低 RY”篮子。

    返回列：
        product_id, rolling_yield  （已按降序排好）
    """
    df = compute_rolling_yield_history(
        api,
        product_ids=product_ids,
        n_trading_days=n_trading_days,
        sign_convention=sign_convention,
    )
    if df.empty:
        return pd.DataFrame(columns=["product_id", "rolling_yield"])

    if agg == "median":
        s = df.groupby("product_id")["rolling_yield"].median()
    else:
        s = df.groupby("product_id")["rolling_yield"].mean()

    out = s.sort_values(ascending=False).rename("rolling_yield").reset_index()
    return out


# --------------------------- 使用示例 ---------------------------
api = TqApi(auth=TqAuth(user_name="ringo", password="Shinny456"))
# quote_ls = api.query_cont_quotes()
# symbol_info_df = api.query_symbol_info(quote_ls)
# if "product_id" in symbol_info_df.columns:
#     product_ids = symbol_info_df["product_id"].tolist()

product_ids = ['v', 'ag', 'j', 'jm', 'nr', 'CJ', 'ps', 'SM', 'lg', 'eb', 'IF', 'sn', 'eg', 'RS', 'SF', 'bc', 'lc', 'ru', 'rb', 'y', 'PX', 'sp', 'i', 'IH', 'l', 'fu', 'T', 'IM', 'ec', 'a', 'fb', 'cs', 'OI', 'AP', 'JR', 'hc', 'si', 'rr', 'PM', 'TF', 'wr', 'jd', 'cu', 'al', 'RM', 'PR', 'lu', 'MA', 'bz', 'ad', 'p', 'PK', 'au', 'SA', 'LR', 'sc', 'CF', 'br', 'WH', 'c', 'ss', 'pg', 'bb', 'IC', 'zn', 'UR', 'pb', 'b', 'TA', 'bu', 'CY', 'ao', 'SR', 'lh', 'ZC', 'ni', 'FG', 'PF', 'TS', 'm', 'TL', 'RI', 'pp', 'PL', 'SH']
print(product_ids)

ry_df = compute_rolling_yield_history(api, product_ids, n_trading_days=10,
                                      sign_convention="backwardation_positive")
rank_df = rank_products_by_ry(api, product_ids, n_trading_days=10,
                              agg="mean", sign_convention="backwardation_positive")
# 设置pandas显示选项，显示完整的DataFrame
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 不限制显示宽度
pd.set_option('display.max_colwidth', None)  # 不限制列宽

print("=" * 80)
print("展期收益率排名（完整数据）")
print("=" * 80)
print(rank_df.assign(rolling_yield=rank_df['rolling_yield'].round(2)).to_string(index=False))

print("\n" + "=" * 80)
print("投资建议分析")
print("=" * 80)

# 分析投资建议
if not rank_df.empty:
    # 高RY品种（适合做多）
    high_ry = rank_df[rank_df['rolling_yield'] > 0]
    if not high_ry.empty:
        print(f"\n🔥 高展期收益率品种（建议做多）：")
        print(f"   品种数量：{len(high_ry)}")
        print(f"   平均RY：{high_ry['rolling_yield'].mean():.2f}")
        print(f"   推荐品种：")
        for _, row in high_ry.head(5).iterrows():
            print(f"     • {row['product_id']}: RY = {row['rolling_yield']:.2f}")
    
    # 低RY品种（适合做空）
    low_ry = rank_df[rank_df['rolling_yield'] < 0]
    if not low_ry.empty:
        print(f"\n📉 低展期收益率品种（建议做空）：")
        print(f"   品种数量：{len(low_ry)}")
        print(f"   平均RY：{low_ry['rolling_yield'].mean():.2f}")
        print(f"   推荐品种：")
        for _, row in low_ry.tail(5).iterrows():
            print(f"     • {row['product_id']}: RY = {row['rolling_yield']:.2f}")
    

    
    print(f"\n📊 统计摘要：")
    print(f"   总品种数：{len(rank_df)}")
    print(f"   最高RY：{rank_df['rolling_yield'].max():.2f} ({rank_df.loc[rank_df['rolling_yield'].idxmax(), 'product_id']})")
    print(f"   最低RY：{rank_df['rolling_yield'].min():.2f} ({rank_df.loc[rank_df['rolling_yield'].idxmin(), 'product_id']})")
    print(f"   平均RY：{rank_df['rolling_yield'].mean():.2f}")
    print(f"   中位数RY：{rank_df['rolling_yield'].median():.2f}")
    
    print(f"\n💡 投资策略建议：")
    print(f"   1. 高RY品种：做多，享受现货溢价")
    print(f"   2. 低RY品种：做空，利用远期升水")
    print(f"   3. 可构建配对交易：做多高RY品种 + 做空低RY品种")
    print(f"   4. 定期重新评估，展期收益率会随时间变化")
else:
    print("⚠️  没有获取到展期收益率数据，请检查网络连接和API状态")
api.close()