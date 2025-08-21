# price_tick_utils.py
# -------------------------------------------------------------------
# 公共工具：读取最小变动单位映射、从文件名解析品种代码、按文件获取最小变动单位
# -------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Dict
import pandas as pd


def load_price_tick_mapping(base_dir: str) -> Dict[str, float]:
    """
    从 base_dir/test_data 下读取 `price_tick.xlsx` 或 `price_tick.csv`，
    返回 {product_id -> price_tick} 的映射（大小写不敏感，三份键）。
    """
    test_data_dir = os.path.join(base_dir, "test_data")
    xlsx_path = os.path.join(test_data_dir, "price_tick.xlsx")
    csv_path = os.path.join(test_data_dir, "price_tick.csv")

    mapping: Dict[str, float] = {}

    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path)
        except Exception:
            df = None
    else:
        df = None

    if df is None and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None

    if df is None:
        return {}

    cols = {c.lower(): c for c in df.columns}
    pid_col = cols.get("product_id")
    tick_col = cols.get("price_tick")
    if not pid_col or not tick_col:
        return {}

    for _, row in df[[pid_col, tick_col]].dropna().iterrows():
        pid = str(row[pid_col]).strip()
        if not pid:
            continue
        try:
            val = float(row[tick_col])
        except Exception:
            continue
        mapping[pid] = val
        mapping[pid.upper()] = val
        mapping[pid.lower()] = val

    return mapping


def extract_instrument_from_filename(csv_path: str) -> str:
    """
    从 CSV 文件名提取品种代码（字母部分），例如：
    - "SHFE.rb2510.csv" -> "rb"
    - "CFFEX.IF2509.csv" -> "IF"
    - "rb2510_tick_2025_07_17_2025_07_17.csv" -> "rb"
    - 回退：若无法识别，返回空字符串
    """
    name = os.path.basename(csv_path)
    if name.endswith(".csv"):
        name = name[:-4]

    contract_part = ""
    if "." in name:
        parts = name.split(".")
        if len(parts) >= 2:
            contract_part = parts[1]
    elif "_tick_" in name:
        contract_part = name.split("_tick_")[0]
    else:
        contract_part = name

    letters: list[str] = []
    for ch in contract_part:
        if ch.isalpha():
            letters.append(ch)
        else:
            break
    return "".join(letters)


def get_tick_size_for_csv(csv_path: str, base_dir: str, default: float = 1.0) -> float:
    """
    读取最小变动单位映射并根据文件名识别品种，返回该文件对应品种的最小变动单位。
    若无法识别或缺失，返回 default（默认 1.0）。
    """
    mapping = load_price_tick_mapping(base_dir)
    if not mapping:
        return float(default)
    instrument = extract_instrument_from_filename(csv_path)
    if not instrument:
        return float(default)
    return float(
        mapping.get(instrument)
        or mapping.get(instrument.upper())
        or mapping.get(instrument.lower())
        or default
    )


