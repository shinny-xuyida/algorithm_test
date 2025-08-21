from __future__ import annotations

import os
import sys
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple

# 确保可以从项目根目录导入 `core`
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)

from core.market_data import tick_reader
from core.price_tick_utils import (
	load_price_tick_mapping,
	extract_instrument_from_filename,
	get_tick_size_for_csv,
)


## 使用 core 公共工具，移除本地重复实现


def detect_last_price_column(csv_path: str) -> str | None:

	with open(csv_path, "r", encoding="utf-8") as f:
		header = f.readline().strip()
	if not header:
		return None
	cols = header.split(",")
	# 优先匹配形如 XXX.last_price 的列，否则退回到 last_price
	for c in cols:
		if c.endswith(".last_price"):
			return c
	for c in cols:
		if c == "last_price":
			return c
	return None


def load_last_prices(csv_path: str) -> List[float]:

	col = detect_last_price_column(csv_path)
	if not col:
		return []
	try:
		df = pd.read_csv(csv_path, usecols=[col])
		series = df[col].dropna()
		return [float(v) for v in series.tolist()]
	except Exception:
		# 回退到通用 tick_reader（较慢，但更健壮）
		last_prices: List[float] = []
		for tick in tick_reader(csv_path, tz_offset=8):
			last_prices.append(float(tick.last))
		return last_prices


def classify_move(delta: float) -> str:

	if delta > 0:
		return "up"
	elif delta < 0:
		return "down"
	else:
		return "flat"


def analyze_moves(last_prices: List[float], threshold_map: Dict[int, float], direction: str = "buy") -> Dict[int, Dict[str, int]]:

	results: Dict[int, Dict[str, int]] = {}
	for label, threshold in threshold_map.items():
		counter: Counter[str] = Counter()
		trigger_count = 0

		for i in range(len(last_prices) - 2):
			delta1 = last_prices[i + 1] - last_prices[i]
			# 方向触发：买入 (T+1 - T) > 阈值；卖出 (T+1 - T) < -阈值
			if (direction == "buy" and delta1 > threshold) or (direction == "sell" and delta1 < -threshold):
				trigger_count += 1
				delta2 = last_prices[i + 2] - last_prices[i + 1]
				counter[classify_move(delta2)] += 1

		# 保存统计数据（按倍数标签聚合，例如 2,3,4,5 或 -2,-3,-4,-5）
		results[label] = {
			"triggers": trigger_count,
			"up": counter.get("up", 0),
			"flat": counter.get("flat", 0),
			"down": counter.get("down", 0),
		}

	return results


def print_summary(results: Dict[int, Dict[str, int]]) -> None:

	print("AO 合约最小变动单位: 1")
	print("触发条件: (T+1 - T) > 阈值，只考虑买入方向 (阈值 ∈ {2,3,4,5})")
	print()
	for threshold in sorted(results.keys()):
		stat = results[threshold]
		n = stat["triggers"]
		if n == 0:
			print(f"阈值={threshold}: 无触发样本")
			continue
		up = stat["up"]
		flat = stat["flat"]
		down = stat["down"]
		print(
			f"阈值={threshold} | 触发数={n} | 上涨={up} ({up/n:.1%}) | 不动={flat} ({flat/n:.1%}) | 下跌={down} ({down/n:.1%})"
		)


def list_all_csv_files(csv_dir: str) -> List[str]:

	files: List[str] = []
	for name in os.listdir(csv_dir):
		if name.lower().endswith(".csv"):
			files.append(os.path.join(csv_dir, name))
	return sorted(files)


def aggregate_results(all_results, labels: List[int], direction: str) -> Dict[int, Dict[str, int]]:

	agg: Dict[int, Dict[str, int]] = {t: {"triggers": 0, "up": 0, "flat": 0, "down": 0} for t in labels}
	for _fname, res in all_results.items():
		sub = res.get(direction, {})
		for t in labels:
			if t in sub:
				agg[t]["triggers"] += sub[t]["triggers"]
				agg[t]["up"] += sub[t]["up"]
				agg[t]["flat"] += sub[t]["flat"]
				agg[t]["down"] += sub[t]["down"]
	return agg


def print_overall_summary(agg: Dict[int, Dict[str, int]], direction: str) -> None:

	label = "买入" if direction == "buy" else "卖出"
	print(f"================ 总体汇总（{label}） ================")
	for k in sorted(agg.keys(), key=lambda x: (abs(x), x)):
		stat = agg[k]
		n = stat["triggers"]
		if n == 0:
			print(f"阈值={k}x: 无触发样本")
			continue
		up = stat["up"]
		flat = stat["flat"]
		down = stat["down"]
		print(
			f"阈值={k}x | 触发数={n} | 上涨={up} ({up/n:.1%}) | 不动={flat} ({flat/n:.1%}) | 下跌={down} ({down/n:.1%})"
		)


def print_per_file_brief(all_results, direction: str) -> None:

	label = "买入" if direction == "buy" else "卖出"
	print(f"================ 按文件明细（{label}） ================")
	for fname in sorted(all_results.keys()):
		entry = all_results[fname]
		res = entry.get(direction, {})
		inst = entry.get("_instrument", "-")
		tick = entry.get("_tick", "-")
		brief = []
		# 固定倍数顺序：买入 2~5，卖出 -2~-5
		labels = [2,3,4,5] if direction == "buy" else [-2,-3,-4,-5]
		for k in labels:
			stat = res.get(k)
			if not stat or stat["triggers"] == 0:
				brief.append(f"{k}x:0")
				continue
			brief.append(f"{k}x:{stat['triggers']}")
		print(f"{os.path.basename(fname)} [inst={inst}, tick={tick}] | " + ", ".join(brief))


def main() -> None:

	base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	csv_dir = os.path.join(base_dir, "test_data")
	if not os.path.isdir(csv_dir):
		raise FileNotFoundError(f"未找到数据目录: {csv_dir}")

	# 读取品种最小变动单位映射（公共函数）
	price_tick_map = load_price_tick_mapping(base_dir)
	all_results: Dict[str, Dict[float, Dict[str, int]]] = {}

	files = list_all_csv_files(csv_dir)
	if not files:
		raise RuntimeError("test_data 目录下未找到任何 CSV 文件")

	print("触发条件: 买入测试阈值 = 2~5 × 最小变动单位；卖出测试阈值 = -2~-5 × 最小变动单位（按文件所属品种自动识别）")
	print()

	for csv_path in files:
		try:
			last_prices = load_last_prices(csv_path)
			if len(last_prices) < 3:
				continue
			# 根据文件获取最小变动单位（公共函数，内部已做回退）
			tick = get_tick_size_for_csv(csv_path, base_dir, default=1.0)
			instrument = extract_instrument_from_filename(csv_path)
			# 构建买入与卖出阈值映射（以倍数为标签，聚合时一致）
			buy_thresholds = {2: 2.0 * tick, 3: 3.0 * tick, 4: 4.0 * tick, 5: 5.0 * tick}
			sell_thresholds = {-2: 2.0 * tick, -3: 3.0 * tick, -4: 4.0 * tick, -5: 5.0 * tick}
			res_buy = analyze_moves(last_prices, buy_thresholds, direction="buy")
			res_sell = analyze_moves(last_prices, sell_thresholds, direction="sell")
			all_results[csv_path] = {"buy": res_buy, "sell": res_sell, "_tick": tick, "_instrument": instrument}
		except Exception as e:
			print(f"文件处理失败: {os.path.basename(csv_path)} | {e}")

	if not all_results:
		raise RuntimeError("没有任何文件得到有效统计结果")

	# 输出总体与按文件结果
	agg_buy = aggregate_results(all_results, labels=[2,3,4,5], direction="buy")
	print_overall_summary(agg_buy, direction="buy")
	print()
	print_per_file_brief(all_results, direction="buy")
	print()
	agg_sell = aggregate_results(all_results, labels=[-2,-3,-4,-5], direction="sell")
	print_overall_summary(agg_sell, direction="sell")
	print()
	print_per_file_brief(all_results, direction="sell")


if __name__ == "__main__":

	main()


