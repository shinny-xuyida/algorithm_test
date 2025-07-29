#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盘口失衡信号评估程序

本程序使用多进程来遍历test_data文件夹中的数据文件，
分析盘口失衡情况下后续3秒价格移动的有效性。

失衡定义参考ice_smart_only_imbalance.py：
Q = (买一量-卖一量)/(买一量+卖一量)

信号判断逻辑：
- 有效信号：失衡方向与后续价格移动方向一致
- 反向信号：失衡方向与后续价格移动方向相反  
- 无效信号：价格既有上涨又有下跌，或完全无变化
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块
from core.market_data import tick_reader, Tick


@dataclass
class ImbalanceSignal:
    """失衡信号数据结构"""
    timestamp: pd.Timestamp
    imbalance_value: float  # Q值 
    imbalance_direction: str  # 'buy_imbalance' 或 'sell_imbalance'
    current_price: float
    future_prices: List[float]  # 后续3秒内的价格
    signal_type: str  # 'valid', 'reverse', 'invalid'


@dataclass
class FileResult:
    """单个文件的分析结果"""
    filename: str
    total_signals: int
    valid_signals: int
    reverse_signals: int
    invalid_signals: int
    buy_imbalance_count: int
    sell_imbalance_count: int
    processing_time: float
    error_msg: Optional[str] = None


class ImbalanceAnalyzer:
    """盘口失衡分析器"""
    
    def __init__(self, imbalance_threshold: float = 0.2, future_window_seconds: int = 3, verbose: bool = True):
        """
        初始化分析器
        
        Args:
            imbalance_threshold: 失衡阈值，默认0.2
            future_window_seconds: 未来价格观察窗口（秒），默认3秒
            verbose: 是否输出详细信息
        """
        self.threshold = imbalance_threshold
        self.window = future_window_seconds
        self.verbose = verbose
    
    def calculate_imbalance(self, tick: Tick) -> Tuple[float, bool, str]:
        """
        计算订单失衡指标
        
        Args:
            tick: 市场数据tick
            
        Returns:
            (Q值, 是否失衡, 失衡方向)
        """
        bid_vol = tick.bid_volume
        ask_vol = tick.ask_volume
        
        # 避免除零错误
        if bid_vol + ask_vol == 0:
            return 0.0, False, ""
        
        # 计算订单失衡指标 Q = (B-A)/(B+A)
        Q = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
        # 判断失衡方向
        if Q > self.threshold:
            return Q, True, "buy_imbalance"
        elif Q < -self.threshold:
            return Q, True, "sell_imbalance"
        else:
            return Q, False, ""
    
    def evaluate_signal(self, current_price: float, future_prices: List[float], 
                       imbalance_direction: str) -> str:
        """
        评估信号有效性
        
        使用更合理的评估逻辑：
        1. 比较最终价格与初始价格的关系
        2. 考虑价格变化的主导方向
        3. 允许正常的价格波动
        
        Args:
            current_price: 当前价格
            future_prices: 后续价格列表
            imbalance_direction: 失衡方向
            
        Returns:
            信号类型: 'valid', 'reverse', 'invalid'
        """
        if not future_prices:
            return 'invalid'
        
        # 方法1: 比较最终价格与初始价格
        final_price = future_prices[-1]
        price_change = final_price - current_price
        
        # 方法2: 统计价格变化方向的主导性
        higher_count = sum(1 for p in future_prices if p > current_price)
        lower_count = sum(1 for p in future_prices if p < current_price)
        equal_count = len(future_prices) - higher_count - lower_count
        
        # 方法3: 计算加权平均价格变化
        total_ticks = len(future_prices)
        net_direction_ratio = (higher_count - lower_count) / total_ticks if total_ticks > 0 else 0
        
        # 综合判断逻辑
        if imbalance_direction == "buy_imbalance":
            # 买方失衡：期望价格上涨
            
            # 主要条件：最终价格上涨
            final_up = price_change > 0
            
            # 次要条件：主导方向向上（上涨次数明显多于下跌次数）
            direction_up = net_direction_ratio > 0.3  # 至少30%的净向上倾斜
            
            # 强信号：最终价格上涨且主导方向向上
            if final_up and direction_up:
                return 'valid'
            # 弱信号：仅最终价格上涨或仅主导方向向上
            elif final_up or (net_direction_ratio > 0.1):
                return 'valid'
            # 明确反向：最终价格下跌且主导方向向下
            elif price_change < 0 and net_direction_ratio < -0.1:
                return 'reverse'
            else:
                return 'invalid'
                
        elif imbalance_direction == "sell_imbalance":
            # 卖方失衡：期望价格下跌
            
            # 主要条件：最终价格下跌
            final_down = price_change < 0
            
            # 次要条件：主导方向向下
            direction_down = net_direction_ratio < -0.3  # 至少30%的净向下倾斜
            
            # 强信号：最终价格下跌且主导方向向下
            if final_down and direction_down:
                return 'valid'
            # 弱信号：仅最终价格下跌或仅主导方向向下
            elif final_down or (net_direction_ratio < -0.1):
                return 'valid'
            # 明确反向：最终价格上涨且主导方向向上
            elif price_change > 0 and net_direction_ratio > 0.1:
                return 'reverse'
            else:
                return 'invalid'
        
        return 'invalid'
    
    def analyze_file(self, file_path: str) -> FileResult:
        """
        分析单个数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            分析结果
        """
        start_time = time.time()
        filename = os.path.basename(file_path)
        
        try:
            # 读取tick数据，禁用tick_reader的print输出
            import contextlib
            import io
            
            if not self.verbose:
                # 捕获并丢弃print输出
                with contextlib.redirect_stdout(io.StringIO()):
                    ticks = list(tick_reader(file_path))
            else:
                ticks = list(tick_reader(file_path))
                
            if len(ticks) < 10:  # 数据太少，跳过
                return FileResult(
                    filename=filename,
                    total_signals=0,
                    valid_signals=0,
                    reverse_signals=0,
                    invalid_signals=0,
                    buy_imbalance_count=0,
                    sell_imbalance_count=0,
                    processing_time=time.time() - start_time,
                    error_msg="数据量太少"
                )
            
            # 存储所有失衡信号
            signals = []
            
            # 遍历tick数据，检测失衡信号
            for i, tick in enumerate(ticks):
                Q, is_imbalanced, direction = self.calculate_imbalance(tick)
                
                if is_imbalanced:
                    # 收集后续3秒内的价格
                    future_prices = []
                    current_time = tick.ts
                    
                    # 向前查找3秒内的价格
                    for j in range(i + 1, len(ticks)):
                        future_tick = ticks[j]
                        time_diff = (future_tick.ts - current_time).total_seconds()
                        
                        if time_diff > self.window:
                            break
                        
                        future_prices.append(future_tick.last)
                    
                    # 评估信号
                    signal_type = self.evaluate_signal(tick.last, future_prices, direction)
                    
                    signals.append(ImbalanceSignal(
                        timestamp=tick.ts,
                        imbalance_value=Q,
                        imbalance_direction=direction,
                        current_price=tick.last,
                        future_prices=future_prices,
                        signal_type=signal_type
                    ))
            
            # 统计结果
            total_signals = len(signals)
            valid_signals = sum(1 for s in signals if s.signal_type == 'valid')
            reverse_signals = sum(1 for s in signals if s.signal_type == 'reverse')
            invalid_signals = sum(1 for s in signals if s.signal_type == 'invalid')
            buy_imbalance_count = sum(1 for s in signals if s.imbalance_direction == 'buy_imbalance')
            sell_imbalance_count = sum(1 for s in signals if s.imbalance_direction == 'sell_imbalance')
            
            return FileResult(
                filename=filename,
                total_signals=total_signals,
                valid_signals=valid_signals,
                reverse_signals=reverse_signals,
                invalid_signals=invalid_signals,
                buy_imbalance_count=buy_imbalance_count,
                sell_imbalance_count=sell_imbalance_count,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return FileResult(
                filename=filename,
                total_signals=0,
                valid_signals=0,
                reverse_signals=0,
                invalid_signals=0,
                buy_imbalance_count=0,
                sell_imbalance_count=0,
                processing_time=time.time() - start_time,
                error_msg=str(e)
            )


def process_single_file_silent(args: Tuple[str, float, int]) -> FileResult:
    """
    处理单个文件的工作函数（用于多进程，静默模式）
    
    Args:
        args: (文件路径, 失衡阈值, 时间窗口)
        
    Returns:
        分析结果
    """
    file_path, threshold, window = args
    analyzer = ImbalanceAnalyzer(threshold, window, verbose=False)
    return analyzer.analyze_file(file_path)


@dataclass
class ThresholdResult:
    """阈值测试结果"""
    threshold: float
    total_signals: int
    valid_signals: int
    reverse_signals: int
    invalid_signals: int
    valid_ratio: float
    reverse_ratio: float
    processing_time: float


class DataEvaluator:
    """数据评估主类"""
    
    def __init__(self, data_dir: str = "test_data", 
                 future_window_seconds: int = 3,
                 n_processes: Optional[int] = None):
        """
        初始化数据评估器
        
        Args:
            data_dir: 数据文件夹路径
            future_window_seconds: 未来观察窗口（秒）
            n_processes: 进程数，默认为CPU核心数-1
        """
        self.data_dir = data_dir
        self.window = future_window_seconds
        self.n_processes = n_processes or max(1, cpu_count() - 1)
    
    def get_data_files(self) -> List[str]:
        """获取所有数据文件路径"""
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        csv_files = list(data_path.glob("*.csv"))
        return [str(f) for f in csv_files]
    
    def test_threshold_range(self, 
                           threshold_start: float = 0.2, 
                           threshold_end: float = 0.8, 
                           threshold_step: float = 0.05,
                           use_multiprocess: bool = True) -> List[ThresholdResult]:
        """
        测试不同失衡阈值的表现
        
        Args:
            threshold_start: 起始阈值
            threshold_end: 结束阈值
            threshold_step: 阈值步长
            use_multiprocess: 是否使用多进程
            
        Returns:
            每个阈值的测试结果
        """
        files = self.get_data_files()
        if not files:
            print("❌ 未找到数据文件")
            return []
        
        # 生成阈值序列
        thresholds = np.arange(threshold_start, threshold_end + threshold_step, threshold_step)
        thresholds = np.round(thresholds, 3)  # 避免浮点数精度问题
        
        print(f"🔍 阈值范围测试")
        print(f"📁 数据文件: {len(files)}个")
        print(f"📊 测试阈值: {threshold_start} → {threshold_end} (步长: {threshold_step})")
        print(f"⚙️  {'多进程' if use_multiprocess else '单进程'}模式")
        if use_multiprocess:
            print(f"🚀 使用 {self.n_processes} 个进程")
        print()
        
        results = []
        
        for i, threshold in enumerate(thresholds, 1):
            print(f"📈 测试阈值 {i}/{len(thresholds)}: {threshold:.3f}", end=" ", flush=True)
            
            start_time = time.time()
            
            if use_multiprocess and len(files) > 1:
                # 多进程处理
                args_list = [(f, threshold, self.window) for f in files]
                
                with Pool(processes=self.n_processes) as pool:
                    file_results = pool.map(process_single_file_silent, args_list)
            else:
                # 单进程处理
                analyzer = ImbalanceAnalyzer(threshold, self.window, verbose=False)
                file_results = []
                
                for file_path in files:
                    result = analyzer.analyze_file(file_path)
                    file_results.append(result)
            
            # 汇总结果
            valid_results = [r for r in file_results if r.error_msg is None]
            
            total_signals = sum(r.total_signals for r in valid_results)
            total_valid = sum(r.valid_signals for r in valid_results)
            total_reverse = sum(r.reverse_signals for r in valid_results)
            total_invalid = sum(r.invalid_signals for r in valid_results)
            
            valid_ratio = total_valid / total_signals if total_signals > 0 else 0
            reverse_ratio = total_reverse / total_signals if total_signals > 0 else 0
            
            processing_time = time.time() - start_time
            
            result = ThresholdResult(
                threshold=threshold,
                total_signals=total_signals,
                valid_signals=total_valid,
                reverse_signals=total_reverse,
                invalid_signals=total_invalid,
                valid_ratio=valid_ratio,
                reverse_ratio=reverse_ratio,
                processing_time=processing_time
            )
            
            results.append(result)
            
            print(f"✅ {processing_time:.1f}s | 信号:{total_signals:,} | 有效:{valid_ratio:.1%} | 反向:{reverse_ratio:.1%}")
        
        return results
    
    def export_results_to_csv(self, results: List[ThresholdResult], filename: str = "threshold_analysis.csv"):
        """
        导出结果到CSV文件
        
        Args:
            results: 测试结果列表
            filename: 输出文件名
        """
        if not results:
            print("❌ 没有结果可导出")
            return
        
        data = []
        for r in results:
            data.append({
                'threshold': r.threshold,
                'total_signals': r.total_signals,
                'valid_signals': r.valid_signals,
                'reverse_signals': r.reverse_signals,
                'invalid_signals': r.invalid_signals,
                'valid_ratio': r.valid_ratio,
                'reverse_ratio': r.reverse_ratio,
                'processing_time': r.processing_time
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"📄 结果已导出到: {filename}")
    
    def print_threshold_summary(self, results: List[ThresholdResult]):
        """打印阈值测试摘要"""
        if not results:
            print("❌ 没有结果可显示")
            return
        
        print(f"\n{'='*80}")
        print("📊 阈值测试摘要")
        print(f"{'='*80}")
        
        # 表头
        print(f"{'阈值':<8} {'总信号':<10} {'有效率':<8} {'反向率':<8} {'信号质量':<10}")
        print("-" * 60)
        
        # 数据行
        for r in results:
            quality = ""
            if r.valid_ratio > 0.4:
                quality = "优秀"
            elif r.valid_ratio > 0.35:
                quality = "良好"
            elif r.valid_ratio > 0.3:
                quality = "一般"
            else:
                quality = "较差"
            
            print(f"{r.threshold:<8.3f} {r.total_signals:<10,} {r.valid_ratio:<8.1%} {r.reverse_ratio:<8.1%} {quality:<10}")
        
        # 找出最佳阈值
        best_result = max(results, key=lambda x: x.valid_ratio)
        print(f"\n🎯 最佳阈值: {best_result.threshold:.3f}")
        print(f"   有效率: {best_result.valid_ratio:.1%}")
        print(f"   信号数: {best_result.total_signals:,}")
        
        # 作图数据输出
        print(f"\n📈 作图数据 (阈值, 有效率, 反向率):")
        print("thresholds = [", end="")
        print(", ".join(f"{r.threshold:.3f}" for r in results), end="")
        print("]")
        
        print("valid_ratios = [", end="")
        print(", ".join(f"{r.valid_ratio:.4f}" for r in results), end="")
        print("]")
        
        print("reverse_ratios = [", end="")
        print(", ".join(f"{r.reverse_ratio:.4f}" for r in results), end="")
        print("]")


def main():
    """主函数"""
    print("🔍 盘口失衡信号阈值优化系统")
    print("="*50)
    
    # 创建评估器
    evaluator = DataEvaluator(
        data_dir="test_data",
        future_window_seconds=3,
        n_processes=None  # 自动选择进程数
    )
    
    # 测试阈值范围
    results = evaluator.test_threshold_range(
        threshold_start=0.2,
        threshold_end=0.8,
        threshold_step=0.05,
        use_multiprocess=True
    )
    
    # 显示结果
    evaluator.print_threshold_summary(results)
    
    # 导出CSV
    evaluator.export_results_to_csv(results)
    
    print(f"\n🎉 阈值优化完成!")
    print("💡 提示:")
    print("  - 可以使用导出的CSV文件进行进一步分析")
    print("  - 建议选择有效率最高且信号数足够的阈值")
    print("  - 可以用上述数据在Python中绘制曲线图")


if __name__ == "__main__":
    main()
