#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盘口失衡信号评估程序

本程序使用多进程来遍历test_data文件夹中的数据文件，
分析盘口失衡情况下后续3秒内第一次价格变动的有效性。

失衡定义参考ice_smart_only_imbalance.py：
Q = (买一量-卖一量)/(买一量+卖一量)

信号触发条件：
- 买入信号：Q > 阈值 且 买一量 >= 5手
- 卖出信号：Q < -阈值 且 卖一量 >= 5手

信号判断逻辑（基于3秒内第一次价格变动）：
- 有效信号：买入信号第一次变动上涨，或卖出信号第一次变动下跌
- 反向信号：买入信号第一次变动下跌，或卖出信号第一次变动上涨
- 无效信号：3秒内价格完全无变化
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
    # 买入信号统计
    buy_valid_signals: int
    buy_reverse_signals: int
    buy_invalid_signals: int
    # 卖出信号统计
    sell_valid_signals: int
    sell_reverse_signals: int
    sell_invalid_signals: int
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
        
        # 判断失衡方向，同时检查量的要求
        if Q > self.threshold:
            # 买入信号：需要买一量≥5手
            if bid_vol >= 5:
                return Q, True, "buy_imbalance"
            else:
                return Q, False, ""
        elif Q < -self.threshold:
            # 卖出信号：需要卖一量≥5手
            if ask_vol >= 5:
                return Q, True, "sell_imbalance"
            else:
                return Q, False, ""
        else:
            return Q, False, ""
    
    def evaluate_signal(self, current_price: float, future_prices: List[float], 
                       imbalance_direction: str) -> str:
        """
        评估信号有效性
        
        新的评估逻辑：
        1. 基于3秒内第一次价格变动来判断信号有效性
        2. 买入信号：第一次变动是上涨则有效，下跌则反向
        3. 卖出信号：第一次变动是下跌则有效，上涨则反向
        4. 如果3秒内价格都没有变化，则认为是无效信号
        
        Args:
            current_price: 当前价格
            future_prices: 后续价格列表
            imbalance_direction: 失衡方向
            
        Returns:
            信号类型: 'valid', 'reverse', 'invalid'
        """
        if not future_prices:
            return 'invalid'
        
        # 找到第一次价格变动
        first_change_direction = None
        for price in future_prices:
            if price > current_price:
                first_change_direction = 'up'
                break
            elif price < current_price:
                first_change_direction = 'down'
                break
        
        # 如果3秒内价格都没有变化，认为是无效信号
        if first_change_direction is None:
            return 'invalid'
        
        # 根据失衡方向和第一次价格变动方向判断信号有效性
        if imbalance_direction == "buy_imbalance":
            # 买入信号：期望价格上涨
            if first_change_direction == 'up':
                return 'valid'
            elif first_change_direction == 'down':
                return 'reverse'
        elif imbalance_direction == "sell_imbalance":
            # 卖出信号：期望价格下跌
            if first_change_direction == 'down':
                return 'valid'
            elif first_change_direction == 'up':
                return 'reverse'
        
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
                    buy_valid_signals=0,
                    buy_reverse_signals=0,
                    buy_invalid_signals=0,
                    sell_valid_signals=0,
                    sell_reverse_signals=0,
                    sell_invalid_signals=0,
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
            
            # 买入信号分别统计
            buy_signals = [s for s in signals if s.imbalance_direction == 'buy_imbalance']
            buy_valid_signals = sum(1 for s in buy_signals if s.signal_type == 'valid')
            buy_reverse_signals = sum(1 for s in buy_signals if s.signal_type == 'reverse')
            buy_invalid_signals = sum(1 for s in buy_signals if s.signal_type == 'invalid')
            
            # 卖出信号分别统计
            sell_signals = [s for s in signals if s.imbalance_direction == 'sell_imbalance']
            sell_valid_signals = sum(1 for s in sell_signals if s.signal_type == 'valid')
            sell_reverse_signals = sum(1 for s in sell_signals if s.signal_type == 'reverse')
            sell_invalid_signals = sum(1 for s in sell_signals if s.signal_type == 'invalid')
            
            return FileResult(
                filename=filename,
                total_signals=total_signals,
                valid_signals=valid_signals,
                reverse_signals=reverse_signals,
                invalid_signals=invalid_signals,
                buy_imbalance_count=buy_imbalance_count,
                sell_imbalance_count=sell_imbalance_count,
                buy_valid_signals=buy_valid_signals,
                buy_reverse_signals=buy_reverse_signals,
                buy_invalid_signals=buy_invalid_signals,
                sell_valid_signals=sell_valid_signals,
                sell_reverse_signals=sell_reverse_signals,
                sell_invalid_signals=sell_invalid_signals,
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
                buy_valid_signals=0,
                buy_reverse_signals=0,
                buy_invalid_signals=0,
                sell_valid_signals=0,
                sell_reverse_signals=0,
                sell_invalid_signals=0,
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


def process_contract_thresholds_silent(args: Tuple[str, List[str], List[float], int]) -> List[Tuple[str, float, FileResult]]:
    """
    处理单个合约的所有阈值测试（用于多进程，合约级别并行）
    
    Args:
        args: (合约名称, 合约文件列表, 阈值列表, 时间窗口)
        
    Returns:
        该合约所有阈值的结果列表: [(合约名, 阈值, 结果), ...]
    """
    contract_name, contract_files, thresholds, window = args
    results = []
    
    for threshold in thresholds:
        for file_path in contract_files:
            analyzer = ImbalanceAnalyzer(threshold, window, verbose=False)
            file_result = analyzer.analyze_file(file_path)
            results.append((contract_name, threshold, file_result))
    
    return results


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


@dataclass
class ContractResult:
    """单个合约的分析结果"""
    contract_name: str
    threshold: float
    total_signals: int
    valid_signals: int
    reverse_signals: int
    invalid_signals: int
    valid_ratio: float
    reverse_ratio: float
    invalid_ratio: float
    # 买入信号统计
    buy_signals: int
    buy_valid_signals: int
    buy_reverse_signals: int
    buy_invalid_signals: int
    buy_valid_ratio: float
    buy_reverse_ratio: float
    buy_invalid_ratio: float
    # 卖出信号统计
    sell_signals: int
    sell_valid_signals: int
    sell_reverse_signals: int
    sell_invalid_signals: int
    sell_valid_ratio: float
    sell_reverse_ratio: float
    sell_invalid_ratio: float
    processing_time: float


@dataclass
class ContractSummary:
    """合约汇总结果"""
    contract_name: str
    avg_valid_ratio: float
    avg_reverse_ratio: float
    avg_invalid_ratio: float
    total_signals_all_thresholds: int
    threshold_count: int


@dataclass
class BuySignalSummary:
    """买入信号汇总结果"""
    contract_name: str
    buy_valid_ratio: float
    buy_reverse_ratio: float
    buy_invalid_ratio: float
    total_buy_signals: int
    
    
@dataclass
class SellSignalSummary:
    """卖出信号汇总结果"""
    contract_name: str
    sell_valid_ratio: float
    sell_reverse_ratio: float
    sell_invalid_ratio: float
    total_sell_signals: int


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
    
    def extract_contract_name(self, file_path: str) -> str:
        """从文件路径中提取合约名称"""
        filename = os.path.basename(file_path)
        # 移除.csv扩展名
        contract_name = filename.replace('.csv', '')
        return contract_name
    
    def group_files_by_contract(self, files: List[str]) -> Dict[str, List[str]]:
        """按合约名称分组文件"""
        contract_groups = defaultdict(list)
        for file_path in files:
            contract_name = self.extract_contract_name(file_path)
            contract_groups[contract_name].append(file_path)
        return dict(contract_groups)
    
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
    
    def test_contract_performance(self, 
                                 buy_threshold_start: float = 0.2,
                                 buy_threshold_end: float = 0.8,
                                 sell_threshold_start: float = -0.2,
                                 sell_threshold_end: float = -0.8,
                                 threshold_step: float = 0.05,
                                 use_multiprocess: bool = True) -> Tuple[List[ContractResult], List[ContractSummary]]:
        """
        按合约测试买入和卖出信号的表现
        
        Args:
            buy_threshold_start: 买入信号起始阈值（正值）
            buy_threshold_end: 买入信号结束阈值（正值）
            sell_threshold_start: 卖出信号起始阈值（负值）
            sell_threshold_end: 卖出信号结束阈值（负值）
            threshold_step: 阈值步长
            use_multiprocess: 是否使用多进程
            
        Returns:
            (详细结果列表, 汇总结果列表)
        """
        files = self.get_data_files()
        if not files:
            print("❌ 未找到数据文件")
            return [], []
        
        # 按合约分组
        contract_groups = self.group_files_by_contract(files)
        
        # 生成阈值序列（正值：买入信号，负值：卖出信号）
        buy_thresholds = np.arange(buy_threshold_start, buy_threshold_end + threshold_step, threshold_step)
        sell_thresholds = np.arange(sell_threshold_start, sell_threshold_end - threshold_step, -threshold_step)
        all_thresholds = np.concatenate([buy_thresholds, sell_thresholds])
        all_thresholds = np.round(all_thresholds, 3)  # 避免浮点数精度问题
        
        print(f"🔍 按合约测试买入和卖出信号强度")
        print(f"📁 合约数量: {len(contract_groups)}个")
        print(f"📊 买入阈值: {buy_threshold_start} → {buy_threshold_end}")
        print(f"📊 卖出阈值: {sell_threshold_start} → {sell_threshold_end}")
        print(f"📊 阈值步长: {threshold_step}")
        print(f"⚙️  {'多进程' if use_multiprocess else '单进程'}模式")
        if use_multiprocess:
            print(f"🚀 使用 {self.n_processes} 个进程")
        print()
        
        all_contract_results = []
        contract_summaries = []
        
        # 遍历每个合约
        for contract_idx, (contract_name, contract_files) in enumerate(contract_groups.items(), 1):
            print(f"📈 处理合约 {contract_idx}/{len(contract_groups)}: {contract_name}")
            
            contract_threshold_results = []
            
            # 遍历所有阈值
            for threshold_idx, threshold in enumerate(all_thresholds, 1):
                threshold_type = "买入" if threshold > 0 else "卖出"
                print(f"  🔍 测试{threshold_type}阈值 {threshold_idx}/{len(all_thresholds)}: {threshold:.3f}", end=" ", flush=True)
                
                start_time = time.time()
                
                if use_multiprocess and len(contract_files) > 1:
                    # 多进程处理当前合约的所有文件
                    args_list = [(f, threshold, self.window) for f in contract_files]
                    
                    with Pool(processes=self.n_processes) as pool:
                        file_results = pool.map(process_single_file_silent, args_list)
                else:
                    # 单进程处理
                    analyzer = ImbalanceAnalyzer(threshold, self.window, verbose=False)
                    file_results = []
                    
                    for file_path in contract_files:
                        result = analyzer.analyze_file(file_path)
                        file_results.append(result)
                
                # 汇总当前合约在当前阈值下的结果
                valid_results = [r for r in file_results if r.error_msg is None]
                
                total_signals = sum(r.total_signals for r in valid_results)
                total_valid = sum(r.valid_signals for r in valid_results)
                total_reverse = sum(r.reverse_signals for r in valid_results)
                total_invalid = sum(r.invalid_signals for r in valid_results)
                
                valid_ratio = total_valid / total_signals if total_signals > 0 else 0
                reverse_ratio = total_reverse / total_signals if total_signals > 0 else 0
                invalid_ratio = total_invalid / total_signals if total_signals > 0 else 0
                
                processing_time = time.time() - start_time
                
                contract_result = ContractResult(
                    contract_name=contract_name,
                    threshold=threshold,
                    total_signals=total_signals,
                    valid_signals=total_valid,
                    reverse_signals=total_reverse,
                    invalid_signals=total_invalid,
                    valid_ratio=valid_ratio,
                    reverse_ratio=reverse_ratio,
                    invalid_ratio=invalid_ratio,
                    processing_time=processing_time
                )
                
                all_contract_results.append(contract_result)
                contract_threshold_results.append(contract_result)
                
                print(f"✅ {processing_time:.1f}s | 信号:{total_signals:,} | 有效:{valid_ratio:.1%}")
            
            # 计算当前合约的平均表现
            if contract_threshold_results:
                avg_valid_ratio = np.mean([r.valid_ratio for r in contract_threshold_results])
                avg_reverse_ratio = np.mean([r.reverse_ratio for r in contract_threshold_results])
                avg_invalid_ratio = np.mean([r.invalid_ratio for r in contract_threshold_results])
                total_signals_all = sum(r.total_signals for r in contract_threshold_results)
                
                contract_summary = ContractSummary(
                    contract_name=contract_name,
                    avg_valid_ratio=avg_valid_ratio,
                    avg_reverse_ratio=avg_reverse_ratio,
                    avg_invalid_ratio=avg_invalid_ratio,
                    total_signals_all_thresholds=total_signals_all,
                    threshold_count=len(all_thresholds)
                )
                
                contract_summaries.append(contract_summary)
            
            print(f"  ✅ 合约 {contract_name} 完成\n")
        
        return all_contract_results, contract_summaries
    
    def test_contract_simple(self, 
                           buy_threshold: float = 0.8,
                           sell_threshold: float = -0.8,
                           use_multiprocess: bool = True,
                           parallel_strategy: str = "contract") -> Tuple[List[ContractResult], List[ContractSummary]]:
        """
        简化版按合约测试：只测试固定的买入和卖出阈值
        
        Args:
            buy_threshold: 买入信号阈值（正值）
            sell_threshold: 卖出信号阈值（负值）
            use_multiprocess: 是否使用多进程
            parallel_strategy: 并行策略 ("contract": 合约级并行, "file": 文件级并行)
            
        Returns:
            (详细结果列表, 汇总结果列表)
        """
        files = self.get_data_files()
        if not files:
            print("❌ 未找到数据文件")
            return [], []
        
        # 按合约分组
        contract_groups = self.group_files_by_contract(files)
        
        # 固定的两个阈值
        thresholds = [buy_threshold, sell_threshold]
        
        print(f"🔍 简化版按合约测试")
        print(f"📁 合约数量: {len(contract_groups)}个")
        print(f"📊 买入阈值: {buy_threshold}")
        print(f"📊 卖出阈值: {sell_threshold}")
        print(f"⚙️  {'多进程' if use_multiprocess else '单进程'}模式")
        if use_multiprocess:
            print(f"🚀 使用 {self.n_processes} 个进程")
        print()
        
        all_contract_results = []
        contract_summaries = []
        
        if use_multiprocess and parallel_strategy == "contract" and len(contract_groups) > 1:
            # 🚀 新策略：合约级并行处理
            print(f"🚀 合约级并行处理 (同时处理 {min(self.n_processes, len(contract_groups))} 个合约)")
            
            start_time_total = time.time()
            
            # 准备多进程参数：每个进程处理一个合约的所有阈值
            args_list = []
            for contract_name, contract_files in contract_groups.items():
                args_list.append((contract_name, contract_files, thresholds, self.window))
            
            # 执行多进程
            with Pool(processes=min(self.n_processes, len(contract_groups))) as pool:
                # 每个进程返回一个合约的所有结果
                contract_results_list = pool.map(process_contract_thresholds_silent, args_list)
            
            # 处理结果
            for contract_results in contract_results_list:
                contract_threshold_results = []
                
                for contract_name, threshold, file_result in contract_results:
                    # 构建ContractResult
                    if file_result.error_msg is None:
                        # 计算买入和卖出信号的比率
                        buy_valid_ratio = file_result.buy_valid_signals / file_result.buy_imbalance_count if file_result.buy_imbalance_count > 0 else 0
                        buy_reverse_ratio = file_result.buy_reverse_signals / file_result.buy_imbalance_count if file_result.buy_imbalance_count > 0 else 0
                        buy_invalid_ratio = file_result.buy_invalid_signals / file_result.buy_imbalance_count if file_result.buy_imbalance_count > 0 else 0
                        sell_valid_ratio = file_result.sell_valid_signals / file_result.sell_imbalance_count if file_result.sell_imbalance_count > 0 else 0
                        sell_reverse_ratio = file_result.sell_reverse_signals / file_result.sell_imbalance_count if file_result.sell_imbalance_count > 0 else 0
                        sell_invalid_ratio = file_result.sell_invalid_signals / file_result.sell_imbalance_count if file_result.sell_imbalance_count > 0 else 0
                        
                        contract_result = ContractResult(
                            contract_name=contract_name,
                            threshold=threshold,
                            total_signals=file_result.total_signals,
                            valid_signals=file_result.valid_signals,
                            reverse_signals=file_result.reverse_signals,
                            invalid_signals=file_result.invalid_signals,
                            valid_ratio=file_result.valid_signals / file_result.total_signals if file_result.total_signals > 0 else 0,
                            reverse_ratio=file_result.reverse_signals / file_result.total_signals if file_result.total_signals > 0 else 0,
                            invalid_ratio=file_result.invalid_signals / file_result.total_signals if file_result.total_signals > 0 else 0,
                            # 买入信号统计
                            buy_signals=file_result.buy_imbalance_count,
                            buy_valid_signals=file_result.buy_valid_signals,
                            buy_reverse_signals=file_result.buy_reverse_signals,
                            buy_invalid_signals=file_result.buy_invalid_signals,
                            buy_valid_ratio=buy_valid_ratio,
                            buy_reverse_ratio=buy_reverse_ratio,
                            buy_invalid_ratio=buy_invalid_ratio,
                            # 卖出信号统计
                            sell_signals=file_result.sell_imbalance_count,
                            sell_valid_signals=file_result.sell_valid_signals,
                            sell_reverse_signals=file_result.sell_reverse_signals,
                            sell_invalid_signals=file_result.sell_invalid_signals,
                            sell_valid_ratio=sell_valid_ratio,
                            sell_reverse_ratio=sell_reverse_ratio,
                            sell_invalid_ratio=sell_invalid_ratio,
                            processing_time=file_result.processing_time
                        )
                    else:
                        # 处理错误情况
                        contract_result = ContractResult(
                            contract_name=contract_name,
                            threshold=threshold,
                            total_signals=0,
                            valid_signals=0,
                            reverse_signals=0,
                            invalid_signals=0,
                            valid_ratio=0,
                            reverse_ratio=0,
                            invalid_ratio=0,
                            # 买入信号统计
                            buy_signals=0,
                            buy_valid_signals=0,
                            buy_reverse_signals=0,
                            buy_invalid_signals=0,
                            buy_valid_ratio=0,
                            buy_reverse_ratio=0,
                            buy_invalid_ratio=0,
                            # 卖出信号统计
                            sell_signals=0,
                            sell_valid_signals=0,
                            sell_reverse_signals=0,
                            sell_invalid_signals=0,
                            sell_valid_ratio=0,
                            sell_reverse_ratio=0,
                            sell_invalid_ratio=0,
                            processing_time=file_result.processing_time
                        )
                    
                    all_contract_results.append(contract_result)
                    contract_threshold_results.append(contract_result)
                
                # 计算该合约的汇总结果
                if contract_threshold_results:
                    contract_name = contract_threshold_results[0].contract_name
                    avg_valid_ratio = np.mean([r.valid_ratio for r in contract_threshold_results])
                    avg_reverse_ratio = np.mean([r.reverse_ratio for r in contract_threshold_results])
                    avg_invalid_ratio = np.mean([r.invalid_ratio for r in contract_threshold_results])
                    total_signals_all = sum(r.total_signals for r in contract_threshold_results)
                    
                    contract_summary = ContractSummary(
                        contract_name=contract_name,
                        avg_valid_ratio=avg_valid_ratio,
                        avg_reverse_ratio=avg_reverse_ratio,
                        avg_invalid_ratio=avg_invalid_ratio,
                        total_signals_all_thresholds=total_signals_all,
                        threshold_count=len(thresholds)
                    )
                    contract_summaries.append(contract_summary)
            
            total_time = time.time() - start_time_total
            print(f"✅ 合约级并行处理完成，总耗时: {total_time:.1f}秒")
            
            # 显示每个合约的处理结果
            for summary in contract_summaries:
                print(f"📈 {summary.contract_name}: 有效率 {summary.avg_valid_ratio:.1%} | 总信号 {summary.total_signals_all_thresholds:,}")
        
        else:
            # 🐌 原有策略：逐个合约处理 (用于文件级并行或单进程)
            print(f"📊 逐个合约处理模式")
            if parallel_strategy == "file":
                print(f"🔧 使用文件级并行策略")
            
            # 遍历每个合约
            for contract_idx, (contract_name, contract_files) in enumerate(contract_groups.items(), 1):
                print(f"📈 处理合约 {contract_idx}/{len(contract_groups)}: {contract_name}")
                
                contract_threshold_results = []
                
                # 遍历两个阈值
                for threshold_idx, threshold in enumerate(thresholds, 1):
                    threshold_type = "买入" if threshold > 0 else "卖出"
                    print(f"  🔍 测试{threshold_type}阈值 {threshold_idx}/{len(thresholds)}: {threshold:.1f}", end=" ", flush=True)
                    
                    start_time = time.time()
                    
                    if use_multiprocess and parallel_strategy == "file" and len(contract_files) > 1:
                        # 文件级多进程处理当前合约的所有文件
                        args_list = [(f, threshold, self.window) for f in contract_files]
                        
                        with Pool(processes=self.n_processes) as pool:
                            file_results = pool.map(process_single_file_silent, args_list)
                    else:
                        # 单进程处理
                        analyzer = ImbalanceAnalyzer(threshold, self.window, verbose=False)
                        file_results = []
                        
                        for file_path in contract_files:
                            result = analyzer.analyze_file(file_path)
                            file_results.append(result)
                    
                    # 汇总当前合约在当前阈值下的结果
                    valid_results = [r for r in file_results if r.error_msg is None]
                    
                    total_signals = sum(r.total_signals for r in valid_results)
                    total_valid = sum(r.valid_signals for r in valid_results)
                    total_reverse = sum(r.reverse_signals for r in valid_results)
                    total_invalid = sum(r.invalid_signals for r in valid_results)
                    
                    valid_ratio = total_valid / total_signals if total_signals > 0 else 0
                    reverse_ratio = total_reverse / total_signals if total_signals > 0 else 0
                    invalid_ratio = total_invalid / total_signals if total_signals > 0 else 0
                    
                    processing_time = time.time() - start_time
                    
                    contract_result = ContractResult(
                        contract_name=contract_name,
                        threshold=threshold,
                        total_signals=total_signals,
                        valid_signals=total_valid,
                        reverse_signals=total_reverse,
                        invalid_signals=total_invalid,
                        valid_ratio=valid_ratio,
                        reverse_ratio=reverse_ratio,
                        invalid_ratio=invalid_ratio,
                        processing_time=processing_time
                    )
                    
                    all_contract_results.append(contract_result)
                    contract_threshold_results.append(contract_result)
                    
                    print(f"✅ {processing_time:.1f}s | 信号:{total_signals:,} | 有效:{valid_ratio:.1%}")
                
                # 计算当前合约的平均表现（只有两个阈值）
                if contract_threshold_results:
                    avg_valid_ratio = np.mean([r.valid_ratio for r in contract_threshold_results])
                    avg_reverse_ratio = np.mean([r.reverse_ratio for r in contract_threshold_results])
                    avg_invalid_ratio = np.mean([r.invalid_ratio for r in contract_threshold_results])
                    total_signals_all = sum(r.total_signals for r in contract_threshold_results)
                    
                    contract_summary = ContractSummary(
                        contract_name=contract_name,
                        avg_valid_ratio=avg_valid_ratio,
                        avg_reverse_ratio=avg_reverse_ratio,
                        avg_invalid_ratio=avg_invalid_ratio,
                        total_signals_all_thresholds=total_signals_all,
                        threshold_count=len(thresholds)
                    )
                    
                    contract_summaries.append(contract_summary)
                
                print(f"  ✅ 合约 {contract_name} 完成\n")
        
        return all_contract_results, contract_summaries
    
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
    
    def export_contract_results_to_csv(self, 
                                      contract_results: List[ContractResult], 
                                      contract_summaries: List[ContractSummary],
                                      detail_filename: str = "contract_detail_analysis.csv",
                                      summary_filename: str = "contract_summary_analysis.csv"):
        """
        导出按合约分析的结果到CSV文件
        
        Args:
            contract_results: 详细结果列表
            contract_summaries: 汇总结果列表
            detail_filename: 详细结果文件名
            summary_filename: 汇总结果文件名
        """
        if not contract_results:
            print("❌ 没有详细结果可导出")
            return
        
        # 导出详细结果
        detail_data = []
        for r in contract_results:
            detail_data.append({
                'contract_name': r.contract_name,
                'threshold': r.threshold,
                'threshold_type': '买入' if r.threshold > 0 else '卖出',
                'total_signals': r.total_signals,
                'valid_signals': r.valid_signals,
                'reverse_signals': r.reverse_signals,
                'invalid_signals': r.invalid_signals,
                'valid_ratio': r.valid_ratio,
                'reverse_ratio': r.reverse_ratio,
                'invalid_ratio': r.invalid_ratio,
                'processing_time': r.processing_time
            })
        
        df_detail = pd.DataFrame(detail_data)
        df_detail.to_csv(detail_filename, index=False, encoding='utf-8-sig')
        print(f"📄 详细结果已导出到: {detail_filename}")
        
        # 导出汇总结果
        if contract_summaries:
            summary_data = []
            for s in contract_summaries:
                summary_data.append({
                    'contract_name': s.contract_name,
                    'avg_valid_ratio': s.avg_valid_ratio,
                    'avg_reverse_ratio': s.avg_reverse_ratio,
                    'avg_invalid_ratio': s.avg_invalid_ratio,
                    'total_signals_all_thresholds': s.total_signals_all_thresholds,
                    'threshold_count': s.threshold_count
                })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(summary_filename, index=False, encoding='utf-8-sig')
            print(f"📄 汇总结果已导出到: {summary_filename}")
    
    def print_contract_summary_and_ranking(self, contract_summaries: List[ContractSummary]):
        """打印合约汇总结果和排名"""
        if not contract_summaries:
            print("❌ 没有汇总结果可显示")
            return
        
        print(f"\n{'='*100}")
        print("📊 按合约分析汇总结果")
        print(f"{'='*100}")
        
        # 表头
        print(f"{'合约名称':<20} {'平均有效率':<12} {'平均反向率':<12} {'平均无效率':<12} {'总信号数':<10}")
        print("-" * 100)
        
        # 数据行
        for s in contract_summaries:
            print(f"{s.contract_name:<20} {s.avg_valid_ratio:<12.1%} {s.avg_reverse_ratio:<12.1%} "
                  f"{s.avg_invalid_ratio:<12.1%} {s.total_signals_all_thresholds:<10,}")
        
        # 按有效率排名
        print(f"\n{'='*60}")
        print("🏆 有效率排名（降序）")
        print(f"{'='*60}")
        valid_ranked = sorted(contract_summaries, key=lambda x: x.avg_valid_ratio, reverse=True)
        
        print(f"{'排名':<6} {'合约名称':<20} {'平均有效率':<12} {'总信号数':<10}")
        print("-" * 60)
        
        for i, s in enumerate(valid_ranked, 1):
            medal = ""
            if i == 1:
                medal = "🥇"
            elif i == 2:
                medal = "🥈"
            elif i == 3:
                medal = "🥉"
            
            print(f"{i:<6} {s.contract_name:<20} {s.avg_valid_ratio:<12.1%} {s.total_signals_all_thresholds:<10,} {medal}")
        
        # 按无效率排名（升序，越低越好）
        print(f"\n{'='*60}")
        print("📉 无效率排名（升序，越低越好）")
        print(f"{'='*60}")
        invalid_ranked = sorted(contract_summaries, key=lambda x: x.avg_invalid_ratio)
        
        print(f"{'排名':<6} {'合约名称':<20} {'平均无效率':<12} {'总信号数':<10}")
        print("-" * 60)
        
        for i, s in enumerate(invalid_ranked, 1):
            medal = ""
            if i == 1:
                medal = "🥇"
            elif i == 2:
                medal = "🥈"
            elif i == 3:
                medal = "🥉"
            
            print(f"{i:<6} {s.contract_name:<20} {s.avg_invalid_ratio:<12.1%} {s.total_signals_all_thresholds:<10,} {medal}")
        
        # 统计信息
        print(f"\n📈 统计信息:")
        avg_valid = np.mean([s.avg_valid_ratio for s in contract_summaries])
        avg_reverse = np.mean([s.avg_reverse_ratio for s in contract_summaries])
        avg_invalid = np.mean([s.avg_invalid_ratio for s in contract_summaries])
        total_signals = sum(s.total_signals_all_thresholds for s in contract_summaries)
        
        print(f"  - 所有合约平均有效率: {avg_valid:.1%}")
        print(f"  - 所有合约平均反向率: {avg_reverse:.1%}")
        print(f"  - 所有合约平均无效率: {avg_invalid:.1%}")
        print(f"  - 总信号数: {total_signals:,}")
        print(f"  - 合约数量: {len(contract_summaries)}")
        
        # 最佳和最差合约
        best_valid = max(contract_summaries, key=lambda x: x.avg_valid_ratio)
        worst_valid = min(contract_summaries, key=lambda x: x.avg_valid_ratio)
        best_invalid = min(contract_summaries, key=lambda x: x.avg_invalid_ratio)
        worst_invalid = max(contract_summaries, key=lambda x: x.avg_invalid_ratio)
        
        print(f"\n🎯 关键指标:")
        print(f"  - 最高有效率: {best_valid.contract_name} ({best_valid.avg_valid_ratio:.1%})")
        print(f"  - 最低有效率: {worst_valid.contract_name} ({worst_valid.avg_valid_ratio:.1%})")
        print(f"  - 最低无效率: {best_invalid.contract_name} ({best_invalid.avg_invalid_ratio:.1%})")
        print(f"  - 最高无效率: {worst_invalid.contract_name} ({worst_invalid.avg_invalid_ratio:.1%})")
    
    def generate_buy_sell_summaries(self, contract_results: List[ContractResult]) -> Tuple[List[BuySignalSummary], List[SellSignalSummary]]:
        """
        根据ContractResult生成买入和卖出信号的汇总
        
        Args:
            contract_results: 合约结果列表
            
        Returns:
            (买入信号汇总列表, 卖出信号汇总列表)
        """
        # 按合约分组
        contract_groups = defaultdict(list)
        for result in contract_results:
            contract_groups[result.contract_name].append(result)
        
        buy_summaries = []
        sell_summaries = []
        
        for contract_name, results in contract_groups.items():
            # 计算买入信号的平均指标
            buy_valid_ratios = [r.buy_valid_ratio for r in results if r.buy_signals > 0]
            buy_reverse_ratios = [r.buy_reverse_ratio for r in results if r.buy_signals > 0]
            buy_invalid_ratios = [r.buy_invalid_ratio for r in results if r.buy_signals > 0]
            total_buy_signals = sum(r.buy_signals for r in results)
            
            if buy_valid_ratios:  # 只有当有买入信号时才创建汇总
                buy_summary = BuySignalSummary(
                    contract_name=contract_name,
                    buy_valid_ratio=np.mean(buy_valid_ratios),
                    buy_reverse_ratio=np.mean(buy_reverse_ratios),
                    buy_invalid_ratio=np.mean(buy_invalid_ratios),
                    total_buy_signals=total_buy_signals
                )
                buy_summaries.append(buy_summary)
            
            # 计算卖出信号的平均指标
            sell_valid_ratios = [r.sell_valid_ratio for r in results if r.sell_signals > 0]
            sell_reverse_ratios = [r.sell_reverse_ratio for r in results if r.sell_signals > 0]
            sell_invalid_ratios = [r.sell_invalid_ratio for r in results if r.sell_signals > 0]
            total_sell_signals = sum(r.sell_signals for r in results)
            
            if sell_valid_ratios:  # 只有当有卖出信号时才创建汇总
                sell_summary = SellSignalSummary(
                    contract_name=contract_name,
                    sell_valid_ratio=np.mean(sell_valid_ratios),
                    sell_reverse_ratio=np.mean(sell_reverse_ratios),
                    sell_invalid_ratio=np.mean(sell_invalid_ratios),
                    total_sell_signals=total_sell_signals
                )
                sell_summaries.append(sell_summary)
        
        return buy_summaries, sell_summaries

    def print_buy_signal_ranking(self, buy_summaries: List[BuySignalSummary]):
        """打印买入信号排名"""
        if not buy_summaries:
            print("❌ 没有买入信号汇总结果可显示")
            return
        
        print(f"\n{'='*100}")
        print("📈 买入信号有效性排名")
        print(f"{'='*100}")
        
        # 按买入信号有效率降序排序
        buy_ranked = sorted(buy_summaries, key=lambda x: (-x.buy_valid_ratio, x.buy_invalid_ratio))
        
        print(f"{'排名':<6} {'合约名称':<20} {'买入有效率':<12} {'买入无效率':<12} {'买入反向率':<12} {'买入信号数':<10} {'推荐':<8}")
        print("-" * 100)
        
        for i, s in enumerate(buy_ranked, 1):
            medal = ""
            recommendation = ""
            
            # 推荐逻辑
            if s.buy_valid_ratio >= 0.4 and s.buy_invalid_ratio <= 0.3:
                recommendation = "🔥优秀"
                if i <= 3:
                    medal = ["🥇", "🥈", "🥉"][i-1]
            elif s.buy_valid_ratio >= 0.35 and s.buy_invalid_ratio <= 0.4:
                recommendation = "✅良好"
            elif s.buy_valid_ratio >= 0.3 and s.buy_invalid_ratio <= 0.5:
                recommendation = "⚠️一般"
            else:
                recommendation = "❌剔除"
            
            print(f"{i:<6} {s.contract_name:<20} {s.buy_valid_ratio:<12.1%} {s.buy_invalid_ratio:<12.1%} "
                  f"{s.buy_reverse_ratio:<12.1%} {s.total_buy_signals:<10,} {recommendation:<8} {medal}")
        
        # 筛选建议
        print(f"\n💡 买入信号筛选建议:")
        excellent = [s for s in buy_summaries if s.buy_valid_ratio >= 0.4 and s.buy_invalid_ratio <= 0.3]
        good = [s for s in buy_summaries if s.buy_valid_ratio >= 0.35 and s.buy_invalid_ratio <= 0.4]
        poor = [s for s in buy_summaries if s.buy_valid_ratio < 0.3 or s.buy_invalid_ratio > 0.5]
        
        print(f"  🔥 优秀合约 ({len(excellent)}个): 买入有效率≥40% 且 买入无效率≤30%")
        if excellent:
            excellent_names = [s.contract_name for s in excellent]
            print(f"     {', '.join(excellent_names)}")
        
        print(f"  ✅ 良好合约 ({len(good)}个): 买入有效率≥35% 且 买入无效率≤40%")
        if good:
            good_names = [s.contract_name for s in good]
            print(f"     {', '.join(good_names)}")
        
        print(f"  ❌ 建议剔除 ({len(poor)}个): 买入有效率<30% 或 买入无效率>50%")
        if poor:
            poor_names = [s.contract_name for s in poor]
            print(f"     {', '.join(poor_names)}")

    def print_sell_signal_ranking(self, sell_summaries: List[SellSignalSummary]):
        """打印卖出信号排名"""
        if not sell_summaries:
            print("❌ 没有卖出信号汇总结果可显示")
            return
        
        print(f"\n{'='*100}")
        print("📉 卖出信号有效性排名")
        print(f"{'='*100}")
        
        # 按卖出信号有效率降序排序
        sell_ranked = sorted(sell_summaries, key=lambda x: (-x.sell_valid_ratio, x.sell_invalid_ratio))
        
        print(f"{'排名':<6} {'合约名称':<20} {'卖出有效率':<12} {'卖出无效率':<12} {'卖出反向率':<12} {'卖出信号数':<10} {'推荐':<8}")
        print("-" * 100)
        
        for i, s in enumerate(sell_ranked, 1):
            medal = ""
            recommendation = ""
            
            # 推荐逻辑
            if s.sell_valid_ratio >= 0.4 and s.sell_invalid_ratio <= 0.3:
                recommendation = "🔥优秀"
                if i <= 3:
                    medal = ["🥇", "🥈", "🥉"][i-1]
            elif s.sell_valid_ratio >= 0.35 and s.sell_invalid_ratio <= 0.4:
                recommendation = "✅良好"
            elif s.sell_valid_ratio >= 0.3 and s.sell_invalid_ratio <= 0.5:
                recommendation = "⚠️一般"
            else:
                recommendation = "❌剔除"
            
            print(f"{i:<6} {s.contract_name:<20} {s.sell_valid_ratio:<12.1%} {s.sell_invalid_ratio:<12.1%} "
                  f"{s.sell_reverse_ratio:<12.1%} {s.total_sell_signals:<10,} {recommendation:<8} {medal}")
        
        # 筛选建议
        print(f"\n💡 卖出信号筛选建议:")
        excellent = [s for s in sell_summaries if s.sell_valid_ratio >= 0.4 and s.sell_invalid_ratio <= 0.3]
        good = [s for s in sell_summaries if s.sell_valid_ratio >= 0.35 and s.sell_invalid_ratio <= 0.4]
        poor = [s for s in sell_summaries if s.sell_valid_ratio < 0.3 or s.sell_invalid_ratio > 0.5]
        
        print(f"  🔥 优秀合约 ({len(excellent)}个): 卖出有效率≥40% 且 卖出无效率≤30%")
        if excellent:
            excellent_names = [s.contract_name for s in excellent]
            print(f"     {', '.join(excellent_names)}")
        
        print(f"  ✅ 良好合约 ({len(good)}个): 卖出有效率≥35% 且 卖出无效率≤40%")
        if good:
            good_names = [s.contract_name for s in good]
            print(f"     {', '.join(good_names)}")
        
        print(f"  ❌ 建议剔除 ({len(poor)}个): 卖出有效率<30% 或 卖出无效率>50%")
        if poor:
            poor_names = [s.contract_name for s in poor]
            print(f"     {', '.join(poor_names)}")

    def print_simple_contract_ranking(self, contract_summaries: List[ContractSummary]):
        """打印简化版合约排名（用于筛选有效合约）"""
        if not contract_summaries:
            print("❌ 没有汇总结果可显示")
            return
        
        print(f"\n{'='*100}")
        print("📊 合约信号有效性排名（用于筛选）")
        print(f"{'='*100}")
        
        # 综合排名：先按有效率降序，再按无效率升序
        print(f"\n🎯 综合排名（有效率↓ + 无效率↑）")
        print(f"{'='*80}")
        
        # 排序逻辑：先按有效率降序，然后按无效率升序
        combined_ranked = sorted(contract_summaries, 
                               key=lambda x: (-x.avg_valid_ratio, x.avg_invalid_ratio))
        
        print(f"{'排名':<6} {'合约名称':<20} {'有效率':<10} {'无效率':<10} {'反向率':<10} {'信号数':<8} {'推荐':<8}")
        print("-" * 80)
        
        for i, s in enumerate(combined_ranked, 1):
            medal = ""
            recommendation = ""
            
            # 推荐逻辑
            if s.avg_valid_ratio >= 0.4 and s.avg_invalid_ratio <= 0.3:
                recommendation = "🔥优秀"
                if i <= 3:
                    medal = ["🥇", "🥈", "🥉"][i-1]
            elif s.avg_valid_ratio >= 0.35 and s.avg_invalid_ratio <= 0.4:
                recommendation = "✅良好"
            elif s.avg_valid_ratio >= 0.3 and s.avg_invalid_ratio <= 0.5:
                recommendation = "⚠️一般"
            else:
                recommendation = "❌剔除"
            
            print(f"{i:<6} {s.contract_name:<20} {s.avg_valid_ratio:<10.1%} {s.avg_invalid_ratio:<10.1%} "
                  f"{s.avg_reverse_ratio:<10.1%} {s.total_signals_all_thresholds:<8,} {recommendation:<8} {medal}")
        
        # 筛选建议
        print(f"\n💡 筛选建议:")
        excellent = [s for s in contract_summaries if s.avg_valid_ratio >= 0.4 and s.avg_invalid_ratio <= 0.3]
        good = [s for s in contract_summaries if s.avg_valid_ratio >= 0.35 and s.avg_invalid_ratio <= 0.4]
        average = [s for s in contract_summaries if s.avg_valid_ratio >= 0.3 and s.avg_invalid_ratio <= 0.5]
        poor = [s for s in contract_summaries if s.avg_valid_ratio < 0.3 or s.avg_invalid_ratio > 0.5]
        
        print(f"  🔥 优秀合约 ({len(excellent)}个): 有效率≥40% 且 无效率≤30%")
        if excellent:
            excellent_names = [s.contract_name for s in excellent]
            print(f"     {', '.join(excellent_names)}")
        
        print(f"  ✅ 良好合约 ({len(good)}个): 有效率≥35% 且 无效率≤40%")
        if good:
            good_names = [s.contract_name for s in good]
            print(f"     {', '.join(good_names)}")
        
        print(f"  ⚠️  一般合约 ({len(average)}个): 有效率≥30% 且 无效率≤50%")
        
        print(f"  ❌ 建议剔除 ({len(poor)}个): 有效率<30% 或 无效率>50%")
        if poor:
            poor_names = [s.contract_name for s in poor]
            print(f"     {', '.join(poor_names)}")
        
        # 统计摘要
        print(f"\n📊 统计摘要:")
        avg_valid = np.mean([s.avg_valid_ratio for s in contract_summaries])
        avg_invalid = np.mean([s.avg_invalid_ratio for s in contract_summaries])
        total_contracts = len(contract_summaries)
        keep_contracts = len(excellent) + len(good)
        
        print(f"  - 总合约数: {total_contracts}")
        print(f"  - 推荐保留: {keep_contracts}个 ({keep_contracts/total_contracts:.1%})")
        print(f"  - 建议剔除: {len(poor)}个 ({len(poor)/total_contracts:.1%})")
        print(f"  - 平均有效率: {avg_valid:.1%}")
        print(f"  - 平均无效率: {avg_invalid:.1%}")
    
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
    print("🔍 合约信号有效性筛选系统")
    print("="*50)
    
    # 创建评估器
    evaluator = DataEvaluator(
        data_dir="test_data",
        future_window_seconds=3,
        n_processes=None  # 自动选择进程数
    )
    
    # 简化版测试：固定阈值0.8和-0.8
    print("🚀 开始合约筛选测试...")
    print("📊 买入信号阈值: 0.8")
    print("📊 卖出信号阈值: -0.8")
    print("🔍 评估指标: 3秒内第一次价格变动有效性")
    print("📋 信号触发条件: 买一量/卖一量 ≥ 5手")
    print("🎯 目标: 筛选出有效的合约，剔除无效合约")
    print("⚡ 并行策略: 合约级多进程加速")
    print()
    
    contract_results, contract_summaries = evaluator.test_contract_simple(
        buy_threshold=0.8,
        sell_threshold=-0.8,
        use_multiprocess=True,
        parallel_strategy="contract"  # 使用合约级并行策略
    )
    
    # 显示简化版排名（用于筛选）
    evaluator.print_simple_contract_ranking(contract_summaries)
    
    # 🆕 生成并显示买入和卖出信号的分别排序
    print(f"\n{'='*100}")
    print("📊 买入和卖出信号分别排序分析")
    print(f"{'='*100}")
    
    buy_summaries, sell_summaries = evaluator.generate_buy_sell_summaries(contract_results)
    
    # 显示买入信号排名
    evaluator.print_buy_signal_ranking(buy_summaries)
    
    # 显示卖出信号排名
    evaluator.print_sell_signal_ranking(sell_summaries)
    
    # 导出CSV文件
    evaluator.export_contract_results_to_csv(
        contract_results, 
        contract_summaries,
        detail_filename="contract_simple_detail.csv",
        summary_filename="contract_simple_summary.csv"
    )
    
    print(f"\n🎉 合约筛选完成!")
    print("💡 输出文件:")
    print("  - contract_simple_detail.csv: 每个合约的详细测试结果")
    print("  - contract_simple_summary.csv: 每个合约的汇总结果")
    print("\n📋 筛选结果:")
    print("  - 🔥优秀合约: 有效率≥40% 且 无效率≤30% -> 强烈推荐")
    print("  - ✅良好合约: 有效率≥35% 且 无效率≤40% -> 可以考虑")
    print("  - ⚠️一般合约: 有效率≥30% 且 无效率≤50% -> 谨慎使用") 
    print("  - ❌建议剔除: 有效率<30% 或 无效率>50% -> 不推荐")
    print("\n🎯 下一步:")
    print("  - 重点关注'优秀'和'良好'级别的合约")
    print("  - 可以对筛选出的合约进行更详细的阈值优化")
    print("  - 建议剔除的合约不适合此策略")


if __name__ == "__main__":
    main()
