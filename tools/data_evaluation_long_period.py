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
    trading_date: str  # 交易日期 (YYYYMMDD 格式)
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
    
    def extract_trading_date_from_filename(self, filename: str) -> str:
        """
        从文件名中提取交易日期
        假设文件格式类似: EXCHANGE.CONTRACT_YYYYMMDD.csv
        """
        try:
            # 移除.csv扩展名
            name_without_ext = filename.replace('.csv', '')
            
            # 查找可能的日期格式 (YYYYMMDD)
            import re
            date_pattern = r'(\d{8})'
            match = re.search(date_pattern, name_without_ext)
            
            if match:
                return match.group(1)
            else:
                # 如果找不到日期，返回unknown
                return "unknown"
        except:
            return "unknown"
    
    def extract_trading_date_from_data(self, file_path: str) -> str:
        """
        从CSV数据中提取交易日期
        读取文件中第一条有效的datetime数据
        """
        try:
            # 读取CSV文件的前几行
            import pandas as pd
            
            # 只读取前10行，节省时间
            df = pd.read_csv(file_path, nrows=10)
            
            if 'datetime' not in df.columns:
                return "unknown"
            
            # 查找第一个非空的datetime值
            for _, row in df.iterrows():
                datetime_str = str(row['datetime'])
                if datetime_str and datetime_str != 'nan' and datetime_str != 'NaN':
                    # 提取日期部分 (YYYY-MM-DD)
                    if ' ' in datetime_str:
                        date_part = datetime_str.split(' ')[0]
                    else:
                        date_part = datetime_str
                    
                    # 转换为YYYYMMDD格式
                    if '-' in date_part and len(date_part) >= 10:
                        date_components = date_part.split('-')
                        if len(date_components) >= 3:
                            year = date_components[0]
                            month = date_components[1].zfill(2)
                            day = date_components[2].zfill(2)
                            return f"{year}{month}{day}"
            
            # 如果没有找到有效日期，返回unknown
            return "unknown"
            
        except Exception as e:
            if self.verbose:
                print(f"  警告: 无法从数据中提取交易日期: {str(e)}")
            return "unknown"
    
    def analyze_file_by_trading_days(self, file_path: str) -> Dict[str, FileResult]:
        """
        分析单个文件，按交易日分组分析
        适用于单个文件包含多个交易日数据的情况
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            按交易日分组的分析结果字典
        """
        import pandas as pd
        
        filename = os.path.basename(file_path)
        results = {}
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            if 'datetime' not in df.columns:
                return results
            
            # 转换datetime列
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['trading_date'] = df['datetime'].dt.date
            
            # 按交易日分组
            trading_dates = df['trading_date'].unique()
            
            if self.verbose:
                print(f"📁 分析文件: {filename}")
                print(f"📅 发现 {len(trading_dates)} 个交易日")
            
            for trading_date in sorted(trading_dates):
                # 过滤当日数据
                daily_df = df[df['trading_date'] == trading_date].copy()
                
                if len(daily_df) < 10:  # 数据太少，跳过
                    continue
                
                trading_date_str = trading_date.strftime('%Y%m%d')
                
                if self.verbose:
                    print(f"  🔍 分析 {trading_date_str} ({len(daily_df):,} 条记录)")
                
                start_time = time.time()
                
                # 转换为tick数据格式进行分析
                ticks = self._convert_df_to_ticks(daily_df, filename)
                
                if len(ticks) < 10:
                    continue
                
                # 分析失衡信号
                signals = []
                for i, tick in enumerate(ticks):
                    imbalance_value, is_imbalanced, direction = self.calculate_imbalance(tick)
                    
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
                        
                        # 评估信号有效性
                        signal_type = self.evaluate_signal(tick.last, future_prices, direction)
                        
                        signal = ImbalanceSignal(
                            timestamp=tick.ts,
                            imbalance_value=imbalance_value,
                            imbalance_direction=direction,
                            current_price=tick.last,
                            future_prices=future_prices,
                            signal_type=signal_type
                        )
                        signals.append(signal)
                
                # 统计结果
                total_signals = len(signals)
                valid_signals = len([s for s in signals if s.signal_type == 'valid'])
                reverse_signals = len([s for s in signals if s.signal_type == 'reverse'])
                invalid_signals = len([s for s in signals if s.signal_type == 'invalid'])
                
                # 买卖分别统计
                buy_signals = [s for s in signals if s.imbalance_direction == 'buy_imbalance']
                sell_signals = [s for s in signals if s.imbalance_direction == 'sell_imbalance']
                
                buy_valid = len([s for s in buy_signals if s.signal_type == 'valid'])
                buy_reverse = len([s for s in buy_signals if s.signal_type == 'reverse'])
                buy_invalid = len([s for s in buy_signals if s.signal_type == 'invalid'])
                
                sell_valid = len([s for s in sell_signals if s.signal_type == 'valid'])
                sell_reverse = len([s for s in sell_signals if s.signal_type == 'reverse'])
                sell_invalid = len([s for s in sell_signals if s.signal_type == 'invalid'])
                
                processing_time = time.time() - start_time
                
                # 创建结果
                result = FileResult(
                    filename=f"{filename}_{trading_date_str}",
                    trading_date=trading_date_str,
                    total_signals=total_signals,
                    valid_signals=valid_signals,
                    reverse_signals=reverse_signals,
                    invalid_signals=invalid_signals,
                    buy_imbalance_count=len(buy_signals),
                    sell_imbalance_count=len(sell_signals),
                    buy_valid_signals=buy_valid,
                    buy_reverse_signals=buy_reverse,
                    buy_invalid_signals=buy_invalid,
                    sell_valid_signals=sell_valid,
                    sell_reverse_signals=sell_reverse,
                    sell_invalid_signals=sell_invalid,
                    processing_time=processing_time
                )
                
                results[trading_date_str] = result
                
                if self.verbose:
                    print(f"    ✅ 完成 ({processing_time:.1f}s)")
                    print(f"       总信号数: {total_signals}")
                    print(f"       有效信号: {valid_signals} ({valid_signals/total_signals*100 if total_signals > 0 else 0:.1f}%)")
                    print(f"       反向信号: {reverse_signals} ({reverse_signals/total_signals*100 if total_signals > 0 else 0:.1f}%)")
                    print(f"       无效信号: {invalid_signals} ({invalid_signals/total_signals*100 if total_signals > 0 else 0:.1f}%)")
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 分析文件失败: {str(e)}")
            return results
    
    def _convert_df_to_ticks(self, df: pd.DataFrame, filename: str) -> List[Tick]:
        """
        将DataFrame转换为Tick对象列表
        """
        ticks = []
        
        # 从文件名提取合约代码
        contract_code = filename.replace('.csv', '')
        
        for _, row in df.iterrows():
            try:
                # 跳过无效数据
                bid_price1 = row.get(f'{contract_code}.bid_price1')
                ask_price1 = row.get(f'{contract_code}.ask_price1')
                bid_volume1 = row.get(f'{contract_code}.bid_volume1', 0)
                ask_volume1 = row.get(f'{contract_code}.ask_volume1', 0)
                
                if (pd.isna(bid_price1) or pd.isna(ask_price1) or bid_volume1 == 0 or ask_volume1 == 0):
                    continue
                
                # 按照Tick类的正确格式创建对象
                tick = Tick(
                    ts=row['datetime'],  # 使用ts而不是timestamp
                    bids=[bid_price1],   # 买价列表
                    asks=[ask_price1],   # 卖价列表
                    bid_volumes=[bid_volume1],  # 买量列表
                    ask_volumes=[ask_volume1],  # 卖量列表
                    last=row.get(f'{contract_code}.last_price', 0),  # 最新价
                    vol=row.get(f'{contract_code}.volume', 0),       # 成交量
                    amt=row.get(f'{contract_code}.amount', 0)        # 成交额
                )
                ticks.append(tick)
                
            except Exception as e:
                if self.verbose:
                    print(f"跳过 {row['datetime']}：{str(e)}")
                continue
        
        return ticks

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
        
        # 首先尝试从数据中提取交易日期
        trading_date = self.extract_trading_date_from_data(file_path)
        
        # 如果从数据中提取失败，再尝试从文件名提取
        if trading_date == "unknown":
            trading_date = self.extract_trading_date_from_filename(filename)
        
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
                    trading_date=trading_date,
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
                trading_date=trading_date,
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
                trading_date=trading_date,
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








@dataclass
class DailyResult:
    """单日分析结果"""
    contract_name: str
    trading_date: str
    total_signals: int
    valid_signals: int
    reverse_signals: int
    invalid_signals: int
    valid_ratio: float
    reverse_ratio: float
    invalid_ratio: float


@dataclass
class ContractStabilityResult:
    """合约稳定性分析结果"""
    contract_name: str
    total_trading_days: int
    avg_valid_ratio: float
    std_valid_ratio: float  # 有效率标准差
    avg_signals_per_day: float
    min_valid_ratio: float
    max_valid_ratio: float
    consistency_score: float  # 稳定性评分 (0-1)
    daily_results: List[DailyResult]


class DataEvaluator:
    """数据评估主类"""
    
    def __init__(self, data_dir: str = r"C:\Users\justr\Desktop\tqsdk_data_2025731_2025807", 
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
        """从文件路径中提取合约名称（不包含日期）"""
        filename = os.path.basename(file_path)
        # 移除.csv扩展名
        name_without_ext = filename.replace('.csv', '')
        
        # 移除日期部分，假设日期格式为YYYYMMDD
        import re
        # 查找并移除8位数字（日期）
        contract_name = re.sub(r'_?\d{8}', '', name_without_ext)
        # 移除可能的尾随下划线
        contract_name = contract_name.rstrip('_')
        
        return contract_name
    
    def group_files_by_contract(self, files: List[str]) -> Dict[str, List[str]]:
        """按合约名称分组文件"""
        contract_groups = defaultdict(list)
        for file_path in files:
            contract_name = self.extract_contract_name(file_path)
            contract_groups[contract_name].append(file_path)
        return dict(contract_groups)
    

    
    def analyze_contract_stability(self, 
                                  threshold: float = 0.8,
                                  use_multiprocess: bool = True) -> List[ContractStabilityResult]:
        """
        分析合约在不同交易日的信号稳定性
        
        Args:
            threshold: 信号阈值
            use_multiprocess: 是否使用多进程
            
        Returns:
            每个合约的稳定性分析结果
        """
        files = self.get_data_files()
        if not files:
            print("❌ 未找到数据文件")
            return []
        
        # 按合约分组文件
        contract_groups = self.group_files_by_contract(files)
        
        print(f"🔍 合约稳定性分析")
        print(f"📁 合约数量: {len(contract_groups)}个")
        print(f"📊 测试阈值: {threshold}")
        print(f"⚙️  {'多进程' if use_multiprocess else '单进程'}模式")
        if use_multiprocess:
            print(f"🚀 使用 {self.n_processes} 个进程")
        print()
        
        stability_results = []
        
        for contract_idx, (contract_name, contract_files) in enumerate(contract_groups.items(), 1):
            print(f"📈 分析合约 {contract_idx}/{len(contract_groups)}: {contract_name}")
            
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
            
            # 处理结果，计算每日的统计数据
            valid_results = [r for r in file_results if r.error_msg is None and r.total_signals > 0]
            
            if not valid_results:
                print(f"  ⚠️  合约 {contract_name} 没有有效数据")
                continue
            
            # 按交易日生成每日结果
            daily_results = []
            for result in valid_results:
                valid_ratio = result.valid_signals / result.total_signals if result.total_signals > 0 else 0
                reverse_ratio = result.reverse_signals / result.total_signals if result.total_signals > 0 else 0
                invalid_ratio = result.invalid_signals / result.total_signals if result.total_signals > 0 else 0
                
                daily_result = DailyResult(
                    contract_name=contract_name,
                    trading_date=result.trading_date,
                    total_signals=result.total_signals,
                    valid_signals=result.valid_signals,
                    reverse_signals=result.reverse_signals,
                    invalid_signals=result.invalid_signals,
                    valid_ratio=valid_ratio,
                    reverse_ratio=reverse_ratio,
                    invalid_ratio=invalid_ratio
                )
                daily_results.append(daily_result)
            
            # 计算稳定性指标
            valid_ratios = [dr.valid_ratio for dr in daily_results]
            
            avg_valid_ratio = np.mean(valid_ratios)
            std_valid_ratio = np.std(valid_ratios)
            min_valid_ratio = np.min(valid_ratios)
            max_valid_ratio = np.max(valid_ratios)
            avg_signals_per_day = np.mean([dr.total_signals for dr in daily_results])
            
            # 计算稳定性评分 (0-1)：标准差越小，评分越高
            # 稳定性评分 = 1 - (标准差 / 最大可能标准差)
            max_possible_std = 0.5  # 理论最大标准差
            consistency_score = max(0, 1 - (std_valid_ratio / max_possible_std))
            
            stability_result = ContractStabilityResult(
                contract_name=contract_name,
                total_trading_days=len(daily_results),
                avg_valid_ratio=avg_valid_ratio,
                std_valid_ratio=std_valid_ratio,
                avg_signals_per_day=avg_signals_per_day,
                min_valid_ratio=min_valid_ratio,
                max_valid_ratio=max_valid_ratio,
                consistency_score=consistency_score,
                daily_results=daily_results
            )
            
            stability_results.append(stability_result)
            
            processing_time = time.time() - start_time
            print(f"  ✅ {processing_time:.1f}s | 交易日:{len(daily_results)} | 平均有效率:{avg_valid_ratio:.1%} | 稳定性:{consistency_score:.3f}")
        
        return stability_results
    

    

    

    
    def print_daily_details(self, stability_results: List[ContractStabilityResult], show_all: bool = False):
        """打印每个合约的每日详细分析结果"""
        if not stability_results:
            print("❌ 没有稳定性分析结果可显示")
            return
        
        print(f"\n{'='*120}")
        print("📊 合约每日信号详细分析")
        print(f"{'='*120}")
        
        for contract_result in stability_results:
            print(f"\n🔍 合约: {contract_result.contract_name}")
            print(f"📊 交易日数: {contract_result.total_trading_days} | 平均有效率: {contract_result.avg_valid_ratio:.1%} | 稳定性评分: {contract_result.consistency_score:.3f}")
            print("-" * 100)
            print(f"{'交易日期':<12} {'总信号数':<10} {'有效信号':<10} {'反向信号':<10} {'无效信号':<10} {'有效率':<10} {'反向率':<10} {'无效率':<10}")
            print("-" * 100)
            
            # 按日期排序
            daily_sorted = sorted(contract_result.daily_results, key=lambda x: x.trading_date)
            
            for daily in daily_sorted:
                print(f"{daily.trading_date:<12} {daily.total_signals:<10} {daily.valid_signals:<10} "
                      f"{daily.reverse_signals:<10} {daily.invalid_signals:<10} {daily.valid_ratio:<10.1%} "
                      f"{daily.reverse_ratio:<10.1%} {daily.invalid_ratio:<10.1%}")
            
            # 每个合约的小结
            print("-" * 100)
            print(f"📈 合约汇总: 最高有效率 {contract_result.max_valid_ratio:.1%} | 最低有效率 {contract_result.min_valid_ratio:.1%} | 标准差 {contract_result.std_valid_ratio:.3f}")
            
            if not show_all:
                # 如果不显示全部，只显示第一个合约作为示例
                print(f"\n💡 (显示第一个合约作为示例，如需查看所有合约详情请查看CSV文件)")
                break

    def print_stability_analysis(self, stability_results: List[ContractStabilityResult]):
        """打印合约稳定性分析结果"""
        if not stability_results:
            print("❌ 没有稳定性分析结果可显示")
            return
        
        print(f"\n{'='*120}")
        print("📊 合约信号稳定性分析结果")
        print(f"{'='*120}")
        
        # 表头
        print(f"{'合约名称':<20} {'交易日数':<8} {'平均有效率':<12} {'有效率标准差':<12} {'最小有效率':<12} {'最大有效率':<12} {'稳定性评分':<12} {'日均信号数':<12}")
        print("-" * 120)
        
        # 数据行
        for r in stability_results:
            print(f"{r.contract_name:<20} {r.total_trading_days:<8} {r.avg_valid_ratio:<12.1%} {r.std_valid_ratio:<12.3f} "
                  f"{r.min_valid_ratio:<12.1%} {r.max_valid_ratio:<12.1%} {r.consistency_score:<12.3f} {r.avg_signals_per_day:<12.1f}")
        
        # 按稳定性评分排名
        print(f"\n{'='*80}")
        print("🏆 稳定性排名（按稳定性评分降序）")
        print(f"{'='*80}")
        stability_ranked = sorted(stability_results, key=lambda x: (-x.consistency_score, -x.avg_valid_ratio))
        
        print(f"{'排名':<6} {'合约名称':<20} {'稳定性评分':<12} {'平均有效率':<12} {'标准差':<10} {'推荐':<8}")
        print("-" * 80)
        
        for i, r in enumerate(stability_ranked, 1):
            medal = ""
            recommendation = ""
            
            # 推荐逻辑：稳定性评分 + 平均有效率
            if r.consistency_score >= 0.7 and r.avg_valid_ratio >= 0.4:
                recommendation = "🔥优秀"
                if i <= 3:
                    medal = ["🥇", "🥈", "🥉"][i-1]
            elif r.consistency_score >= 0.6 and r.avg_valid_ratio >= 0.35:
                recommendation = "✅良好"
            elif r.consistency_score >= 0.5 and r.avg_valid_ratio >= 0.3:
                recommendation = "⚠️一般"
            else:
                recommendation = "❌不稳"
            
            print(f"{i:<6} {r.contract_name:<20} {r.consistency_score:<12.3f} {r.avg_valid_ratio:<12.1%} "
                  f"{r.std_valid_ratio:<10.3f} {recommendation:<8} {medal}")
        
        # 筛选建议
        print(f"\n💡 稳定性筛选建议:")
        excellent = [r for r in stability_results if r.consistency_score >= 0.7 and r.avg_valid_ratio >= 0.4]
        good = [r for r in stability_results if r.consistency_score >= 0.6 and r.avg_valid_ratio >= 0.35]
        unstable = [r for r in stability_results if r.consistency_score < 0.5 or r.avg_valid_ratio < 0.3]
        
        print(f"  🔥 优秀稳定 ({len(excellent)}个): 稳定性≥0.7 且 平均有效率≥40%")
        if excellent:
            excellent_names = [r.contract_name for r in excellent]
            print(f"     {', '.join(excellent_names)}")
        
        print(f"  ✅ 良好稳定 ({len(good)}个): 稳定性≥0.6 且 平均有效率≥35%")
        if good:
            good_names = [r.contract_name for r in good]
            print(f"     {', '.join(good_names)}")
        
        print(f"  ❌ 不稳定 ({len(unstable)}个): 稳定性<0.5 或 平均有效率<30%")
        if unstable:
            unstable_names = [r.contract_name for r in unstable]
            print(f"     {', '.join(unstable_names)}")
        
        # 统计摘要
        print(f"\n📊 稳定性统计摘要:")
        avg_stability = np.mean([r.consistency_score for r in stability_results])
        avg_std = np.mean([r.std_valid_ratio for r in stability_results])
        total_contracts = len(stability_results)
        stable_contracts = len(excellent) + len(good)
        
        print(f"  - 总合约数: {total_contracts}")
        print(f"  - 稳定合约: {stable_contracts}个 ({stable_contracts/total_contracts:.1%})")
        print(f"  - 不稳定合约: {len(unstable)}个 ({len(unstable)/total_contracts:.1%})")
        print(f"  - 平均稳定性评分: {avg_stability:.3f}")
        print(f"  - 平均有效率标准差: {avg_std:.3f}")
    
    def export_stability_results_to_csv(self, 
                                       stability_results: List[ContractStabilityResult],
                                       summary_filename: str = "contract_stability_summary.csv",
                                       daily_filename: str = "contract_daily_analysis.csv"):
        """
        导出稳定性分析结果到CSV文件
        
        Args:
            stability_results: 稳定性分析结果列表
            summary_filename: 汇总结果文件名
            daily_filename: 每日详细结果文件名
        """
        if not stability_results:
            print("❌ 没有稳定性分析结果可导出")
            return
        
        # 导出汇总结果
        summary_data = []
        for r in stability_results:
            summary_data.append({
                'contract_name': r.contract_name,
                'total_trading_days': r.total_trading_days,
                'avg_valid_ratio': r.avg_valid_ratio,
                'std_valid_ratio': r.std_valid_ratio,
                'avg_signals_per_day': r.avg_signals_per_day,
                'min_valid_ratio': r.min_valid_ratio,
                'max_valid_ratio': r.max_valid_ratio,
                'consistency_score': r.consistency_score
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values(['consistency_score', 'avg_valid_ratio'], ascending=[False, False])
        df_summary.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        print(f"📄 稳定性汇总结果已导出到: {summary_filename}")
        
        # 导出每日详细结果
        daily_data = []
        for r in stability_results:
            for daily in r.daily_results:
                daily_data.append({
                    'contract_name': daily.contract_name,
                    'trading_date': daily.trading_date,
                    'total_signals': daily.total_signals,
                    'valid_signals': daily.valid_signals,
                    'reverse_signals': daily.reverse_signals,
                    'invalid_signals': daily.invalid_signals,
                    'valid_ratio': daily.valid_ratio,
                    'reverse_ratio': daily.reverse_ratio,
                    'invalid_ratio': daily.invalid_ratio
                })
        
        df_daily = pd.DataFrame(daily_data)
        df_daily = df_daily.sort_values(['contract_name', 'trading_date'])
        df_daily.to_csv(daily_filename, index=False, encoding='utf-8-sig')
        print(f"📄 每日详细结果已导出到: {daily_filename}")


def main():
    """主函数"""
    print("🔍 合约信号有效性筛选与稳定性分析系统")
    print("="*60)
    
    # 创建评估器
    evaluator = DataEvaluator(
        data_dir=r"C:\Users\justr\Desktop\tqsdk_data_2025731_2025807",
        future_window_seconds=3,
        n_processes=None  # 自动选择进程数
    )
    
    print("🚀 开始买方0.8强度信号分析...")
    print("📊 买方信号阈值: 0.8")
    print("🔍 评估指标: 3秒内第一次价格变动有效性")
    print("📋 信号触发条件: 买一量 ≥ 5手")
    print("🎯 目标: 测试每个合约在不同交易日的有效信号、反向信号和无效信号")
    print("📈 分析范围: 7.31 - 8.7 历史数据")
    print("⚡ 并行策略: 多进程加速")
    print()
    
    # 进行稳定性分析
    stability_results = evaluator.analyze_contract_stability(
        threshold=0.8,  # 只测试买方0.8强度
        use_multiprocess=True
    )
    
    # 显示每日详细分析结果（先显示第一个合约作为示例）
    evaluator.print_daily_details(stability_results, show_all=False)
    
    # 显示稳定性分析结果
    evaluator.print_stability_analysis(stability_results)
    
    # 导出稳定性分析结果
    evaluator.export_stability_results_to_csv(
        stability_results,
        summary_filename="contract_stability_summary.csv",
        daily_filename="contract_daily_analysis.csv"
    )
    
    print(f"\n🎉 买方0.8强度信号分析完成!")
    print("💡 输出文件:")
    print("  - contract_stability_summary.csv: 合约稳定性汇总结果")
    print("  - contract_daily_analysis.csv: 每个合约每日详细分析结果")
    print("\n📊 分析内容:")
    print("  - 每个合约在不同交易日的信号表现")
    print("  - 有效信号、反向信号、无效信号统计")
    print("  - 跨交易日的稳定性评分")
    print("\n🎯 稳定性评价标准:")
    print("  - 🔥优秀稳定: 稳定性≥0.7 且 平均有效率≥40%")
    print("  - ✅良好稳定: 稳定性≥0.6 且 平均有效率≥35%")
    print("  - ⚠️一般稳定: 稳定性≥0.5 且 平均有效率≥30%")
    print("  - ❌不稳定: 稳定性<0.5 或 平均有效率<30%")
    print("\n📈 关键指标说明:")
    print("  - 稳定性评分: 基于有效率标准差计算，值越大越稳定")
    print("  - 平均有效率: 所有交易日有效率的平均值")
    print("  - 有效率标准差: 反映不同交易日间有效率的波动程度")
    print("\n🎯 下一步建议:")
    print("  - 重点关注'优秀稳定'和'良好稳定'的合约")
    print("  - 检查每日详细数据，了解信号在时间维度上的表现")
    print("  - 对于不稳定的合约，可以考虑调整阈值或剔除")


if __name__ == "__main__":
    main()
