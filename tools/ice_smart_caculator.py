# ice_smart_caculator.py
# 冰山智能策略概率计算器：分析盘口失衡和micro-price条件满足时的价格变动概率

import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.market_data import tick_reader, Tick

@dataclass
class SignalResult:
    """信号分析结果"""
    ts: pd.Timestamp           # 信号触发时间
    Q: float                   # 盘口失衡指标
    micro_price: float         # 量加权中价
    side: str                  # 交易方向 ('buy' or 'sell')
    current_price: float       # 当前参考价格
    window_prices: list        # 时间窗口内的价格序列
    higher_count: int          # 高于触发价格的tick数
    lower_count: int           # 低于触发价格的tick数
    equal_count: int           # 等于触发价格的tick数
    direction: str             # 变动方向 ('favorable', 'unfavorable', 'no_change')

class IceSmartCalculator:
    """
    冰山智能策略概率计算器
    
    基于盘口失衡指标Q值和micro_price的双重条件分析价格变动概率：
    
    Q值计算：Q = (总买量-总卖量)/(总买量+总卖量)
    - Q > 0：买盘强于卖盘
    - Q < 0：卖盘强于买盘
    
    micro_price计算：量加权的中间价格
    - 更接近买一价：市场偏向下跌趋势
    - 更接近卖一价：市场偏向上涨趋势
    
    信号触发条件（双重条件都满足）：
    - 买入方向：Q > threshold（买盘强）且 micro_price更接近卖一价（上涨趋势）
    - 卖出方向：Q < -threshold（卖盘强）且 micro_price更接近买一价（下跌趋势）
    """
    
    def __init__(self, imbalance_threshold: float = 0.5, prediction_window: float = 3.0):
        """
        初始化计算器
        
        Args:
            imbalance_threshold: 盘口失衡阈值，默认0.5
            prediction_window: 预测窗口（秒），默认3秒
        """
        self.threshold = imbalance_threshold
        self.window = prediction_window
        
    def _calculate_market_metrics(self, tick: Tick) -> Tuple[float, float]:
        """
        计算市场指标（支持5档数据）
        
        Args:
            tick: 市场数据tick
            
        Returns:
            (Q值, micro_price)
        """
        # 确定使用的档位数量（优先使用5档，如果没有则使用1档）
        available_levels = min(len(tick.bids), len(tick.asks), len(tick.bid_volumes), len(tick.ask_volumes))
        use_levels = min(5, available_levels) if available_levels >= 5 else 1
        
        # 计算总买量和总卖量
        total_bid_vol = sum(tick.bid_volumes[:use_levels])
        total_ask_vol = sum(tick.ask_volumes[:use_levels])
        
        # 避免除零错误
        if total_bid_vol + total_ask_vol == 0:
            return 0.0, (tick.bid + tick.ask) / 2
        
        # 1. 计算盘口失衡指标 Q = (B-A)/(B+A)
        Q = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        
        # 2. 计算加权micro-price
        if use_levels == 1:
            # 单档情况：使用原有计算方式
            micro_price = (tick.bid * total_ask_vol + tick.ask * total_bid_vol) / (total_bid_vol + total_ask_vol)
        else:
            # 多档情况：使用量加权的多档价格
            weighted_bid_price = sum(tick.bids[i] * tick.bid_volumes[i] for i in range(use_levels)) / total_bid_vol
            weighted_ask_price = sum(tick.asks[i] * tick.ask_volumes[i] for i in range(use_levels)) / total_ask_vol
            micro_price = (weighted_bid_price * total_ask_vol + weighted_ask_price * total_bid_vol) / (total_bid_vol + total_ask_vol)
        
        return Q, micro_price
    
    def _check_signal_conditions(self, tick: Tick, side: str) -> Tuple[bool, float, float]:
        """
        检查信号条件是否满足
        
        Args:
            tick: 市场数据tick
            side: 交易方向 ('buy' or 'sell')
            
        Returns:
            (是否满足条件, Q值, micro_price)
        """
        Q, micro_price = self._calculate_market_metrics(tick)
        
        # 双重条件判断：Q值条件 AND micro_price方向条件
        if side == "buy":
            # 买入：Q > threshold（买盘强）且 micro_price更接近卖一价（市场上涨趋势）
            q_condition = Q > self.threshold
            micro_condition = abs(micro_price - tick.ask) < abs(micro_price - tick.bid)
            signal_triggered = q_condition and micro_condition
        else:
            # 卖出：Q < -threshold（卖盘强）且 micro_price更接近买一价（市场下跌趋势）
            q_condition = Q < -self.threshold  
            micro_condition = abs(micro_price - tick.bid) < abs(micro_price - tick.ask)
            signal_triggered = q_condition and micro_condition
        
        return signal_triggered, Q, micro_price
    
    def _analyze_price_movement_in_window(self, ticks: List[Tick], current_idx: int, target_time: pd.Timestamp, current_price: float) -> Tuple[List[float], int, int, int]:
        """
        分析未来时间窗口内所有tick的价格行为
        
        Args:
            ticks: 所有tick数据
            current_idx: 当前tick索引
            target_time: 目标时间
            current_price: 触发信号时的价格
            
        Returns:
            (窗口内价格列表, 高于当前价格的tick数, 低于当前价格的tick数, 等于当前价格的tick数)
        """
        window_prices = []
        higher_count = 0  # 高于触发价格的tick数
        lower_count = 0   # 低于触发价格的tick数  
        equal_count = 0   # 等于触发价格的tick数
        
        # 从当前位置往后查找，直到目标时间
        for i in range(current_idx + 1, len(ticks)):
            if ticks[i].ts > target_time:
                break
                
            tick_price = (ticks[i].bid + ticks[i].ask) / 2
            window_prices.append(tick_price)
            
            # 统计价格分布情况（使用小的容忍度来判断相等）
            if tick_price > current_price + 0.005:  # 高于触发价格
                higher_count += 1
            elif tick_price < current_price - 0.005:  # 低于触发价格
                lower_count += 1
            else:  # 基本等于触发价格
                equal_count += 1
        
        return window_prices, higher_count, lower_count, equal_count
    
    def _classify_price_movement(self, higher_count: int, lower_count: int, equal_count: int, side: str) -> str:
        """
        基于时间窗口内价格行为分类变动方向
        
        Args:
            higher_count: 高于触发价格的tick数
            lower_count: 低于触发价格的tick数
            equal_count: 等于触发价格的tick数
            side: 交易方向
            
        Returns:
            'favorable': 有利方向, 'unfavorable': 不利方向, 'no_change': 偏离不明显
        """
        total_ticks = higher_count + lower_count + equal_count
        
        # 如果窗口内没有tick数据，视为无变化
        if total_ticks == 0:
            return 'no_change'
        
        if side == "buy":
            # 买入情况：
            if lower_count > 0 and higher_count == 0:
                # 所有tick都 <= 触发价格：价格持续下跌或持平，对买入有利
                return 'favorable'
            elif higher_count > 0 and lower_count == 0:
                # 所有tick都 >= 触发价格：价格持续上涨或持平，对买入不利
                return 'unfavorable'
            else:
                # 有高有低，或全部相等：价格偏离不明显
                return 'no_change'
        else:
            # 卖出情况：
            if higher_count > 0 and lower_count == 0:
                # 所有tick都 >= 触发价格：价格持续上涨或持平，对卖出有利
                return 'favorable'
            elif lower_count > 0 and higher_count == 0:
                # 所有tick都 <= 触发价格：价格持续下跌或持平，对卖出不利
                return 'unfavorable'
            else:
                # 有高有低，或全部相等：价格偏离不明显
                return 'no_change'
    
    def _analyze_side(self, ticks: List[Tick], side: str) -> Dict:
        """
        分析单个交易方向的数据
        
        Args:
            ticks: 已读取的tick数据列表
            side: 分析的交易方向 ('buy' or 'sell')
            
        Returns:
            分析结果字典
        """
        print(f"开始分析交易方向：{side}")
        
        # 添加Q值分布分析
        q_values = []
        q_condition_count = 0      # Q条件满足次数
        micro_condition_count = 0  # micro_price条件满足次数
        signal_count = 0           # 双重条件都满足次数
        
        if len(ticks) < 2:
            print("数据量不足，无法进行分析")
            return {}
        
        # 存储信号结果
        signal_results = []
        
        # 逐个tick分析
        for i, tick in enumerate(ticks):
            # 检查信号条件
            signal_triggered, Q, micro_price = self._check_signal_conditions(tick, side)
            
            # 收集Q值用于分析
            q_values.append(Q)
            
            # 分别统计各个条件的满足情况
            if side == "buy":
                q_condition = Q > self.threshold
                micro_condition = abs(micro_price - tick.ask) < abs(micro_price - tick.bid)
            else:
                q_condition = Q < -self.threshold  
                micro_condition = abs(micro_price - tick.bid) < abs(micro_price - tick.ask)
            
            if q_condition:
                q_condition_count += 1
            if micro_condition:
                micro_condition_count += 1
            if signal_triggered:
                signal_count += 1
            
            if signal_triggered:
                # 计算目标时间
                target_time = tick.ts + pd.Timedelta(seconds=self.window)
                
                # 分析时间窗口内的价格行为
                current_price = (tick.bid + tick.ask) / 2
                window_prices, higher_count, lower_count, equal_count = self._analyze_price_movement_in_window(
                    ticks, i, target_time, current_price
                )
                
                # 分类价格变动方向
                direction = self._classify_price_movement(higher_count, lower_count, equal_count, side)
                
                # 存储结果
                result = SignalResult(
                    ts=tick.ts,
                    Q=Q,
                    micro_price=micro_price,
                    side=side,
                    current_price=current_price,
                    window_prices=window_prices,
                    higher_count=higher_count,
                    lower_count=lower_count,
                    equal_count=equal_count,
                    direction=direction
                )
                signal_results.append(result)
        
        # 统计结果
        total_signals = len(signal_results)
        favorable_count = sum(1 for r in signal_results if r.direction == 'favorable')
        unfavorable_count = sum(1 for r in signal_results if r.direction == 'unfavorable')
        no_change_count = sum(1 for r in signal_results if r.direction == 'no_change')
        
        # 计算概率
        favorable_prob = (favorable_count / total_signals * 100) if total_signals > 0 else 0
        unfavorable_prob = (unfavorable_count / total_signals * 100) if total_signals > 0 else 0
        no_change_prob = (no_change_count / total_signals * 100) if total_signals > 0 else 0
        
        results = {
            'side': side,
            'total_signals': total_signals,
            'favorable_count': favorable_count,
            'unfavorable_count': unfavorable_count,
            'no_change_count': no_change_count,
            'favorable_probability': favorable_prob,
            'unfavorable_probability': unfavorable_prob,
            'no_change_probability': no_change_prob,
            'signal_details': signal_results,
            'q_values': q_values,
            'q_condition_count': q_condition_count,
            'micro_condition_count': micro_condition_count,
            'signal_count': signal_count
        }
        
        return results
    
    def analyze_csv_both_sides(self, csv_path: str) -> Dict:
        """
        分析CSV文件中的数据（买入和卖出两个方向）
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            包含买入和卖出两个方向分析结果的字典
        """
        print(f"开始分析 {csv_path}")
        print(f"失衡阈值：{self.threshold}, 预测窗口：{self.window}秒")
        
        # 只读取一次数据
        ticks = list(tick_reader(csv_path))
        print(f"共读取 {len(ticks)} 条tick数据")
        
        if len(ticks) < 2:
            print("数据量不足，无法进行分析")
            return {}
        
        # 分析买入和卖出两个方向
        buy_results = self._analyze_side(ticks, "buy")
        sell_results = self._analyze_side(ticks, "sell")
        
        # 打印Q值分布分析（使用买入方向的Q值，因为Q值是一样的）
        if buy_results.get('q_values'):
            self._print_q_distribution_analysis(buy_results['q_values'], buy_results, sell_results)
        
        return {
            'buy': buy_results,
            'sell': sell_results,
            'total_ticks': len(ticks)
        }
    
    def analyze_csv(self, csv_path: str, side: str = "buy") -> Dict:
        """
        分析CSV文件中的数据（单个方向，保持向后兼容）
        
        Args:
            csv_path: CSV文件路径
            side: 分析的交易方向 ('buy' or 'sell')
            
        Returns:
            分析结果字典
        """
        all_results = self.analyze_csv_both_sides(csv_path)
        return all_results.get(side, {})
    
    def _print_q_distribution_analysis(self, q_values: List[float], buy_results: Dict, sell_results: Dict):
        """打印Q值分布分析和条件满足统计"""
        import numpy as np
        q_array = np.array(q_values)
        print(f"\nQ值分布统计:")
        print(f"Q值范围: [{q_array.min():.4f}, {q_array.max():.4f}]")
        print(f"Q值均值: {q_array.mean():.4f}")
        print(f"Q值标准差: {q_array.std():.4f}")
        
        # 统计不同区间的Q值数量
        extreme_positive = np.sum(q_array > 0.5)
        extreme_negative = np.sum(q_array < -0.5)
        moderate_positive = np.sum((q_array > 0.2) & (q_array <= 0.5))
        moderate_negative = np.sum((q_array < -0.2) & (q_array >= -0.5))
        
        print(f"Q > 0.5 (极强买压): {extreme_positive} 次 ({extreme_positive/len(q_values)*100:.2f}%)")
        print(f"Q < -0.5 (极强卖压): {extreme_negative} 次 ({extreme_negative/len(q_values)*100:.2f}%)")
        print(f"0.2 < Q <= 0.5 (中等买压): {moderate_positive} 次 ({moderate_positive/len(q_values)*100:.2f}%)")
        print(f"-0.5 <= Q < -0.2 (中等卖压): {moderate_negative} 次 ({moderate_negative/len(q_values)*100:.2f}%)")
        
        print(f"\n双重条件满足统计:")
        print(f"买入方向:")
        print(f"  Q条件满足 (Q > {self.threshold}): {buy_results['q_condition_count']} 次 ({buy_results['q_condition_count']/len(q_values)*100:.2f}%)")
        print(f"  micro_price条件满足: {buy_results['micro_condition_count']} 次 ({buy_results['micro_condition_count']/len(q_values)*100:.2f}%)")
        print(f"  双重条件都满足: {buy_results['signal_count']} 次 ({buy_results['signal_count']/len(q_values)*100:.2f}%)")
        
        print(f"卖出方向:")
        print(f"  Q条件满足 (Q < -{self.threshold}): {sell_results['q_condition_count']} 次 ({sell_results['q_condition_count']/len(q_values)*100:.2f}%)")
        print(f"  micro_price条件满足: {sell_results['micro_condition_count']} 次 ({sell_results['micro_condition_count']/len(q_values)*100:.2f}%)")
        print(f"  双重条件都满足: {sell_results['signal_count']} 次 ({sell_results['signal_count']/len(q_values)*100:.2f}%)")

    def print_summary(self, results: Dict, side: str):
        """
        打印单个方向的分析结果摘要
        """
        if not results:
            print("没有分析结果")
            return
        
        print("\n" + "="*60)
        print("盘口失衡Q值信号分析结果")
        print("="*60)
        print(f"交易方向: {side}")
        print(f"失衡阈值: {self.threshold}")
        print(f"预测窗口: {self.window}秒")
        if side == "buy":
            print(f"信号条件: Q > {self.threshold} (买盘强) 且 micro_price更接近卖一价 (上涨趋势)")
        else:
            print(f"信号条件: Q < -{self.threshold} (卖盘强) 且 micro_price更接近买一价 (下跌趋势)")
        print("-"*60)
        print(f"总共满足条件次数: {results['total_signals']}")
        print(f"朝着有利方向移动: {results['favorable_count']} 次 ({results['favorable_probability']:.2f}%)")
        print(f"朝着不利方向移动: {results['unfavorable_count']} 次 ({results['unfavorable_probability']:.2f}%)")
        print(f"价格偏离不明显: {results['no_change_count']} 次 ({results['no_change_probability']:.2f}%)")
        print("-"*60)
        
        if results['total_signals'] > 0:
            # 计算窗口分析统计
            avg_window_ticks = sum(r.higher_count + r.lower_count + r.equal_count for r in results['signal_details']) / results['total_signals']
            
            print(f"平均窗口内tick数: {avg_window_ticks:.1f}")
            
            # 分析有利/不利情况的窗口特征
            if results['favorable_count'] > 0:
                favorable_signals = [r for r in results['signal_details'] if r.direction == 'favorable']
                if side == "buy":
                    avg_favorable_lower = sum(r.lower_count for r in favorable_signals) / len(favorable_signals)
                    print(f"有利情况下平均低价tick数: {avg_favorable_lower:.1f}")
                else:
                    avg_favorable_higher = sum(r.higher_count for r in favorable_signals) / len(favorable_signals) 
                    print(f"有利情况下平均高价tick数: {avg_favorable_higher:.1f}")
            
            if results['unfavorable_count'] > 0:
                unfavorable_signals = [r for r in results['signal_details'] if r.direction == 'unfavorable']
                if side == "buy":
                    avg_unfavorable_higher = sum(r.higher_count for r in unfavorable_signals) / len(unfavorable_signals)
                    print(f"不利情况下平均高价tick数: {avg_unfavorable_higher:.1f}")
                else:
                    avg_unfavorable_lower = sum(r.lower_count for r in unfavorable_signals) / len(unfavorable_signals)
                    print(f"不利情况下平均低价tick数: {avg_unfavorable_lower:.1f}")
        
        print("="*60)
    
    def print_detailed_signals(self, all_results: Dict, max_details: int = 10):
        """
        打印详细的信号触发记录
        
        Args:
            all_results: 包含买入和卖出结果的字典
            max_details: 最多显示的详细记录数量
        """
        print("\n" + "="*80)
        print("详细信号触发记录")
        print("="*80)
        
        # 合并买入和卖出的信号记录
        all_signals = []
        if 'buy' in all_results and all_results['buy'].get('signal_details'):
            all_signals.extend(all_results['buy']['signal_details'])
        if 'sell' in all_results and all_results['sell'].get('signal_details'):
            all_signals.extend(all_results['sell']['signal_details'])
        
        # 按时间排序
        all_signals.sort(key=lambda x: x.ts)
        
        print(f"总共记录到 {len(all_signals)} 次信号触发")
        print(f"显示前 {min(max_details, len(all_signals))} 条记录:")
        print("-" * 80)
        
        for i, signal in enumerate(all_signals[:max_details]):
            total_window_ticks = signal.higher_count + signal.lower_count + signal.equal_count
            print(f"{i+1:2d}. {signal.ts} | {signal.side:4s} | Q={signal.Q:6.3f} | "
                  f"价格={signal.current_price:7.2f} | 窗口tick={total_window_ticks:2d} | "
                  f"高={signal.higher_count:2d} 低={signal.lower_count:2d} 等={signal.equal_count:2d} | "
                  f"结果={signal.direction}")
        
        if len(all_signals) > max_details:
            print(f"... 还有 {len(all_signals) - max_details} 条记录")
        
        print("="*80)

def main():
    """
    主函数示例
    """
    # 创建计算器
    calculator = IceSmartCalculator(imbalance_threshold=0.5, prediction_window=3.0)
    
    # 使用指定的CSV文件路径
    csv_path = r"C:\Users\justr\Desktop\SHFE.cu2510.0.2025-07-07 00_00_00.2025-07-09 00_00_00.csv"
    
    print("开始分析铜期货数据...")
    print(f"文件路径: {csv_path}")
    print("-" * 80)
    
    try:
        # 一次性分析买入和卖出两个方向
        all_results = calculator.analyze_csv_both_sides(csv_path)
        
        if not all_results:
            print("分析失败，没有获得结果")
            return
        
        # 分别输出买入和卖出方向的摘要
        print("\n【买入方向分析摘要】")
        if 'buy' in all_results:
            calculator.print_summary(all_results['buy'], "buy")
        
        print("\n【卖出方向分析摘要】")  
        if 'sell' in all_results:
            calculator.print_summary(all_results['sell'], "sell")
        
        # 对比分析
        if 'buy' in all_results and 'sell' in all_results:
            print("\n" + "="*60)
            print("买卖方向对比分析")
            print("="*60)
            print(f"买入信号总数: {all_results['buy']['total_signals']}")
            print(f"卖出信号总数: {all_results['sell']['total_signals']}")
            print(f"买入不利概率: {all_results['buy']['unfavorable_probability']:.2f}%")
            print(f"卖出不利概率: {all_results['sell']['unfavorable_probability']:.2f}%")
            print("="*60)
        
        # 输出详细的信号触发记录
        calculator.print_detailed_signals(all_results, max_details=20)
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_path}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查CSV文件格式是否正确")

if __name__ == "__main__":
    main()
