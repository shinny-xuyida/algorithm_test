# algorithm_comparison.py
# 算法对比控制文件：运行多种策略并对比表现
# 
# 多进程版本 - 提升回测性能
# 
# 使用方法:
#   python algorithm_comparison_test_data.py                    # 默认多进程模式
#   python algorithm_comparison_test_data.py --processes 4     # 指定4个进程
#   python algorithm_comparison_test_data.py --no-multiprocess # 单进程模式
#   python algorithm_comparison_test_data.py --benchmark       # 性能基准测试
#
# 注意：在Windows下使用多进程时，务必确保脚本在 if __name__ == "__main__": 保护下运行

import sys
import os
import time
import glob
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from functools import partial

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strategy import (
    IceBestStrategy, 
    IceHangStrategy, 
    IceSmartStrategy,
    IceSmartOnlyImbalanceStrategy
)
from core.backtest_engine import run_backtest

def process_single_file(csv_path: str, side: str, total_qty: int, slice_qty: int, start_time: str) -> Tuple[str, Dict[str, Dict]]:
    """
    处理单个文件的所有策略（用于多进程）
    返回: (filename, file_results)
    """
    filename = os.path.basename(csv_path)
    
    try:
        print(f"🔄 [{os.getpid()}] 开始处理: {filename}")
        
        # 加载数据
        from core.market_data import tick_reader
        ticks_data = list(tick_reader(csv_path))
        if not ticks_data:
            print(f"⚠️  [{os.getpid()}] 警告: {filename} 数据为空")
            return filename, {}
        
        # 创建策略
        strategies = {
            "冰山对价策略": IceBestStrategy(side, total_qty, slice_qty),
            "冰山挂价策略": IceHangStrategy(side, total_qty, slice_qty),
            "冰山智能策略": IceSmartStrategy(side, total_qty, slice_qty, imbalance_threshold=0.2),
            "智能失衡策略": IceSmartOnlyImbalanceStrategy(side, total_qty, slice_qty, imbalance_threshold=0.2)
        }
        
        # 运行所有策略
        file_results = {}
        for strategy_name, strategy in strategies.items():
            try:
                result = run_backtest(
                    csv_path=csv_path,
                    strategy=strategy,
                    start_time=start_time,
                    ticks_data=ticks_data
                )
                result['策略名称'] = strategy_name
                file_results[strategy_name] = result
            except Exception as e:
                file_results[strategy_name] = {
                    '策略名称': strategy_name,
                    '执行状态': '失败',
                    '错误信息': str(e)
                }
        
        print(f"✅ [{os.getpid()}] 完成处理: {filename}")
        return filename, file_results
        
    except Exception as e:
        print(f"❌ [{os.getpid()}] 处理 {filename} 失败: {str(e)}")
        return filename, {}

class AlgorithmComparison:
    """算法对比器"""
    
    def __init__(self, results_dir: str = "results", n_processes: int = None):
        self.results = {}
        self.batch_results = {}  # 存储批量测试结果
        self.results_dir = results_dir
        # 设置进程数，默认为CPU核心数
        self.n_processes = n_processes if n_processes is not None else max(1, cpu_count() - 1)
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"🔧 配置信息: 使用 {self.n_processes} 个进程进行并行处理")
        
    def create_strategies(self, side: str, total_qty: int, slice_qty: int) -> Dict[str, Any]:
        """创建所有策略实例"""
        strategies = {
            "冰山对价策略": IceBestStrategy(side, total_qty, slice_qty),
            "冰山挂价策略": IceHangStrategy(side, total_qty, slice_qty),
            "冰山智能策略": IceSmartStrategy(side, total_qty, slice_qty, imbalance_threshold=0.3),
            "智能失衡策略": IceSmartOnlyImbalanceStrategy(side, total_qty, slice_qty, imbalance_threshold=0.3)
        }
        return strategies
    
    def run_single_strategy(self, strategy_name: str, strategy: Any, csv_path: str, start_time: str, ticks_data: list) -> Dict:
        """运行单个策略（使用预加载的数据）"""
        try:
            result = run_backtest(
                csv_path=csv_path,
                strategy=strategy,
                start_time=start_time,
                ticks_data=ticks_data
            )
            result['策略名称'] = strategy_name
            return result
        except Exception as e:
            return {
                '策略名称': strategy_name,
                '执行状态': '失败',
                '错误信息': str(e)
            }
    
    def run_all_strategies(self, csv_path: str, side: str = "buy", total_qty: int = 200, 
                          slice_qty: int = 5, start_time: str = None) -> Dict[str, Dict]:
        """运行所有策略并收集结果（优化：数据只加载一次）"""
        if start_time is None:
            start_time = "2025-07-17 14:00:00"
        
        print(f"📈 加载数据: {os.path.basename(csv_path)}")
        
        # 只加载一次数据
        try:
            from core.market_data import tick_reader
            ticks_data = list(tick_reader(csv_path))
            if not ticks_data:
                print(f"⚠️  警告: {csv_path} 数据为空")
                return {}
            print(f"✅ 数据加载完成，共 {len(ticks_data)} 条tick数据")
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return {}
        
        # 创建策略并运行
        strategies = self.create_strategies(side, total_qty, slice_qty)
        results = {}
        
        print(f"🚀 开始运行 {len(strategies)} 个策略...")
        for i, (strategy_name, strategy) in enumerate(strategies.items(), 1):
            print(f"  ({i}/{len(strategies)}) 运行 {strategy_name}...")
            result = self.run_single_strategy(strategy_name, strategy, csv_path, start_time, ticks_data)
            results[strategy_name] = result
        
        self.results = results
        return results
    
    def batch_run_all_files(self, data_dir: str = "test_data", side: str = "buy", 
                           total_qty: int = 200, slice_qty: int = 5, start_time: str = None, 
                           use_multiprocess: bool = True) -> Dict[str, Dict[str, Dict]]:
        """批量运行test_data目录下的所有CSV文件（支持多进程）"""
        
        # 如果启用多进程，调用多进程版本
        if use_multiprocess and self.n_processes > 1:
            return self.batch_run_all_files_multiprocess(data_dir, side, total_qty, slice_qty, start_time)
        
        # 否则使用单进程版本
        
        # 获取所有CSV文件
        csv_pattern = os.path.join(data_dir, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"❌ 在 {data_dir} 目录下未找到CSV文件")
            return {}
        
        print(f"🔍 发现 {len(csv_files)} 个数据文件")
        print("⚡ 使用单进程模式")
        print("="*80)
        
        batch_results = {}
        
        for i, csv_path in enumerate(csv_files, 1):
            filename = os.path.basename(csv_path)
            print(f"\n📊 [{i}/{len(csv_files)}] 处理文件: {filename}")
            print("-" * 60)
            
            # 运行该文件的所有策略
            file_results = self.run_all_strategies(
                csv_path=csv_path,
                side=side,
                total_qty=total_qty,
                slice_qty=slice_qty,
                start_time=start_time
            )
            
            if file_results:
                batch_results[filename] = file_results
                # 打印该文件的简要结果
                self.print_file_summary(filename, file_results)
            else:
                print(f"❌ {filename} 处理失败")
        
        self.batch_results = batch_results
        return batch_results
    
    def batch_run_all_files_multiprocess(self, data_dir: str = "test_data", side: str = "buy", 
                                        total_qty: int = 200, slice_qty: int = 5, start_time: str = None) -> Dict[str, Dict[str, Dict]]:
        """多进程批量运行test_data目录下的所有CSV文件"""
        
        # 获取所有CSV文件
        csv_pattern = os.path.join(data_dir, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"❌ 在 {data_dir} 目录下未找到CSV文件")
            return {}
        
        print(f"🔍 发现 {len(csv_files)} 个数据文件")
        print(f"🚀 使用 {self.n_processes} 个进程并行处理")
        print("="*80)
        
        # 创建进程池并执行
        batch_results = {}
        start_time_total = time.time()
        
        try:
            # 使用partial创建带参数的函数
            process_func = partial(
                process_single_file,
                side=side,
                total_qty=total_qty,
                slice_qty=slice_qty,
                start_time=start_time
            )
            
            # 根据文件数量优化进程数
            optimal_processes = min(self.n_processes, len(csv_files))
            
            # 创建进程池
            with Pool(processes=optimal_processes) as pool:
                print(f"📊 开始并行处理... (使用 {optimal_processes} 个进程)")
                
                # 并行处理所有文件，使用imap获得进度反馈
                completed = 0
                for filename, file_results in pool.imap(process_func, csv_files):
                    completed += 1
                    progress = completed / len(csv_files) * 100
                    
                    if file_results:
                        batch_results[filename] = file_results
                        print(f"\n📈 [{completed}/{len(csv_files)}] ({progress:.1f}%) 处理完成: {filename}")
                        self.print_file_summary(filename, file_results)
                    else:
                        print(f"❌ [{completed}/{len(csv_files)}] ({progress:.1f}%) {filename} 处理失败")
        
        except Exception as e:
            print(f"❌ 多进程处理失败: {str(e)}")
            return {}
        
        end_time_total = time.time()
        total_duration = end_time_total - start_time_total
        
        print(f"\n⏱️  总执行时间: {total_duration:.2f} 秒")
        print(f"📊 平均每文件处理时间: {total_duration/len(csv_files):.2f} 秒")
        
        self.batch_results = batch_results
        return batch_results
    
    def print_file_summary(self, filename: str, results: Dict[str, Dict]):
        """打印单个文件的简要结果"""
        print(f"\n📈 {filename} 结果:")
        
        # 获取有效结果并排序
        valid_results = []
        for strategy_name, result in results.items():
            if result.get('执行状态') != '失败' and isinstance(result.get('价格滑点'), (int, float)):
                valid_results.append((strategy_name, result))
        
        if not valid_results:
            print("  ❌ 无有效结果")
            return
        
        # 按滑点排序（买入时滑点越小越好）
        valid_results.sort(key=lambda x: x[1].get('价格滑点', float('inf')))
        
        print("  排名:")
        for i, (strategy_name, result) in enumerate(valid_results, 1):
            slippage = result.get('价格滑点', 'N/A')
            slippage_str = f"{slippage:.4f}" if isinstance(slippage, (int, float)) else str(slippage)
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"  {i}."
            print(f"    {medal} {strategy_name} (滑点: {slippage_str})")
    
    def generate_overall_statistics(self, batch_results: Dict[str, Dict[str, Dict]]) -> Dict[str, Any]:
        """生成整体统计数据（以平均成交价第一名次数为主，买入为最低次数，卖出为最高次数）"""
        if not batch_results:
            return {}
        # 统计数据
        strategy_rankings = defaultdict(list)  # 策略名 -> [排名列表]
        strategy_stats = defaultdict(lambda: {
            'first_count': 0, 'last_count': 0, 'total_files': 0, 'avg_rank': 0,
            'exec_times': [], 'avg_exec_time': 0, 'min_exec_time': float('inf'), 'max_exec_time': 0,
            'order_counts': [], 'avg_order_count': 0, 'min_order_count': float('inf'), 'max_order_count': 0,
            'avg_prices': [], 'avg_avg_price': 0, 'min_avg_price': float('inf'), 'max_avg_price': float('-inf')
        })
        total_files = len(batch_results)
        file_count = 0
        print("\n🔄 正在生成整体统计数据...")
        for filename, file_results in batch_results.items():
            # 获取该文件的有效结果并排序
            valid_results = []
            for strategy_name, result in file_results.items():
                if result.get('执行状态') != '失败' and isinstance(result.get('平均成交价格'), (int, float)):
                    valid_results.append((strategy_name, result))
            if not valid_results:
                continue
            file_count += 1
            # 动态判断side
            side = 'buy'
            for _, res in valid_results:
                if 'side' in res:
                    side = res.get('side', 'buy')
                    break
            # 排序方向
            if side == 'buy':
                valid_results.sort(key=lambda x: x[1].get('平均成交价格', float('inf')))
            else:
                valid_results.sort(key=lambda x: x[1].get('平均成交价格', float('-inf')), reverse=True)
            # 记录每个策略的排名、执行时长、委托笔数、均价
            for rank, (strategy_name, result) in enumerate(valid_results, 1):
                strategy_stats[strategy_name]['total_files'] += 1
                strategy_rankings[strategy_name].append(rank)
                # 执行时长
                exec_time = result.get('执行时长(秒)', None)
                if isinstance(exec_time, (int, float)) and exec_time is not None:
                    strategy_stats[strategy_name]['exec_times'].append(exec_time)
                    if exec_time < strategy_stats[strategy_name]['min_exec_time']:
                        strategy_stats[strategy_name]['min_exec_time'] = exec_time
                    if exec_time > strategy_stats[strategy_name]['max_exec_time']:
                        strategy_stats[strategy_name]['max_exec_time'] = exec_time
                # 委托笔数
                order_count = result.get('委托笔数', None)
                if isinstance(order_count, (int, float)) and order_count is not None:
                    strategy_stats[strategy_name]['order_counts'].append(order_count)
                    if order_count < strategy_stats[strategy_name]['min_order_count']:
                        strategy_stats[strategy_name]['min_order_count'] = order_count
                    if order_count > strategy_stats[strategy_name]['max_order_count']:
                        strategy_stats[strategy_name]['max_order_count'] = order_count
                # 平均成交价格
                avg_price = result.get('平均成交价格', None)
                if isinstance(avg_price, (int, float)) and avg_price is not None:
                    strategy_stats[strategy_name]['avg_prices'].append(avg_price)
                    if avg_price < strategy_stats[strategy_name]['min_avg_price']:
                        strategy_stats[strategy_name]['min_avg_price'] = avg_price
                    if avg_price > strategy_stats[strategy_name]['max_avg_price']:
                        strategy_stats[strategy_name]['max_avg_price'] = avg_price
                if rank == 1:
                    strategy_stats[strategy_name]['first_count'] += 1
                elif rank == len(valid_results):
                    strategy_stats[strategy_name]['last_count'] += 1
        # 计算平均排名、平均执行时长、平均委托笔数、平均成交价格
        for strategy_name, rankings in strategy_rankings.items():
            if rankings:
                strategy_stats[strategy_name]['avg_rank'] = sum(rankings) / len(rankings)
            # 平均执行时长
            exec_times = strategy_stats[strategy_name]['exec_times']
            if exec_times:
                strategy_stats[strategy_name]['avg_exec_time'] = sum(exec_times) / len(exec_times)
            else:
                strategy_stats[strategy_name]['min_exec_time'] = None
                strategy_stats[strategy_name]['max_exec_time'] = None
            # 平均委托笔数
            order_counts = strategy_stats[strategy_name]['order_counts']
            if order_counts:
                strategy_stats[strategy_name]['avg_order_count'] = sum(order_counts) / len(order_counts)
            else:
                strategy_stats[strategy_name]['min_order_count'] = None
                strategy_stats[strategy_name]['max_order_count'] = None
            # 平均成交价格
            avg_prices = strategy_stats[strategy_name]['avg_prices']
            if avg_prices:
                strategy_stats[strategy_name]['avg_avg_price'] = sum(avg_prices) / len(avg_prices)
            else:
                strategy_stats[strategy_name]['min_avg_price'] = None
                strategy_stats[strategy_name]['max_avg_price'] = None
        # 计算百分比
        for strategy_name, stats in strategy_stats.items():
            total = stats['total_files']
            if total > 0:
                stats['first_rate'] = stats['first_count'] / total * 100
                stats['last_rate'] = stats['last_count'] / total * 100
        # 汇总主方向
        overall_side = 'buy'
        for file_results in batch_results.values():
            for result in file_results.values():
                if 'side' in result:
                    overall_side = result.get('side', 'buy')
                    break
            break
        return {
            'strategy_stats': dict(strategy_stats),
            'strategy_rankings': dict(strategy_rankings),
            'total_valid_files': file_count,
            'total_files': total_files,
            'side': overall_side
        }

    def print_overall_statistics(self, batch_results: Dict[str, Dict[str, Dict]]):
        """打印整体统计结果（以平均成交价第一名次数为主，买入为最低次数，卖出为最高次数）"""
        stats = self.generate_overall_statistics(batch_results)
        if not stats:
            print("❌ 无有效统计数据")
            return
        strategy_stats = stats['strategy_stats']
        total_valid_files = stats['total_valid_files']
        side = stats.get('side', 'buy')
        print("\n" + "="*120)
        print("📊 整体统计报告")
        print("="*120)
        print(f"📁 总计处理文件: {stats['total_files']} 个")
        print(f"✅ 有效结果文件: {total_valid_files} 个")
        print(f"📈 方向: {'买入' if side == 'buy' else '卖出'} (排名以{'最低' if side == 'buy' else '最高'}均价第一次数为优)")
        print()
        # 排序：按第一名次数降序，然后按平均排名升序
        sorted_strategies = sorted(
            strategy_stats.items(),
            key=lambda x: (-x[1]['first_count'], x[1]['avg_rank'])
        )
        # 打印综合表现统计表
        print("📈 策略综合表现统计:")
        print("-" * 120)
        print(f"{'策略名称':<18} {'第一名次数':<10} {'第一名率':<10} {'最后名次数':<12} {'最后名率':<10} {'平均排名':<10} {'均价(均值)':<16}")
        print("-" * 120)
        for strategy_name, stats_data in sorted_strategies:
            first_count = stats_data['first_count']
            first_rate = stats_data['first_rate']
            last_count = stats_data['last_count']
            last_rate = stats_data['last_rate']
            avg_rank = stats_data['avg_rank']
            avg_avg_price = stats_data['avg_avg_price']
            avg_avg_price_str = f"{avg_avg_price:.4f}" if avg_avg_price is not None else "N/A"
            print(f"{strategy_name:<18} {first_count:<10} {first_rate:<9.1f}% {last_count:<12} {last_rate:<9.1f}% {avg_rank:<9.2f} {avg_avg_price_str:<16}")
        print("-" * 120)
        # 打印执行时长统计表
        print("\n⏱️  策略执行时长统计:")
        print("-" * 120)
        print(f"{'策略名称':<18} {'平均执行时长(秒)':<16} {'最短执行时长(秒)':<16} {'最长执行时长(秒)':<16} {'有效测试数':<12}")
        print("-" * 120)
        sorted_by_time = sorted(
            strategy_stats.items(),
            key=lambda x: x[1]['avg_exec_time'] if x[1]['avg_exec_time'] > 0 else float('inf')
        )
        for strategy_name, stats_data in sorted_by_time:
            avg_time = stats_data['avg_exec_time']
            min_time = stats_data['min_exec_time']
            max_time = stats_data['max_exec_time']
            valid_count = len(stats_data['exec_times'])
            avg_time_str = f"{avg_time:.2f}" if avg_time > 0 else "N/A"
            min_time_str = f"{min_time:.2f}" if min_time is not None and min_time != float('inf') else "N/A"
            max_time_str = f"{max_time:.2f}" if max_time is not None and max_time > 0 else "N/A"
            print(f"{strategy_name:<18} {avg_time_str:<16} {min_time_str:<16} {max_time_str:<16} {valid_count:<12}")
        print("-" * 120)
        # 打印委托笔数统计表
        print("\n📊 策略委托笔数统计:")
        print("-" * 120)
        print(f"{'策略名称':<18} {'平均委托笔数':<16} {'最短委托笔数':<16} {'最长委托笔数':<16} {'有效测试数':<12}")
        print("-" * 120)
        for strategy_name, stats_data in sorted_strategies:
            avg_order_count = stats_data['avg_order_count']
            min_order_count = stats_data['min_order_count']
            max_order_count = stats_data['max_order_count']
            valid_count = len(stats_data['order_counts'])
            avg_order_count_str = f"{avg_order_count:.2f}" if avg_order_count > 0 else "N/A"
            min_order_count_str = f"{min_order_count:.2f}" if min_order_count is not None and min_order_count != float('inf') else "N/A"
            max_order_count_str = f"{max_order_count:.2f}" if max_order_count is not None and max_order_count > 0 else "N/A"
            print(f"{strategy_name:<18} {avg_order_count_str:<16} {min_order_count_str:<16} {max_order_count_str:<16} {valid_count:<12}")
        print("-" * 120)
        # 找出最佳和最差策略（以第一名次数为主）
        best_strategy = max(strategy_stats.items(), key=lambda x: x[1]['first_count'], default=(None, None))
        worst_strategy = min(strategy_stats.items(), key=lambda x: x[1]['first_count'], default=(None, None))
        most_stable = min(strategy_stats.items(), key=lambda x: x[1]['avg_rank'])
        fastest_strategy = min(
            [(name, stats_data) for name, stats_data in strategy_stats.items() if stats_data['avg_exec_time'] > 0],
            key=lambda x: x[1]['avg_exec_time'],
            default=(None, None)
        )
        print("\n🏆 策略评价:")
        if best_strategy[0]:
            print(f"  🥇 最佳策略: {best_strategy[0]} (第一名次数最多 {best_strategy[1]['first_count']} 次)")
        print(f"  📈 最稳定策略: {most_stable[0]} (平均排名 {most_stable[1]['avg_rank']:.2f})")
        if worst_strategy[0]:
            print(f"  🔻 表现最差策略: {worst_strategy[0]} (第一名次数最少 {worst_strategy[1]['first_count']} 次)")
        if fastest_strategy[0]:
            print(f"  ⚡ 执行最快策略: {fastest_strategy[0]} (平均执行时长 {fastest_strategy[1]['avg_exec_time']:.2f} 秒)")
        print(f"\n💡 策略推荐:")
        if best_strategy[0]:
            print(f"  🌟 强烈推荐: {best_strategy[0]} (第一名次数最多，适合追求极致成交价)")
        elif most_stable[1]['avg_rank'] < 2.5:
            print(f"  🌟 稳定推荐: {most_stable[0]} (平均排名优秀且稳定)")
        else:
            print("  🤔 各策略表现相近，建议根据具体市场情况选择")
        if fastest_strategy[0]:
            print(f"  ⚡ 速度考量: 如果对执行速度有要求，推荐 {fastest_strategy[0]} (平均执行时长 {fastest_strategy[1]['avg_exec_time']:.2f} 秒)")
        print("="*120)
    
    def save_batch_results(self, batch_results: Dict[str, Dict[str, Dict]], filename: str = None):
        """保存批量测试结果"""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"batch_comparison_results_{timestamp}.txt"
        
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"批量算法对比测试结果 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                
                # 写入每个文件的详细结果
                for filename, file_results in batch_results.items():
                    f.write(f"📊 文件: {filename}\n")
                    f.write("-" * 40 + "\n")
                    
                    for strategy_name, result in file_results.items():
                        f.write(f"【{strategy_name}】\n")
                        for key, value in result.items():
                            f.write(f"  {key}: {value}\n")
                        f.write("\n")
                
                # 写入整体统计
                stats = self.generate_overall_statistics(batch_results)
                if stats:
                    f.write("="*50 + "\n")
                    f.write("📊 整体统计报告\n")
                    f.write("="*50 + "\n\n")
                    
                    strategy_stats = stats['strategy_stats']
                    sorted_strategies = sorted(
                        strategy_stats.items(),
                        key=lambda x: (-x[1]['first_count'], x[1]['avg_rank'])
                    )
                    
                    f.write("策略综合表现统计:\n")
                    f.write("-" * 50 + "\n")
                    for strategy_name, stat in sorted_strategies:
                        f.write(f"【{strategy_name}】\n")
                        f.write(f"  第一名次数: {stat['first_count']}\n")
                        f.write(f"  第一名率: {stat['first_rate']:.1f}%\n")
                        f.write(f"  最后名次数: {stat['last_count']}\n")
                        f.write(f"  最后名率: {stat['last_rate']:.1f}%\n")
                        f.write(f"  平均排名: {stat['avg_rank']:.2f}\n")
                        f.write("\n")
                    
                    f.write("策略执行时长统计:\n")
                    f.write("-" * 50 + "\n")
                    # 按平均执行时长排序
                    sorted_by_time = sorted(
                        strategy_stats.items(),
                        key=lambda x: x[1]['avg_exec_time'] if x[1]['avg_exec_time'] > 0 else float('inf')
                    )
                    
                    for strategy_name, stat in sorted_by_time:
                        avg_time = stat['avg_exec_time']
                        min_time = stat['min_exec_time']
                        max_time = stat['max_exec_time']
                        valid_count = len(stat['exec_times'])
                        
                        f.write(f"【{strategy_name}】\n")
                        f.write(f"  平均执行时长: {avg_time:.2f}秒\n" if avg_time > 0 else "  平均执行时长: N/A\n")
                        f.write(f"  最短执行时长: {min_time:.2f}秒\n" if min_time is not None and min_time != float('inf') else "  最短执行时长: N/A\n")
                        f.write(f"  最长执行时长: {max_time:.2f}秒\n" if max_time is not None and max_time > 0 else "  最长执行时长: N/A\n")
                        f.write(f"  有效测试数: {valid_count}\n")
                        f.write("\n")
                    
                    f.write("策略委托笔数统计:\n")
                    f.write("-" * 50 + "\n")
                    # 按平均委托笔数排序
                    sorted_by_order_count = sorted(
                        strategy_stats.items(),
                        key=lambda x: x[1]['avg_order_count'] if x[1]['avg_order_count'] > 0 else float('inf')
                    )
                    
                    for strategy_name, stat in sorted_by_order_count:
                        avg_order_count = stat['avg_order_count']
                        min_order_count = stat['min_order_count']
                        max_order_count = stat['max_order_count']
                        valid_count = len(stat['order_counts'])
                        
                        f.write(f"【{strategy_name}】\n")
                        f.write(f"  平均委托笔数: {avg_order_count:.2f}\n" if avg_order_count > 0 else "  平均委托笔数: N/A\n")
                        f.write(f"  最短委托笔数: {min_order_count:.2f}\n" if min_order_count is not None and min_order_count != float('inf') else "  最短委托笔数: N/A\n")
                        f.write(f"  最长委托笔数: {max_order_count:.2f}\n" if max_order_count is not None and max_order_count > 0 else "  最长委托笔数: N/A\n")
                        f.write(f"  有效测试数: {valid_count}\n")
                        f.write("\n")
            
            print(f"✅ 批量测试结果已保存到: {filepath}")
        except Exception as e:
            print(f"❌ 保存失败: {str(e)}")

    def print_comparison_table(self, results: Dict[str, Dict]):
        """打印对比表格"""
        print("\n" + "="*110)
        print("📊 策略表现对比")
        print("="*110)
        
        # 打印表头
        print(f"{'策略名称':<16} {'平均成交价格':<12} {'市场VWAP':<12} {'价格滑点':<10} "
              f"{'执行时长(秒)':<12} {'成交笔数':<8} {'委托笔数':<8}")
        print("-" * 110)
        
        # 收集有效结果用于排序
        valid_results = []
        
        for strategy_name, result in results.items():
            if result.get('执行状态') == '失败':
                print(f"{strategy_name:<16} {'执行失败':<50}")
                continue
            
            # 获取各项指标
            avg_price = result.get('平均成交价格', 'N/A')
            market_vwap = result.get('市场VWAP', 'N/A')
            slippage = result.get('价格滑点', 'N/A')
            exec_duration = result.get('执行时长(秒)', 'N/A')
            fill_count = result.get('成交笔数', 'N/A')
            order_count = result.get('委托笔数', 'N/A')
            
            # 格式化显示
            avg_price_str = f"{avg_price:.4f}" if isinstance(avg_price, (int, float)) else str(avg_price)
            market_vwap_str = f"{market_vwap:.4f}" if isinstance(market_vwap, (int, float)) else str(market_vwap)
            slippage_str = f"{slippage:.4f}" if isinstance(slippage, (int, float)) else str(slippage)
            exec_duration_str = f"{exec_duration:.2f}" if isinstance(exec_duration, (int, float)) else str(exec_duration)
            
            print(f"{strategy_name:<16} {avg_price_str:<12} {market_vwap_str:<12} {slippage_str:<10} "
                  f"{exec_duration_str:<12} {fill_count:<8} {order_count:<8}")
            
            # 添加到有效结果中用于排序
            if isinstance(slippage, (int, float)):
                valid_results.append((strategy_name, result))
        
        print("-" * 110)
        
        # 按滑点排序并显示排名
        if valid_results:
            # 买入时滑点越小越好，卖出时滑点越大越好
            side = 'buy'  # 默认买入
            if any(results.values()):
                first_result = list(results.values())[0]
                if 'side' in first_result:
                    side = first_result.get('side', 'buy')
            
            if side == "buy":
                valid_results.sort(key=lambda x: x[1].get('价格滑点', float('inf')))
            else:
                valid_results.sort(key=lambda x: x[1].get('价格滑点', float('-inf')), reverse=True)
            
            print("\n🏆 策略排名（按价格滑点排序）:")
            for i, (strategy_name, result) in enumerate(valid_results, 1):
                slippage = result.get('价格滑点', 'N/A')
                slippage_str = f"{slippage:.4f}" if isinstance(slippage, (int, float)) else str(slippage)
                print(f"  {i}. {strategy_name} (滑点: {slippage_str})")
        
        print("="*110)
    
    def save_results_to_file(self, results: Dict[str, Dict], filename: str = None):
        """保存结果到指定文件夹"""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"comparison_results_{timestamp}.txt"
        
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"算法对比测试结果 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                
                for strategy_name, result in results.items():
                    f.write(f"【{strategy_name}】\n")
                    for key, value in result.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
            
            print(f"结果已保存到: {filepath}")
        except Exception as e:
            print(f"保存失败: {str(e)}")


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='算法对比测试系统')
    parser.add_argument('--processes', type=int, default=None, 
                        help=f'进程数量 (默认: {max(1, cpu_count() - 1)})')
    parser.add_argument('--no-multiprocess', action='store_true', 
                        help='禁用多进程，使用单进程模式')
    parser.add_argument('--benchmark', action='store_true', 
                        help='运行性能基准测试（分别测试单进程和多进程）')
    
    args = parser.parse_args()
    
    # 创建对比器，指定结果保存目录和进程数
    comparator = AlgorithmComparison(
        results_dir="results", 
        n_processes=args.processes
    )
    
    print("🚀 开始批量算法对比测试")
    print("="*80)
    
    # 判断是否运行基准测试
    if args.benchmark:
        print("🔥 运行性能基准测试...")
        
        # 单进程测试
        print("\n📍 单进程模式基准测试:")
        start_time = time.time()
        batch_results_single = comparator.batch_run_all_files(
            data_dir="test_data",
            side="buy",
            total_qty=200,
            slice_qty=5,
            start_time="2025-07-18 09:40:00",
            use_multiprocess=False
        )
        single_duration = time.time() - start_time
        
        # 多进程测试
        print(f"\n📍 多进程模式基准测试 ({comparator.n_processes} 进程):")
        start_time = time.time()
        batch_results_multi = comparator.batch_run_all_files(
            data_dir="test_data",
            side="buy",
            total_qty=200,
            slice_qty=5,
            start_time="2025-07-18 09:40:00",
            use_multiprocess=True
        )
        multi_duration = time.time() - start_time
        
        # 性能对比
        print("\n" + "="*80)
        print("⚡ 性能基准测试结果")
        print("="*80)
        print(f"🐌 单进程模式耗时: {single_duration:.2f} 秒")
        print(f"🚀 多进程模式耗时: {multi_duration:.2f} 秒")
        if single_duration > 0:
            speedup = single_duration / multi_duration
            print(f"📈 加速比: {speedup:.2f}x")
            efficiency = speedup / comparator.n_processes * 100
            print(f"📊 并行效率: {efficiency:.1f}%")
        print("="*80)
        
        batch_results = batch_results_multi
    else:
        # 正常运行
        use_multiprocess = not args.no_multiprocess
        batch_results = comparator.batch_run_all_files(
            data_dir="test_data",
            side="buy",
            total_qty=200,
            slice_qty=5,
            start_time="2025-07-18 09:40:00",
            use_multiprocess=use_multiprocess
        )
    
    if batch_results:
        # 打印整体统计
        comparator.print_overall_statistics(batch_results)
        
        # 保存批量结果
        comparator.save_batch_results(batch_results)
        
        print(f"\n🎉 批量测试完成！共处理 {len(batch_results)} 个数据文件")
    else:
        print("❌ 批量测试失败，无有效结果")


if __name__ == "__main__":
    main() 