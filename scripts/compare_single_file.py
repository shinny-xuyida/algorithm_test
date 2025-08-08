"""
compare_single_file.py
单文件策略对比脚本：运行多种策略并对比表现
"""

import sys
import os
import time
from typing import Dict, Any

# 添加项目根目录到路径（脚本位于 scripts/ 目录，需添加上级目录）
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from strategy import (
    IceBestStrategy, 
    IceHangStrategy, 
    IceSmartStrategy,
    IceSmartOnlyImbalanceStrategy
)
from core.backtest_engine import run_backtest

class AlgorithmComparison:
    """算法对比器"""
    
    def __init__(self, results_dir: str = "results"):
        self.results = {}
        self.results_dir = results_dir
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        
    def create_strategies(self, side: str, total_qty: int, slice_qty: int) -> Dict[str, Any]:
        """创建所有策略实例"""
        strategies = {
            "冰山对价策略": IceBestStrategy(side, total_qty, slice_qty),
            "冰山挂价策略": IceHangStrategy(side, total_qty, slice_qty),
            "冰山智能策略": IceSmartStrategy(side, total_qty, slice_qty, imbalance_threshold=0.2),
            "智能失衡策略": IceSmartOnlyImbalanceStrategy(side, total_qty, slice_qty, imbalance_threshold=0.2)
        }
        return strategies
    
    def run_single_strategy(self, strategy_name: str, strategy: Any, csv_path: str, start_time: str, ticks_data: list = None) -> Dict:
        """运行单个策略"""
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
        """运行所有策略并收集结果"""
        if start_time is None:
            start_time = "2025-07-17 14:00:00"
        
        # 读取数据
        from core.market_data import tick_reader
        ticks_data = list(tick_reader(csv_path))
        
        # 创建策略并运行
        strategies = self.create_strategies(side, total_qty, slice_qty)
        results = {}
        
        for strategy_name, strategy in strategies.items():
            result = self.run_single_strategy(strategy_name, strategy, csv_path, start_time, ticks_data)
            results[strategy_name] = result
        
        self.results = results
        return results
    
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
    csv_path = r"C:\Users\justr\Desktop\rb2510_tick_2025_07_17_2025_07_17.csv"
    
    if not os.path.exists(csv_path):
        print(f"找不到数据文件: {csv_path}")
        return
    
    # 创建对比器，指定结果保存目录
    comparator = AlgorithmComparison(results_dir="results")
    
    # 运行策略对比
    results = comparator.run_all_strategies(
        csv_path=csv_path,
        side="buy",
        total_qty=200,
        slice_qty=5,
        start_time="2025-07-17 14:00:00"
    )
    
    # 打印对比表格
    comparator.print_comparison_table(results)
    
    # 保存结果
    comparator.save_results_to_file(results)


if __name__ == "__main__":
    main() 