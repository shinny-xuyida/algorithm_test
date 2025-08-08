#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多交易日合约测试脚本
测试单个包含多个交易日的合约文件
"""

import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

# 添加项目根目录到路径（脚本位于 scripts/ 目录，需添加上级目录）
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# 导入分析器
from tools.data_evaluation_long_period import ImbalanceAnalyzer


def process_single_contract(args):
    """处理单个合约的函数，用于多进程调用"""
    contract, data_path = args
    
    try:
        test_file = os.path.join(data_path, contract)
        
        if not os.path.exists(test_file):
            return contract, None, f"文件不存在"
        
        # 创建分析器实例
        analyzer = ImbalanceAnalyzer(
            imbalance_threshold=0.8,
            future_window_seconds=3,
            verbose=False
        )
        
        start_time = time.time()
        daily_results = analyzer.analyze_file_by_trading_days(test_file)
        process_time = time.time() - start_time
        
        if daily_results:
            # 计算统计信息
            total_signals_all = sum(r.total_signals for r in daily_results.values())
            total_valid_all = sum(r.valid_signals for r in daily_results.values())
            
            # 各日有效率、反向率、无效率
            valid_ratios = [r.valid_signals / r.total_signals for r in daily_results.values() if r.total_signals > 0]
            reverse_ratios = [r.reverse_signals / r.total_signals for r in daily_results.values() if r.total_signals > 0]
            invalid_ratios = [r.invalid_signals / r.total_signals for r in daily_results.values() if r.total_signals > 0]
            
            if valid_ratios:
                avg_valid_ratio = np.mean(valid_ratios)
                overall_valid_ratio = total_valid_all / total_signals_all if total_signals_all > 0 else 0
                
                # 计算稳定性指标
                def calculate_stability_metrics(ratios):
                    """计算稳定性指标"""
                    if len(ratios) <= 1:
                        return 0.0, 1.0, 0.0  # std, stability_score, range
                    
                    mean_val = np.mean(ratios)
                    std_val = np.std(ratios)
                    cv = std_val / mean_val if mean_val > 0 else float('inf')  # 变异系数
                    
                    # 稳定性评分：基于变异系数，越小越稳定
                    # CV < 0.1 为非常稳定，CV > 1.0 为很不稳定
                    stability_score = max(0, min(1, 1 - cv))
                    
                    volatility_range = np.max(ratios) - np.min(ratios)  # 波动范围
                    
                    return std_val, stability_score, volatility_range
                
                # 计算各类信号的稳定性
                valid_std, valid_stability, valid_range = calculate_stability_metrics(valid_ratios)
                reverse_std, reverse_stability, reverse_range = calculate_stability_metrics(reverse_ratios)
                invalid_std, invalid_stability, invalid_range = calculate_stability_metrics(invalid_ratios)
                
                # 处理每日详细数据
                daily_data = []
                for trading_date, daily_result in sorted(daily_results.items()):
                    daily_valid_ratio = daily_result.valid_signals / daily_result.total_signals if daily_result.total_signals > 0 else 0
                    daily_reverse_ratio = daily_result.reverse_signals / daily_result.total_signals if daily_result.total_signals > 0 else 0
                    daily_invalid_ratio = daily_result.invalid_signals / daily_result.total_signals if daily_result.total_signals > 0 else 0
                    daily_valid_minus_reverse = daily_valid_ratio - daily_reverse_ratio  # 修正：有效率-反向率
                    
                    daily_data.append({
                        'trading_date': daily_result.trading_date,
                        'total_signals': daily_result.total_signals,
                        'valid_ratio': daily_valid_ratio,
                        'reverse_ratio': daily_reverse_ratio,
                        'invalid_ratio': daily_invalid_ratio,
                        'valid_minus_reverse': daily_valid_minus_reverse  # 修正字段名
                    })
                
                # 计算整体比率
                total_reverse_all = sum(r.reverse_signals for r in daily_results.values())
                total_invalid_all = sum(r.invalid_signals for r in daily_results.values())
                
                overall_reverse_ratio = total_reverse_all / total_signals_all if total_signals_all > 0 else 0
                overall_invalid_ratio = total_invalid_all / total_signals_all if total_signals_all > 0 else 0
                
                # 计算各日有效率和反向率的波动指标
                daily_valid_ratios = [r.valid_signals / r.total_signals for r in daily_results.values() if r.total_signals > 0]
                daily_reverse_ratios = [r.reverse_signals / r.total_signals for r in daily_results.values() if r.total_signals > 0]
                
                # 计算波动范围和变异系数
                if len(daily_valid_ratios) > 1:
                    valid_range = max(daily_valid_ratios) - min(daily_valid_ratios)
                    valid_mean = np.mean(daily_valid_ratios)
                    valid_std = np.std(daily_valid_ratios)
                    valid_cv = valid_std / valid_mean if valid_mean > 0 else 0
                    
                    reverse_range = max(daily_reverse_ratios) - min(daily_reverse_ratios)
                    reverse_mean = np.mean(daily_reverse_ratios)
                    reverse_std = np.std(daily_reverse_ratios)
                    reverse_cv = reverse_std / reverse_mean if reverse_mean > 0 else 0
                else:
                    valid_range = reverse_range = 0.0
                    valid_cv = reverse_cv = 0.0
                
                result = {
                     'trading_days': len(daily_results),
                     'total_signals': total_signals_all,
                     'total_valid': total_valid_all,
                     'total_reverse': total_reverse_all,
                     'total_invalid': total_invalid_all,
                     'overall_valid_ratio': overall_valid_ratio,
                     'overall_reverse_ratio': overall_reverse_ratio,
                     'overall_invalid_ratio': overall_invalid_ratio,
                     'valid_volatility_range': valid_range,
                     'reverse_volatility_range': reverse_range,
                     'valid_cv': valid_cv,
                     'reverse_cv': reverse_cv,
                     'process_time': process_time,
                     'daily_data': daily_data  # 每日详细数据
                 }
                
                return contract, result, None
            else:
                return contract, None, "无有效数据"
        else:
            return contract, None, "分析失败"
            
    except Exception as e:
        return contract, None, f"处理异常: {str(e)}"


def export_results_to_files(all_contract_results, sorted_contracts, total_time, num_processes):
    """导出结果到文件"""
    import pandas as pd
    from datetime import datetime
    
    # 创建results目录
    os.makedirs('results', exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 导出汇总数据
    summary_data = []
    for rank, contract in enumerate(sorted_contracts, 1):
        result = all_contract_results[contract]
        summary_data.append({
            '排名': rank,
            '合约': contract.replace('.csv', ''),
            '总信号数': result['total_signals'],
            '整体有效率': result['overall_valid_ratio'],
            '整体反向率': result['overall_reverse_ratio'],
            '整体无效率': result['overall_invalid_ratio'],
            '有效率减反向率': result['overall_valid_ratio'] - result['overall_reverse_ratio'],
            '有效率波动范围': result['valid_volatility_range'],
            '反向率波动范围': result['reverse_volatility_range'],
            '有效率变异系数': result['valid_cv'],
            '反向率变异系数': result['reverse_cv'],
            '交易日数': result['trading_days'],
            '总有效信号': result['total_valid'],
            '总反向信号': result['total_reverse'],
            '总无效信号': result['total_invalid']
        })
    
    # 保存汇总数据
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"results/合约信号汇总_{timestamp}.xlsx"
    
    with pd.ExcelWriter(summary_filename, engine='openpyxl') as writer:
        # 汇总表
        summary_df.to_excel(writer, sheet_name='合约汇总', index=False)
        
        # 每日详细数据（前10名）
        daily_data_all = []
        for rank, contract in enumerate(sorted_contracts[:10], 1):
            result = all_contract_results[contract]
            daily_data = result.get('daily_data', [])
            
            for daily in daily_data:
                daily_data_all.append({
                    '排名': rank,
                    '合约': contract.replace('.csv', ''),
                    '交易日期': daily['trading_date'],
                    '总信号数': daily['total_signals'],
                    '有效率': daily['valid_ratio'],
                    '反向率': daily['reverse_ratio'],
                    '无效率': daily['invalid_ratio'],
                    '有效率减反向率': daily['valid_minus_reverse']
                })
        
        if daily_data_all:
            daily_df = pd.DataFrame(daily_data_all)
            daily_df.to_excel(writer, sheet_name='前10名每日详细', index=False)
        
        # 添加分析说明
        info_data = [
            ['分析参数', ''],
            ['失衡阈值', '0.8 (买方强度)'],
            ['观察窗口', '3秒'],
            ['分析时间', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['处理时间', f"{total_time:.1f}秒"],
            ['使用进程数', str(num_processes)],
            ['合约总数', str(len(all_contract_results))],
            ['', ''],
            ['评价标准', ''],
            ['优秀', '整体有效率 >= 50%'],
            ['良好', '整体有效率 >= 40%'],
            ['一般', '整体有效率 >= 30%'],
            ['较差', '整体有效率 < 30%'],
            ['', ''],
            ['稳定性指标', ''],
            ['变异系数 < 0.1', '非常稳定'],
            ['变异系数 0.1-0.2', '稳定'],
            ['变异系数 0.2-0.5', '一般'],
            ['变异系数 > 0.5', '不稳定']
        ]
        
        info_df = pd.DataFrame(info_data, columns=['项目', '说明'])
        info_df.to_excel(writer, sheet_name='分析说明', index=False)
    
    print(f"\n💾 数据导出完成:")
    print(f"📄 汇总文件: {summary_filename}")
    
    # 2. 导出纯文本报告
    report_filename = f"results/信号分析报告_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("多合约信号分析报告\n")
        f.write("="*80 + "\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"失衡阈值: 0.8 (买方强度)\n")
        f.write(f"观察窗口: 3秒\n")
        f.write(f"处理时间: {total_time:.1f}秒\n")
        f.write(f"使用进程数: {num_processes}\n")
        f.write(f"分析合约数: {len(all_contract_results)}\n\n")
        
        # 汇总统计
        total_signals_global = sum(r['total_signals'] for r in all_contract_results.values())
        total_valid_global = sum(r['total_valid'] for r in all_contract_results.values())
        total_reverse_global = sum(r['total_reverse'] for r in all_contract_results.values())
        total_invalid_global = sum(r['total_invalid'] for r in all_contract_results.values())
        
        f.write("全局统计:\n")
        f.write(f"  总信号数: {total_signals_global:,}\n")
        f.write(f"  总有效信号: {total_valid_global:,}\n")
        f.write(f"  总反向信号: {total_reverse_global:,}\n")
        f.write(f"  总无效信号: {total_invalid_global:,}\n")
        f.write(f"  全局有效率: {total_valid_global/total_signals_global:.1%}\n")
        f.write(f"  全局反向率: {total_reverse_global/total_signals_global:.1%}\n")
        f.write(f"  全局无效率: {total_invalid_global/total_signals_global:.1%}\n\n")
        
        # 前10名合约
        f.write("排名前10的合约:\n")
        f.write("-" * 80 + "\n")
        for rank, contract in enumerate(sorted_contracts[:10], 1):
            result = all_contract_results[contract]
            f.write(f"{rank:2d}. {contract.replace('.csv', ''):<20} "
                   f"有效率: {result['overall_valid_ratio']:>6.1%} "
                   f"信号数: {result['total_signals']:>6,} "
                   f"交易日: {result['trading_days']}天\n")
        
        f.write(f"\n详细数据请查看Excel文件: {summary_filename}\n")
    
    print(f"📄 报告文件: {report_filename}")
    print(f"📁 输出目录: results/")


def test_multi_contracts(num_processes=None):
    """测试多个合约文件 - 多进程版本"""
    print("🔍 多合约测试 - 买方0.8强度信号分析 (多进程版)")
    print("="*80)
    
    # 新的数据路径
    data_path = r"C:\Users\justr\Desktop\tqsdk_data_2025731_2025807"
    
    # 自动获取目录下所有CSV文件
    import glob
    csv_pattern = os.path.join(data_path, "*.csv")
    contract_files = glob.glob(csv_pattern)
    contracts = [os.path.basename(file) for file in contract_files]
    
    # 排除可能的非合约文件（如果文件名不符合预期格式）
    contracts = [f for f in contracts if not f.startswith('tqsdk_') and f.endswith('.csv')]
    
    # 自动配置进程数
    if num_processes is None:
        num_processes = min(cpu_count(), len(contracts))  # 不超过CPU核心数和合约数
    
    print(f"📂 数据路径: {data_path}")
    print(f"📁 发现合约文件: {len(contracts)} 个")
    print(f"🔄 进程数量: {num_processes}")
    print(f"💻 CPU核心数: {cpu_count()}")
    
    # 显示前几个合约文件作为示例
    if contracts:
        print(f"📝 合约文件示例:")
        for i, contract in enumerate(sorted(contracts)[:10]):
            print(f"  {i+1}. {contract}")
        if len(contracts) > 10:
            print(f"  ... 还有 {len(contracts) - 10} 个文件")
    
    # 检查文件存在性
    existing_contracts = []
    for contract in contracts:
        test_file = os.path.join(data_path, contract)
        if os.path.exists(test_file):
            existing_contracts.append(contract)
        else:
            print(f"⚠️ 跳过不存在的文件: {contract}")
    
    if not existing_contracts:
        print("❌ 没有找到任何有效的合约文件")
        return
    
    print(f"✅ 有效合约数量: {len(existing_contracts)}")
    
    # 准备多进程参数
    process_args = [(contract, data_path) for contract in existing_contracts]
    
    print(f"\n🚀 开始多进程处理...")
    total_start_time = time.time()
    
    # 存储结果
    all_contract_results = {}
    failed_contracts = []
    
    # 使用多进程池处理
    try:
        with Pool(processes=num_processes) as pool:
            # 显示进度
            print("📊 处理进度:")
            results = pool.map(process_single_contract, process_args)
            
            # 处理结果
            for contract, result, error in results:
                if result is not None:
                    all_contract_results[contract] = result
                    print(f"✅ {contract.replace('.csv', ''):<18} ({result['process_time']:.1f}s)")
                else:
                    failed_contracts.append((contract, error))
                    print(f"❌ {contract.replace('.csv', ''):<18} - {error}")
    
    except Exception as e:
        print(f"❌ 多进程执行出错: {str(e)}")
        return
    
    total_time = time.time() - total_start_time
    
    # 显示失败的合约
    if failed_contracts:
        print(f"\n⚠️ 处理失败的合约:")
        for contract, error in failed_contracts:
            print(f"  - {contract}: {error}")
    
    if not all_contract_results:
        print("❌ 没有获得任何分析结果")
        return
    
    # 显示最终汇总结果
    print(f"\n{'='*160}")
    print(f"📊 多合约信号汇总 - 买方0.8强度 (按整体有效率降序排列)")
    print(f"{'='*160}")
    print(f"{'合约':<18} {'总信号':<10} {'整体有效率':<12} {'整体反向率':<12} {'整体无效率':<12} {'有效-反向':<12} "
          f"{'有效率波动':<12} {'反向率波动':<12} {'有效率CV':<10} {'反向率CV':<10} {'评价':<8}")
    print("-" * 160)
    
    # 按整体有效率降序排列
    sorted_contracts = sorted(all_contract_results.keys(), 
                             key=lambda x: all_contract_results[x]['overall_valid_ratio'], 
                             reverse=True)
    
    for contract in sorted_contracts:
        result = all_contract_results[contract]
        
        # 评价逻辑
        overall_valid_ratio = result['overall_valid_ratio']
        overall_reverse_ratio = result['overall_reverse_ratio']
        overall_invalid_ratio = result['overall_invalid_ratio']
        valid_range = result['valid_volatility_range']
        reverse_range = result['reverse_volatility_range']
        valid_cv = result['valid_cv']
        reverse_cv = result['reverse_cv']
        
        if overall_valid_ratio >= 0.5:
            rating = "🔥 优秀"
        elif overall_valid_ratio >= 0.4:
            rating = "✅ 良好"
        elif overall_valid_ratio >= 0.3:
            rating = "⚠️ 一般"
        else:
            rating = "❌ 较差"
        
        # 计算有效率减反向率
        valid_minus_reverse = overall_valid_ratio - overall_reverse_ratio
        
        print(f"{contract.replace('.csv', ''):<18} {result['total_signals']:<10,} "
              f"{overall_valid_ratio:<12.1%} {overall_reverse_ratio:<12.1%} {overall_invalid_ratio:<12.1%} {valid_minus_reverse:<+12.1%} "
              f"{valid_range:<12.1%} {reverse_range:<12.1%} {valid_cv:<10.3f} {reverse_cv:<10.3f} {rating:<8}")
    
    # 全局统计
    total_signals_global = sum(r['total_signals'] for r in all_contract_results.values())
    total_valid_global = sum(r['total_valid'] for r in all_contract_results.values())
    total_reverse_global = sum(r['total_reverse'] for r in all_contract_results.values())
    total_invalid_global = sum(r['total_invalid'] for r in all_contract_results.values())
    total_trading_days = sum(r['trading_days'] for r in all_contract_results.values())
    
    # 全局整体比率
    global_valid_ratio = total_valid_global / total_signals_global if total_signals_global > 0 else 0
    global_reverse_ratio = total_reverse_global / total_signals_global if total_signals_global > 0 else 0
    global_invalid_ratio = total_invalid_global / total_signals_global if total_signals_global > 0 else 0
    
    # 各合约的整体比率
    overall_valid_ratios = [r['overall_valid_ratio'] for r in all_contract_results.values()]
    overall_reverse_ratios = [r['overall_reverse_ratio'] for r in all_contract_results.values()]
    overall_invalid_ratios = [r['overall_invalid_ratio'] for r in all_contract_results.values()]
    
    # 波动指标
    valid_ranges = [r['valid_volatility_range'] for r in all_contract_results.values()]
    reverse_ranges = [r['reverse_volatility_range'] for r in all_contract_results.values()]
    valid_cvs = [r['valid_cv'] for r in all_contract_results.values()]
    reverse_cvs = [r['reverse_cv'] for r in all_contract_results.values()]
    
    print("-" * 160)
    print(f"📈 全局统计汇总:")
    print(f"  - 合约数量: {len(all_contract_results)}")
    print(f"  - 总交易日数: {total_trading_days}")
    print(f"  - 总信号数: {total_signals_global:,}")
    print(f"  - 总有效信号: {total_valid_global:,}")
    print(f"  - 总反向信号: {total_reverse_global:,}")
    print(f"  - 总无效信号: {total_invalid_global:,}")
    print(f"  - 全局整体有效率: {global_valid_ratio:.1%}")
    print(f"  - 全局整体反向率: {global_reverse_ratio:.1%}")
    print(f"  - 全局整体无效率: {global_invalid_ratio:.1%}")
    # 计算有效率减反向率的统计
    overall_valid_minus_reverse = [overall_valid_ratios[i] - overall_reverse_ratios[i] for i in range(len(overall_valid_ratios))]
    global_valid_minus_reverse = global_valid_ratio - global_reverse_ratio
    
    print(f"  - 合约平均整体有效率: {np.mean(overall_valid_ratios):.1%}")
    print(f"  - 合约平均整体反向率: {np.mean(overall_reverse_ratios):.1%}")
    print(f"  - 合约平均整体无效率: {np.mean(overall_invalid_ratios):.1%}")
    print(f"  - 全局有效率减反向率: {global_valid_minus_reverse:+.1%}")
    print(f"  - 合约平均有效率减反向率: {np.mean(overall_valid_minus_reverse):+.1%}")
    print(f"  - 合约平均有效率波动范围: {np.mean(valid_ranges):.1%}")
    print(f"  - 合约平均反向率波动范围: {np.mean(reverse_ranges):.1%}")
    print(f"  - 合约平均有效率变异系数: {np.mean(valid_cvs):.3f}")
    print(f"  - 合约平均反向率变异系数: {np.mean(reverse_cvs):.3f}")
    
    # 只显示排名前10的合约每日详细统计
    top_contracts = sorted_contracts[:10]
    print(f"\n{'='*140}")
    print(f"📅 排名前10合约的每日详细统计")
    print(f"{'='*140}")
    
    for rank, contract in enumerate(top_contracts, 1):
        result = all_contract_results[contract]
        daily_data = result.get('daily_data', [])
        
        if not daily_data:
            continue
            
        print(f"\n📊 第{rank}名 - 合约: {contract.replace('.csv', '')} (整体有效率: {result['overall_valid_ratio']:.1%})")
        print("-" * 90)
        print(f"{'交易日期':<12} {'总信号':<10} {'有效率':<10} {'反向率':<10} {'无效率':<10} {'有效-反向':<12}")
        print("-" * 90)
        
        for daily in daily_data:
            print(f"{daily['trading_date']:<12} {daily['total_signals']:<10,} {daily['valid_ratio']:<10.1%} "
                  f"{daily['reverse_ratio']:<10.1%} {daily['invalid_ratio']:<10.1%} {daily['valid_minus_reverse']:<+12.1%}")
        
        # 显示该合约的统计摘要
        print("-" * 140)
        print(f"小计: 交易日{result['trading_days']}天, 总信号{result['total_signals']:,}, "
              f"整体有效率{result['overall_valid_ratio']:.1%}, "
              f"整体反向率{result['overall_reverse_ratio']:.1%}, "
              f"整体无效率{result['overall_invalid_ratio']:.1%}")
        print(f"      有效率波动{result['valid_volatility_range']:.1%}, "
              f"反向率波动{result['reverse_volatility_range']:.1%}, "
              f"有效率CV{result['valid_cv']:.3f}, "
              f"反向率CV{result['reverse_cv']:.3f}")

    # 显示处理时间和性能统计
    avg_process_time = sum(r['process_time'] for r in all_contract_results.values()) / len(all_contract_results)
    sequential_time = sum(r['process_time'] for r in all_contract_results.values())
    speedup_ratio = sequential_time / total_time if total_time > 0 else 1
    
    print(f"\n{'='*160}")
    print(f"⏱️ 性能统计:")
    print(f"  - 多进程总时间: {total_time:.1f}s")
    print(f"  - 估计串行时间: {sequential_time:.1f}s") 
    print(f"  - 性能提升倍数: {speedup_ratio:.1f}x")
    print(f"  - 平均每合约处理时间: {avg_process_time:.1f}s")
    print(f"  - 使用进程数: {num_processes}")
    print(f"  - 处理效率: {len(all_contract_results)/total_time:.1f} 合约/秒")
    
    print(f"\n{'='*160}")
    print(f"📊 波动指标说明:")
    print(f"  - 波动范围: 最大值 - 最小值，越小越稳定")
    print(f"  - 变异系数(CV): 标准差 / 均值，越小越稳定")
    print(f"  - CV < 0.1: 🔥 非常稳定")
    print(f"  - CV 0.1-0.2: ✅ 稳定") 
    print(f"  - CV 0.2-0.5: ⚠️ 一般")
    print(f"  - CV > 0.5: ❌ 不稳定")
    
    # 导出数据到文件
    export_results_to_files(all_contract_results, sorted_contracts, total_time, num_processes)


if __name__ == "__main__":
    # Windows上多进程需要此保护
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 可以通过参数指定进程数，默认自动选择
    # test_multi_contracts(num_processes=4)  # 指定4个进程
    test_multi_contracts()  # 自动选择进程数

