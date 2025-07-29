# contract_multiplier.py
# -------------------------------------------------------------------
# 合约乘数配置模块：存储各品种的合约乘数信息
# -------------------------------------------------------------------

from typing import Optional

# 合约乘数字典
CONTRACT_MULTIPLIERS = {
    # 中金所股指期货
    "IC": 200,  # 中证500股指期货
    "IF": 300,  # 沪深300股指期货  
    "IH": 300,  # 上证50股指期货
    "IM": 200,  # 中证1000股指期货
    
    # 中金所国债期货
    "T": 10000,   # 10年期国债期货
    "TF": 10000,  # 5年期国债期货
    "TL": 10000,  # 2年期国债期货
    "TS": 20000,  # 30年期国债期货
    
    # 郑商所农产品
    "AP": 10,  # 苹果期货
    "CF": 5,   # 棉花期货
    "CJ": 5,   # 红枣期货
    "CY": 5,   # 棉纱期货
    "FG": 20,  # 玻璃期货
    "JR": 20,  # 粳稻期货
    "LR": 20,  # 晚籼稻期货
    "MA": 10,  # 甲醇期货
    "OI": 10,  # 菜籽油期货
    "PF": 5,   # 短纤期货
    "PK": 5,   # 花生期货
    "PM": 50,  # 普麦期货
    "PR": 15,  # 尿素期货
    "PX": 5,   # PX期货
    "RI": 20,  # 早籼稻期货
    "RM": 10,  # 菜籽粕期货
    "RS": 10,  # 菜籽期货
    "SA": 20,  # 纯碱期货
    "SF": 5,   # 硅铁期货
    "SH": 30,  # 纸浆期货
    "SM": 5,   # 锰硅期货
    "SR": 10,  # 白糖期货
    "TA": 5,   # PTA期货
    "UR": 20,  # 尿素期货
    "WH": 20,  # 强麦期货
    "ZC": 100, # 动力煤期货
    
    # 大商所农产品
    "a": 10,   # 豆一期货
    "b": 10,   # 豆二期货
    "bb": 500, # 胶合板期货
    "bz": 30,  # 石油期货
    "c": 10,   # 玉米期货
    "cs": 10,  # 玉米淀粉期货
    "eb": 5,   # 苯乙烯期货
    "eg": 10,  # 乙二醇期货
    "fb": 10,  # 纤维板期货
    "i": 100,  # 铁矿石期货
    "j": 100,  # 焦炭期货
    "jd": 10,  # 鸡蛋期货
    "jm": 60,  # 焦煤期货
    "l": 5,    # 聚乙烯期货
    "lg": 90,  # 玻璃期货
    "lh": 16,  # 生猪期货
    "m": 10,   # 豆粕期货
    "p": 10,   # 棕榈油期货
    "pg": 20,  # 液化石油气期货
    "pp": 5,   # 聚丙烯期货
    "rr": 10,  # 粳米期货
    "v": 5,    # PVC期货
    "y": 10,   # 豆油期货
    
    # 大商所其他
    "lc": 1,   # 碳酸锂期货
    "ps": 3,   # 苯乙烯期货
    "si": 5,   # 硅铁期货
    "bc": 5,   # 国际铜期货
    "ec": 50,  # 集装箱运价期货
    "lu": 10,  # 低硫燃料油期货
    "nr": 10,  # 20号胶期货
    "sc": 1000, # 原油期货
    
    # 上期所金属期货
    "ad": 10,  # 不锈钢期货
    "ag": 15,  # 白银期货
    "al": 5,   # 铝期货
    "ao": 20,  # 氧化铝期货
    "au": 1000, # 黄金期货
    "br": 5,   # 丁二烯橡胶期货
    "bu": 10,  # 沥青期货
    "cu": 5,   # 铜期货
    "fu": 10,  # 燃料油期货
    "hc": 10,  # 热轧卷板期货
    "ni": 1,   # 镍期货
    "pb": 5,   # 铅期货
    "rb": 10,  # 螺纹钢期货
    "ru": 10,  # 天然橡胶期货
    "sn": 1,   # 锡期货
    "sp": 10,  # 纸浆期货
    "ss": 5,   # 不锈钢期货
    "wr": 10,  # 线材期货
    "zn": 5,   # 锌期货
}


def get_contract_multiplier(contract_code: str) -> Optional[int]:
    """
    根据合约代码获取合约乘数
    
    Args:
        contract_code: 合约代码，例如 "cu2510", "IF2312" 等
        
    Returns:
        合约乘数，如果找不到则返回None
        
    Examples:
        >>> get_contract_multiplier("cu2510")
        5
        >>> get_contract_multiplier("IF2312") 
        300
        >>> get_contract_multiplier("unknown")
        None
    """
    if not contract_code:
        return None
    
    # 提取品种代码（去掉数字部分）
    instrument_code = ""
    for char in contract_code:
        if char.isalpha():
            instrument_code += char
        else:
            break
    
    return CONTRACT_MULTIPLIERS.get(instrument_code)


def get_all_instruments():
    """
    获取所有支持的品种代码列表
    
    Returns:
        所有品种代码的列表
    """
    return list(CONTRACT_MULTIPLIERS.keys())


def is_supported_instrument(contract_code: str) -> bool:
    """
    检查合约是否被支持
    
    Args:
        contract_code: 合约代码
        
    Returns:
        True表示支持，False表示不支持
    """
    return get_contract_multiplier(contract_code) is not None


def extract_contract_from_filename(csv_path: str) -> str:
    """
    从CSV文件名中提取合约代码
    
    Args:
        csv_path: CSV文件路径，支持多种格式：
        - "SHFE.cu2510.0.2025-07-07 00_00_00.2025-07-09 00_00_00.csv"
        - "rb2510_tick_2025_07_17_2025_07_17.csv"
        
    Returns:
        合约代码，例如 "cu2510", "rb2510"
    """
    import os
    filename = os.path.basename(csv_path)
    
    # 去掉文件扩展名
    filename_without_ext = filename.replace('.csv', '')
    
    # 格式1：SHFE.cu2510.0.2025-07-07 形式
    if '.' in filename_without_ext:
        parts = filename_without_ext.split('.')
        if len(parts) >= 2:
            return parts[1]  # 假设格式为 "交易所.合约代码.其他信息.csv"
    
    # 格式2：rb2510_tick_2025_07_17_2025_07_17 形式
    elif '_tick_' in filename_without_ext:
        parts = filename_without_ext.split('_tick_')
        if len(parts) >= 1:
            return parts[0]  # 取_tick_之前的部分作为合约代码
    
    # 格式3：如果没有特殊分隔符，尝试提取字母+数字的组合
    else:
        # 寻找连续的字母后跟数字的模式
        import re
        match = re.match(r'^([a-zA-Z]+\d+)', filename_without_ext)
        if match:
            return match.group(1)
    
    return ""


def get_contract_info_from_file(csv_path: str, default_multiplier: int = 1, verbose: bool = True) -> tuple[str, int]:
    """
    从CSV文件路径自动提取合约代码并获取合约乘数
    
    Args:
        csv_path: CSV文件路径
        default_multiplier: 当找不到合约乘数时使用的默认值，默认为1
        verbose: 是否打印提取过程信息，默认为True
        
    Returns:
        (合约代码, 合约乘数) 的元组
        
    Examples:
        >>> get_contract_info_from_file("SHFE.cu2510.0.2025-07-07.csv")
        ('cu2510', 5)
        >>> get_contract_info_from_file("unknown.xy9999.csv", default_multiplier=10)
        ('xy9999', 10)
    """
    # 提取合约代码
    contract_code = extract_contract_from_filename(csv_path)
    if verbose:
        print(f"检测到合约: {contract_code}")
    
    # 获取合约乘数
    multiplier = get_contract_multiplier(contract_code)
    if multiplier is None:
        if verbose:
            print(f"警告: 未找到合约 {contract_code} 的乘数信息，使用默认值 {default_multiplier}")
        multiplier = default_multiplier
    else:
        if verbose:
            print(f"合约乘数: {multiplier}")
    
    return contract_code, multiplier


# 测试用例
if __name__ == "__main__":
    # 测试一些常见合约
    test_contracts = ["cu2510", "IF2312", "au2409", "rb2501", "unknown123"]
    
    print("合约乘数查询测试:")
    print("-" * 40)
    for contract in test_contracts:
        multiplier = get_contract_multiplier(contract)
        if multiplier:
            print(f"{contract:<10} -> {multiplier}")
        else:
            print(f"{contract:<10} -> 未找到")
    
    print(f"\n总共支持 {len(get_all_instruments())} 个品种")
    
    # 测试文件名解析功能
    print("\n文件名解析测试:")
    print("-" * 40)
    
    # 测试旧格式
    test_filename1 = "SHFE.cu2510.0.2025-07-07 00_00_00.2025-07-09 00_00_00.csv"
    extracted1 = extract_contract_from_filename(test_filename1)
    print(f"旧格式文件名: {test_filename1}")
    print(f"提取的合约: {extracted1}")
    
    # 测试新格式
    test_filename2 = "rb2510_tick_2025_07_17_2025_07_17.csv"
    extracted2 = extract_contract_from_filename(test_filename2)
    print(f"新格式文件名: {test_filename2}")
    print(f"提取的合约: {extracted2}")
    
    # 测试一体化合约信息获取
    print("\n一体化合约信息获取测试:")
    print("-" * 40)
    
    # 测试旧格式
    contract_code1, multiplier1 = get_contract_info_from_file(test_filename1)
    print(f"旧格式返回结果: 合约={contract_code1}, 乘数={multiplier1}")
    
    # 测试新格式
    contract_code2, multiplier2 = get_contract_info_from_file(test_filename2)
    print(f"新格式返回结果: 合约={contract_code2}, 乘数={multiplier2}") 