from tqsdk import TqApi, TqAuth
import os
import pandas as pd

api = TqApi(auth=TqAuth("ringo", "Shinny456"))


ls = api.query_cont_quotes()

ls_symbol_info = api.query_symbol_info(ls)
# 直接转为 DataFrame，选取所需列，并保证每个品种唯一
df = pd.DataFrame(ls_symbol_info)[["product_id", "price_tick"]]
df = df.dropna(subset=["product_id", "price_tick"])  # 去掉缺失
df = df.drop_duplicates(subset=["product_id"], keep="first")  # 每个品种唯一
df = df.sort_values(by=["product_id"]).reset_index(drop=True)

# 输出到项目根目录的 results/price_tick.xlsx（无则创建目录）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(base_dir, "results")
os.makedirs(out_dir, exist_ok=True)
xlsx_path = os.path.join(out_dir, "price_tick.xlsx")

try:
    df.to_excel(xlsx_path, index=False)
    print(f"已保存: {xlsx_path}")
except Exception as e:
    # 若环境缺少 openpyxl/xlsxwriter，则回退为 CSV
    csv_path = os.path.join(out_dir, "price_tick.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Excel 写入失败({e})，已改存 CSV: {csv_path}")
finally:
    api.close()








