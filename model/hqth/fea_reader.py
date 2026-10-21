import pandas as pd 
import matplotlib.pyplot as plt



name_list = ["2023q1_num-1400.fea", "2023q2_num-1400.fea", "2023q3_num-1400.fea", "2023q4_num-1400.fea", "2024q1_num-1400.fea", "2024q2_num-1400.fea"]

for name in name_list:
    # 读取 Feather 文件
    print("📂 正在读取 Feather 文件...")
    data = pd.read_feather(rf"/project/model_share/share_1_new/factor_dly/fac20250528_1400icbase/{name}")
    print("✅ Feather 文件读取完成！")
    data = data.set_index("date").sort_index()
    date_list = data.index.unique().tolist()
    Code_list_data = data["Code"].unique().tolist()

    total_stock_count = len(Code_list_data)
    total_date_count = len(date_list)
    stock_appearance_days = data.groupby("Code").size()  # 返回 Series：每只股票对应的行数（即出现的天数）
    stocks_per_day = data.groupby("date")["Code"].nunique()  # 返回 Series：每个日期对应的不重复股票数
    # stock_appearance_days[:10]
    stocks_per_day[-10:]
    stock_appearance_days_list = stock_appearance_days.to_list()
    stocks_per_day_list = stocks_per_day.to_list()


    plt.figure(figsize=(10, 6))
    plt.hist(stock_appearance_days_list, bins=50)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(stock_appearance_days_list, marker='o', linestyle='-', color='b')
    plt.show()



    plt.figure(figsize=(10, 6))
    plt.hist(stocks_per_day_list, bins=50)
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(stocks_per_day_list, marker='o', linestyle='-', color='b')
    plt.show()
    good_code_index = stock_appearance_days[stock_appearance_days > 800].index.tolist()
    good_data = data[data["Code"].isin(good_code_index)]
    good_data[:10]
    # 1. 获取所有唯一日期和股票代码
    all_dates = good_data.index.unique()          # 因为你的 data 已设定 "date" 为 index
    all_codes = good_data["Code"].unique()

    # 2. 创建 MultiIndex：所有日期 × 所有股票
    full_index = pd.MultiIndex.from_product([all_dates, all_codes], names=["date", "Code"])

    # 3. 将原数据以 MultiIndex 形式重建，并 reindex 到完整索引上
    good_data_full = good_data.reset_index().set_index(["date", "Code"]).reindex(full_index)

    # 1. 重置索引，把 "date" 和 "Code" 变回普通列
    good_data_full_reset = good_data_full.reset_index()

    # 2. 保存为 feather 格式（.fea 只是简称，实际上是 .feather 文件）
    good_data_full_reset.to_feather(rf"/project/model_share/share_1_new/factor_csy/1400icbase/{name}")