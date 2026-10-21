import pandas as pd
import numpy as np
from tqdm import tqdm

data = pd.read_feather(rf"/project/model_share/user96_factor/fac20250613/fill_zero.fea")


for season in tqdm(['2023q1', '2023q2', '2023q3', '2023q4', 
                    '2024q1', '2024q2', '2024q3', '2024q4']):
    
    fac1 = pd.read_feather(rf"/home/user96/Factor_splitting/fac20250613_de_score_{season}.fea")
    fac2 = np.loadtxt(rf"/home/user96/20250903_trainmodel1/model_test/nn--fac20250613_ic500--label1--dropout0.2/ALL{season}-feature_map.fea", dtype=str) 
    
    topindex_list = fac1.nlargest(500, 'value')['index'].tolist()
    topindex_list.sort()
    fac2_list = [name.split('=')[0] for name in fac2.tolist()]
    fac2_list.sort()

    # 合并两个列表
    all_factors = set(topindex_list).union(set(fac2_list))
    print(f"Season: {season}")
    print(f"Total unique factors: {all_factors}")

    # 筛选列
    filtered_data = data[[col for col in data.columns if col in all_factors or col in ['Code', 'date']]]
    print(f"Filtered data shape: {filtered_data.shape}")
    filtered_data.to_feather(rf"/project/model_share/user96_factor/fac20250613_IC_DE/{season}_num-{len(all_factors)}.fea")

