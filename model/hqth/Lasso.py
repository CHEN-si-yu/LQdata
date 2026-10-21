import os
import pickle
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from lassonet import LassoNetRegressor

warnings.filterwarnings("ignore")


# 获取训练和测试时间段
def get_train_date_split(season):
    test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2) + "01"
    start_date = datetime.strptime(test_start, "%Y%m%d")
    valid_date_split = [(start_date - timedelta(days=d)).strftime("%Y%m%d") for d in [-90, 0, 720]]
    valid_date_split.reverse()

    train_start = valid_date_split[0]
    test_end = valid_date_split[-1]

    print(f"valid_date_split: {valid_date_split}")
    print(f"train_start: {train_start}")
    print(f"test_start: {test_start}")
    print(f"test_end: {test_end}")

    return train_start, test_start, test_end


# 对数值列进行标准化并截尾
def normed(df):
    df_num = df.select_dtypes(include=[np.number])
    df_num = np.clip(df_num, df_num.quantile(0.005), df_num.quantile(0.995), axis=1)
    df_num = (df_num - df_num.mean()) / df_num.std()
    df_num = df_num.fillna(0)
    df[df_num.columns] = df_num
    return df


from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import time

def train(season):

        print(f"Training for season: {season}")

        save_path = "./output/"
        os.makedirs(save_path, exist_ok=True)

        # 路径保存与加载
        path_file = os.path.join(save_path, f"{season}_lassonet_path.pkl")
     
        print("Loaded path from lassonet_path.pkl")
        with open(path_file, "rb") as f:
            path = pickle.load(f)
            

        # 模型评估
        n_selected, mse, lambda_ = [], [], []
        selected = []
        for save in path:
            n_selected.append(save.selected.sum())
            selected.append(save.selected.cpu().numpy())
            mse.append(save.val_loss)
            lambda_.append(save.lambda_)
        
        # 保存selected特征
        selected_df = pd.DataFrame(selected, columns=[f"feature_{i}" for i in range(len(selected[0]))])
        selected_df['n_selected'] = n_selected
        selected_df['mse'] = mse
        selected_df['lambda'] = lambda_
        selected_df.to_csv(os.path.join(save_path, f"{season}_selected_features.csv"), index=False)

        # 绘图
        plt.figure(figsize=(12, 12))

        plt.subplot(311)
        plt.grid(True)
        plt.plot(n_selected, mse, ".-")
        plt.xlabel("number of selected features")
        plt.ylabel("MSE")

        plt.subplot(312)
        plt.grid(True)
        plt.plot(lambda_, mse, ".-")
        plt.xlabel("lambda")
        plt.xscale("log")
        plt.ylabel("MSE")

        plt.subplot(313)
        plt.grid(True)
        plt.plot(lambda_, n_selected, ".-")
        plt.xlabel("lambda")
        plt.xscale("log")
        plt.ylabel("number of selected features")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{season}_boston-group.png"))
        plt.close()

        print(f"Finished training for season: {season}")


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == "__main__":
    seasons = ["2023q1", "2023q2", "2023q3", "2023q4", "2024q1", "2024q2", "2024q3", "2024q4"]
    
    for season in seasons:
        train(season)
        print(f"Completed training for {season}")