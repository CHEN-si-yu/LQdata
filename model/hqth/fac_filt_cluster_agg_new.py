import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import warnings
from datetime import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
warnings.filterwarnings("ignore")

# label和liqduid
label_path = rf'/home/user96/label_data1'
label_name = rf'label1'
liquid_path = rf'/home/user96/label_data1'
liquid_name = rf'can_trade_amt1'
liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")

'''原始因子部分'''
fac_path = rf'/project/model_share/share_1_new/factor_data/fac20250613_out'
fac_name = rf'fac20250613'
fac_data = pd.read_feather(rf'{fac_path}/{fac_name}/{fac_name}.fea')
season_list = ["2023q1", "2023q2", "2023q3", "2023q4", "2024q1", "2024q2", "2024q3", "2024q4"]
date_list = [x for x in fac_data['date'].unique() if x in ret_data.index and x in liquid_data.index]
date_list.sort()
fac_data = fac_data.set_index('date')

# 对于season季度的测试集，在季度开始前（test_start前）获取前month个月的日期列表，用这些日期来评测因子
def get_eval_date(all_date, season, month):
    test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2)
    start_date = datetime.strptime(test_start, "%Y%m")
    train_start = (start_date - relativedelta(months=month)).strftime("%Y%m")
    train_date_list = [x for x in all_date if train_start <= x < test_start][:-10]
    not_train_date = [x for x in date_list if (x >= "202402") & (x <= "20240223")]
    train_date_list = [x for x in train_date_list if x not in not_train_date]
    train_date_list.sort()
    return train_date_list, train_date_list[0], train_date_list[-1]

# 按ic调整多空头收益
def adjust_sign(info_in):
    info = info_in.astype('float')
    sign_ic = np.sign(info['ic'])
    head_cols = [x for x in info.columns if 'head' in x]
    tail_cols = [x for x in info.columns if 'tail' in x]
    temp_head = info[head_cols].copy()
    temp_tail = info[tail_cols].copy()
    cond = sign_ic < 0
    info.loc[cond, head_cols] = temp_tail.loc[cond].values
    info.loc[cond, tail_cols] = temp_head.loc[cond].values
    return info


# 多进程计算每日因子的回测表现
class FacMetric(Dataset):

    def __init__(self, data, date):
        self.fac_data = data
        self.date_list = date

    def __getitem__(self, index):
        # 取date日的因子，计算表现
        date = self.date_list[index]
        ret = ret_data.loc[date].dropna()
        ret_rank = ret.rank(pct=True, method='dense')
        amt = liquid_data.loc[date].dropna()
        amt_ret = pd.concat([amt, ret], axis=1, keys=['amt', 'ret']).fillna(0)
        amt_ret['amt_ret'] = amt_ret['amt'] * amt_ret['ret']


        fac_td = self.fac_data.loc[date].set_index('Code')
        fac_rank = fac_td.rank(pct=True, method='dense')

        fac_list = fac_td.columns.tolist()
        fac_info = pd.DataFrame(index=fac_list)

        # ir和rank_ic
        fac_info['ic'] = fac_td.corrwith(ret)
        fac_info['rank_ic'] = fac_rank.corrwith(ret_rank)

        # 打分头/尾10%的收益率——兼容数据问题，防止除0错误
        head10p = (fac_rank > 0.9).T
        tail10p = (fac_rank < 0.1).T
        fac_info['head10p'] = head10p.dot(ret.reindex(fac_rank.index).fillna(0)) / head10p.sum(axis=1).replace(0, np.nan)
        fac_info['tail10p'] = tail10p.dot(ret.reindex(fac_rank.index).fillna(0)) / tail10p.sum(axis=1).replace(0, np.nan)

        def htamt_ret(code_list, tot_amt):
            amt_ret_new = amt_ret.reindex(code_list).fillna(0)
            amt_ret_new['cum_amt'] = amt_ret_new['amt'].cumsum()
            amt_ret_ht = amt_ret_new.loc[amt_ret_new['cum_amt'] <= tot_amt]
            return amt_ret_ht['amt_ret'].sum() / amt_ret_ht['amt'].sum()

        money = '1.5e9'
        htamt = {}
        for fac_name in fac_list:
            htamt[fac_name] = {}
            code_head = fac_td[fac_name].sort_values(ascending=False).dropna().index.tolist()
            code_tail = fac_td[fac_name].sort_values(ascending=True).dropna().index.tolist()
            htamt[fac_name][f'head{money}'] = htamt_ret(code_head, eval(money))
            htamt[fac_name][f'tail{money}'] = htamt_ret(code_tail, eval(money))

        fac_info = pd.concat([fac_info, pd.DataFrame(htamt).T], axis=1)
        fac_info.insert(0, 'date', date)
        return fac_info

    def __len__(self):
        return len(self.date_list)


# 合并每日因子表现至fac_info_all
cal_res = DataLoader(FacMetric(fac_data, date_list), collate_fn=lambda x: x, num_workers=128)
fac_info_all = pd.concat([res[0] for res in tqdm(cal_res)])
fac_info_all = fac_info_all.reset_index(drop=False).rename(columns={'index': 'fac_name'})
fac_name_all = fac_info_all['fac_name'].unique()
# fac_info_all.reset_index(drop=True).to_feather(rf'/project/user40/model_share/share_2/factor_dly/20250625_fac_info_all.fea')

# fac_info_all = pd.read_feather(rf'/project/user40/model_share/share_2/factor_dly/0625all+zjh0701all_fac_info_all.fea')
# fac_info_all = pd.read_feather(rf'/project/user40/model_share/share_2/factor_dly/20250625_fac_info_all.fea')
# fac_name_all = fac_info_all['fac_name'].unique()







# 截面相关性计算
def get_corr(fac_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for date in tqdm(date_list):
        save_path = os.path.join(save_dir, f"{date}_corr.npy")
        # 断点重启：如果对应季度文件已存在则跳过
        if os.path.exists(save_path):
            print(f"[已完成] 跳过 {date}")
            continue
        
        c_fac = fac_data.loc[date]

        try:
            x = c_fac.iloc[:,1:].to_numpy()
            x = np.nan_to_num(x, nan=0, posinf=0, neginf=0) 
            x_std = (x - x.mean(axis=0)) / (x.std(axis=0))
            corr = (x_std.T @ x_std) / (x_std.shape[0] - 1)
        except Exception as e:
            print(f"[出错跳过] {date}，错误信息：{e}")
            continue
        
        # 如果corr全为NaN，则跳过
        if np.isnan(corr).all():
            print(f"[跳过空值] {date}")
            continue
        np.save(save_path, corr.astype(np.float32))
get_corr(fac_data=fac_data, save_dir=rf'/project/model_share/share_1_new/factor_csy/fac20250613_new')

# 读取存储的相关性矩阵
load_dir = r'/project/model_share/share_1_new/factor_csy/fac20250613_new'
npy_files = sorted([f for f in os.listdir(load_dir) if f.endswith("_corr.npy")])
corr_dict = {}
for fname in tqdm(npy_files):
    try:
        date = fname.replace("_corr.npy", "")
        file_path = os.path.join(load_dir, fname)
        corr = np.load(file_path)
        corr_dict[date] = corr
    except Exception as e:
        print(f"[读取失败] {fname} 错误：{e}")
        continue




# 分season计算因子相关性，并进行层次聚类
season_extreme_fac_dict = dict()
season_cluster_dict = dict()
season_rankic_dict = dict()
season_base_dict = dict()


for season in tqdm(season_list):
    _, eval_start, eval_end = get_eval_date(date_list, season, month=24)


    # 因子表现部分
    fac_info = fac_info_all.loc[fac_info_all['date'].between(eval_start, eval_end)]
    # 按照因子进行分组-as_index控制不把分组列作为索引
    fac_info = fac_info.groupby('fac_name', as_index=False)
    
    # 计算各项指标在前两年的均值
    fac_info_mean = []
    for fac_name in fac_name_all:
        res = fac_info.get_group(fac_name)
        res = res.loc[:, 'ic':].mean()
        res['fac_name'] = fac_name
        fac_info_mean.append(res)
    fac_info_mean = pd.concat(fac_info_mean, axis=1).T.set_index('fac_name', drop=True)
    # 调整多空方向
    fac_info_mean = adjust_sign(fac_info_mean)

    # 筛选出正向和负向极端因子——有效极端信息需要较高的IC要求，初步设置为+-0.05
    pos_extreme_facs = fac_info_mean[fac_info_mean['rank_ic'] > 0.05].index.tolist()
    neg_extreme_facs = fac_info_mean[fac_info_mean['rank_ic'] < -0.05].index.tolist()
    season_extreme_fac_dict[season] = {
        'pos': pos_extreme_facs,
        'neg': neg_extreme_facs
    }

    # 聚类部分
    season_corr_list = [corr_dict[date] for date in sorted(corr_dict) if eval_start <= date <= eval_end]
    # 计算平均相关性矩阵
    season_corr = np.stack(season_corr_list)
    corr_mean = np.nanmean(season_corr, axis=0)
    corr_mean = np.nan_to_num(corr_mean, nan=0.0)
    corr_mean = np.clip(corr_mean, -1, 1)
    # 转换为距离矩阵（1 - 相关性）
    dist_matrix = 1 - np.abs(corr_mean)
    dist_vector = squareform(dist_matrix, checks=False)
    # 层次聚类
    Z = linkage(dist_vector, method='average')  
    # 按照相关性距离进行聚类——簇内最低相关性为0.8
    threshold = 0.2
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')


    # 根据因子表现，筛选每一个簇中IC最高的和头尾收益差值最大的
    fac_info_mean['cluster'] = fac_info_mean.index.map(dict(zip(fac_name_all, cluster_labels)))
    fac_info_mean['head_tail_strength'] = fac_info_mean[['head1.5e9', 'tail1.5e9']].abs().max(axis=1)
    # rank_ic取绝对值用于后续评估，防止出现负向因子被错误剔除
    fac_info_mean['rank_ic'] = fac_info_mean['rank_ic'].abs()
    fac_info_mean = fac_info_mean.replace([np.inf, -np.inf, np.nan], 0)
    fac_info_mean = fac_info_mean.reset_index(drop=False)

    # 强簇中删除IC小于0.01的因子
    # cluster_dict = dict()
    # rankic_dict = dict()
    # for cid, group in fac_info_mean.groupby('cluster'):
    #     strong_facs = group[group['rank_ic'] > 0.01]['fac_name'].tolist()
    #     if strong_facs:
    #         cluster_dict[cid] = strong_facs
    #         rankic_dict[cid] = group.loc[group['rank_ic'] > 0.01, 'rank_ic'].tolist()
    #     else:
    #         # 全部因子纳入，并在 cluster key 上加标识（如 cid = '3_weak'）
    #         cluster_dict[f'{cid}_weak'] = group['fac_name'].tolist()
    #         rankic_dict[f'{cid}_weak'] = group['rank_ic'].tolist()

    # 保留所有因子进行IC加权聚合
    cluster_dict = dict()
    rankic_dict = dict()
    base_dict = dict()
    for cid, group in fac_info_mean.groupby('cluster'):
        # 判断是否为强簇（有IC大于0.01的因子）
        is_strong = (group['rank_ic'] > 0.01).any()
        fac_list = group['fac_name'].tolist()
        ic_list = group['rank_ic'].tolist()  # 已取绝对值
        base_list = group['head_tail_strength'].tolist()  

        key = cid if is_strong else f'{cid}_weak'
        cluster_dict[key] = fac_list
        rankic_dict[key] = ic_list
        base_dict[key] = base_list
    season_cluster_dict[season] = cluster_dict
    season_rankic_dict[season] = rankic_dict
    season_base_dict[season] = base_dict
    

for season, cluster_map in season_cluster_dict.items():
    total_clusters = len(cluster_map)
    weak_clusters = sum(1 for cid in cluster_map if isinstance(cid, str) and cid.endswith('_weak'))
    print(f"{season}: 总簇数 = {total_clusters}，其中 weak 簇数 = {weak_clusters}")



    





























from sklearn.decomposition import PCA


# 多进程标准化
def normed(df):
    df = np.clip(df, df.quantile(0.005), df.quantile(0.995), axis=1)
    df = (df - df.mean()) / df.std()
    df = df.fillna(0)
    return df
class stdfacDataset(Dataset):
    def __init__(self, date_list, fac, cluster_dict, rankic_dict, base_dict, pos_extreme_facs, neg_extreme_facs):
        self.date_list = date_list
        self.fac = fac
        self.cluster_dict = cluster_dict
        self.rankic_dict = rankic_dict
        self.base_dict = base_dict
        self.pos_extreme_facs = pos_extreme_facs
        self.neg_extreme_facs = neg_extreme_facs
        
    def __len__(self):
        return len(self.date_list)

    def aggregate_factors(df, facs, cid, method='rankic', top_n=None):
        if method == 'equal':
            weights = np.ones(len(facs)) / len(facs)
        elif method == 'rankic':
            weights = np.array(rankic_dict[cid])
            weights = weights / weights.sum()
        elif method == 'base':
            weights = np.array(base_dict[cid])
            weights = weights / weights.sum()
        elif method == 'topn':
            ic_scores = np.array(rankic_dict[cid])
            top_indices = np.argsort(-np.abs(ic_scores))[:top_n]
            facs = [facs[i] for i in top_indices]
            weights = ic_scores[top_indices]
            weights = weights / weights.sum()
        elif method == 'pca':
            pca = PCA(n_components=1)
            pca_agg = pca.fit_transform(df[facs].fillna(0).values).flatten()
            return pd.Series(pca_agg, index=df.index)
        
        return (df[facs] * weights).sum(axis=1)
    
    def __getitem__(self, idx):
        date = self.date_list[idx]
        # 加载当日数据
        df = self.fac.loc[date].set_index('Code', drop=True).sort_index()

        # 1. 计算极端强度指标(额外信息的引入)
        pos_df = df[self.pos_extreme_facs]
        neg_df = df[self.neg_extreme_facs]

        pos_df = (pos_df - pos_df.mean()) / pos_df.std()
        neg_df = (neg_df - neg_df.mean()) / neg_df.std()  

        pos_extreme_strength = (
            (pos_df - 2).clip(lower=0).abs().sum(axis=1) +      # 正向IC的极大值部分
            (-2 - neg_df).clip(lower=0).abs().sum(axis=1)        # 负向IC的极小值部分
        )
        neg_extreme_strength = (
            (neg_df - 2).clip(lower=0).abs().sum(axis=1) +      # 负向IC的极大值部分
            (-2 - pos_df).clip(lower=0).abs().sum(axis=1)        # 正向IC的极小值部分
        )

        # 2. 聚合因子
        fac_dict = {}
        for cid, facs in self.cluster_dict.items():
            fac_name = f'cluster_{cid}_agg'
            fac_dict[fac_name] = aggregate_factors(df, facs, cid, method='rankic')

            # weights1 = np.array(self.rankic_dict[cid])
            # weights1 = weights1 / weights1.sum()
            # weights2 = np.array(self.base_dict[cid])
            # weights2 = weights2 / weights2.sum()
            # fac_name = f'cluster_{cid}_agg'
            # fac_dict[fac_name] = (df[facs] * weights1).sum(axis=1)
        
        # 3. 按指定顺序构造 sel_df：强簇 + 强度 + 弱簇
        sel_df = pd.DataFrame(index=df.index)

        # 强簇在前
        for fname in sorted(fac_dict):
            if not fname.endswith("_weak_agg"):
                sel_df[fname] = fac_dict[fname]

        # 极端强度居中
        sel_df["pos_extreme_strength"] = pos_extreme_strength.values
        sel_df["neg_extreme_strength"] = neg_extreme_strength.values

        # 弱簇在后
        for fname in sorted(fac_dict):
            if fname.endswith("_weak_agg"):
                sel_df[fname] = fac_dict[fname]
        
        # 强度指标可以一起标准化
        sel_df = normed(sel_df)
        sel_df = sel_df.reset_index()
        sel_df['date'] = date
        return sel_df
def std_collate(batch):
    """合并多个Series为DataFrame"""
    return pd.concat(batch, axis=0)

new_fac_name = rf'threshold_{threshold}'
new_data_path = rf'/project/model_share/share_1_new/factor_csy/fac20250613_new'
os.makedirs(rf'{new_data_path}/{new_fac_name}', exist_ok=True)
for season in season_list:
    pos_extreme_facs = season_extreme_fac_dict[season]['pos']
    neg_extreme_facs = season_extreme_fac_dict[season]['neg']
    cluster_dict = season_cluster_dict[season]
    rankic_dict = season_rankic_dict[season]
    base_dict = season_base_dict[season]
    # 提前进行标准化以缩短训练时间
    get_sfac = DataLoader(
        stdfacDataset(
            date_list=date_list, 
            fac=fac_data, 
            cluster_dict=cluster_dict, 
            rankic_dict=rankic_dict, 
            base_dict=base_dict,
            pos_extreme_facs=pos_extreme_facs, 
            neg_extreme_facs=neg_extreme_facs
            ),
        batch_size=64,
        shuffle=False,           # 保持日期顺序
        num_workers=128,     
        collate_fn=std_collate,    
        persistent_workers=True 
    )
    fea = pd.concat(res for res in tqdm(get_sfac))
    fea = fea.reset_index(drop=True)

    strong_cnt = sum(1 for k in cluster_dict.keys() if not str(k).endswith('_weak'))
    weak_cnt = sum(1 for k in cluster_dict.keys() if str(k).endswith('_weak'))
    file_name = f"{season}_strong{strong_cnt}_weak{weak_cnt}_all{strong_cnt + 2 + weak_cnt}.fea"

    fea.to_feather(
        rf'{new_data_path}/{new_fac_name}/{file_name}'
        )