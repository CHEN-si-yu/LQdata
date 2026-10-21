import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# 数据存储目录
data_path = rf'/project/model_share/share_1_new'
fac_path = rf'{data_path}/factor_data'
fac_name = rf'fac20250212'
label_path = rf'{data_path}/label_data'
label_name = rf'label1'
liquid_path = rf'{data_path}/label_data'
liquid_name = rf'can_trade_amt1'

#     low = data_X.quantile(0.005)
#     up = data_X.quantile(0.995)
#     data_X = data_X.where(data_X >= low, low, axis=1)
#     data_X = data_X.where(data_X <= up, up, axis=1)

#     data_X = (data_X - data_X.mean()) / data_X.std()
#     data_X = data_X.fillna(0)

# path_list = os.listdir(rf'{fac_path}/{fac_name}')
# for i in tqdm(path_list):
#     data = pd.read_feather(rf'{fac_path}/{fac_name}/{i}')
#     data = data.set_index('date')
#     for date in tqdm(data.index.unique()):
#         cdata = data.loc[date]
#         cdata = cdata.drop('Code',axis=1)
        
#         low = cdata.quantile(0.005)
#         up = cdata.quantile(0.995)
#         cdata = cdata.where(cdata >= low, low, axis=1)
#         cdata = cdata.where(cdata <= up, up, axis=1)
#         cdata = (cdata - cdata.mean()) / cdata.std()
#         cdata = cdata.fillna(0)

#         data.loc[date,1:] = cdata
#     data = data.reset_index()
#     os.makedirs(rf'{fac_path}/{fac_name}_ts', exist_ok=True)
#     data.to_feather(rf'{fac_path}/{fac_name}_ts/{i}')

# 读取完整因子集和其他数据
fac_data = pd.read_feather(rf'{fac_path}/{fac_name}/{fac_name}.fea')
liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
# barra_data = pd.read_feather(rf'{fac_path}/barra_fac.fea')
season_list = ["2023q1", "2023q2", "2023q3", "2023q4", "2024q1", "2024q2"]
date_list = [x for x in fac_data['date'].unique() if x in ret_data.index and x in liquid_data.index]
date_list.sort()
fac_data = fac_data.sort_values(by=['date', 'Code']).set_index('date')
# barra_data = barra_data.sort_values(by=['date', 'Code']).set_index('date')

# 更新因子数据
# other_fac = os.listdir(rf'/project/model_share/other_data/other_feature/Static.a1674')
# append_date = []
# for file in other_fac:
#     date, ext = os.path.splitext(file)
#     append_date.append(date)
# append_date = [x for x in append_date if x not in fac_data.index]
# for date in append_date:
#     cdata = pd.read_feather(rf'/project/model_share/other_data/other_feature/Static.a1674/20240701.fea')

# 多进程计算每日因子的回测表现
class FacMetric(Dataset):

    def __init__(self, data, date):
        self.fac_data = data
        self.date_list = date

    def __getitem__(self, index):
        # 取date日的因子，计算表现
        date = self.date_list[index]
        fac_td = self.fac_data.loc[date].set_index('Code')
        fac_rank = fac_td.rank(pct=True, method='dense')
        ret = ret_data.loc[date].dropna()
        ret_rank = ret.rank(pct=True, method='dense')
        amt = liquid_data.loc[date].dropna()
        amt_ret = pd.concat([amt, ret], axis=1, keys=['amt', 'ret']).fillna(0)
        amt_ret['amt_ret'] = amt_ret['amt'] * amt_ret['ret']
        fac_list = fac_td.columns.tolist()
        fac_info = pd.DataFrame(index=fac_list)

        # ir和rank_ic
        fac_info['ic'] = fac_td.corrwith(ret)
        fac_info['rank_ic'] = fac_rank.corrwith(ret_rank)

        # 打分头/尾10%的收益率
        head10p = (fac_rank > 0.9).T
        tail10p = (fac_rank < 0.1).T
        fac_info['head10p'] = head10p.dot(ret.reindex(fac_rank.index).fillna(0)) / head10p.sum(axis=1)
        fac_info['tail10p'] = tail10p.dot(ret.reindex(fac_rank.index).fillna(0)) / tail10p.sum(axis=1)

        # 打分头/尾1.5e9金额按流动性买入的收益（和模型端label_ret的计算逻辑一致）
        def htamt_ret(code_list, tot_amt):
            amt_ret_new = amt_ret.reindex(code_list).fillna(0)
            amt_ret_new['cum_amt'] = amt_ret_new['amt'].cumsum()
            # 筛选累计资金不超过规划资金的行，确保不会超出预算
            amt_ret_ht = amt_ret_new.loc[amt_ret_new['cum_amt'] <= tot_amt]
            return amt_ret_ht['amt_ret'].sum() / amt_ret_ht['amt'].sum()

        money = '1.5e9'
        htamt = {}
        for fac_name in fac_list:
            htamt[fac_name] = {}
            # 获取某个因子的各个股票的排序（降序取头。升序取尾）
            # 从因子值大的开始买和从因子值小的开始买，所以是因子的头部收益和尾部收益
            code_head = fac_td[fac_name].sort_values(ascending=False).dropna().index.tolist()
            code_tail = fac_td[fac_name].sort_values(ascending=True).dropna().index.tolist()
            # eval用于将字符串转变为float
            htamt[fac_name][f'head{money}'] = htamt_ret(code_head, eval(money))
            htamt[fac_name][f'tail{money}'] = htamt_ret(code_tail, eval(money))

        fac_info = pd.concat([fac_info, pd.DataFrame(htamt).T], axis=1)
        fac_info.insert(0, 'date', date)
        return fac_info

    def __len__(self):
        return len(self.date_list)


# 合并每日因子表现至fac_info_all
cal_res = DataLoader(FacMetric(fac_data, date_list), collate_fn=lambda x: x, num_workers=64)
fac_info_all = pd.concat([res[0] for res in tqdm(cal_res)])
fac_info_all = fac_info_all.reset_index(drop=False).rename(columns={'index': 'fac_name'})
fac_name_all = fac_info_all['fac_name'].unique()

# 读取barra风险因子
# barra_filename = os.listdir(rf'/project/model_share/other_data/barra_factors')
# barra_fac_list = []
# for file in barra_filename:
#     fac_name = file[:-4]
#     barra_fac = pd.read_feather(rf'/project/model_share/other_data/barra_factors/{file}')
#     barra_fac = barra_fac.set_index('index').stack().rename(fac_name)
#     barra_fac_list.append(barra_fac)
# barra_fac = pd.concat(barra_fac_list, axis=1)
# barra_fac.index.names = ['date', 'Code']
# barra_fac = barra_fac.reset_index().sort_values(by = ['date', 'Code'])
# barra_fac = barra_fac.reset_index(drop=True)
# barra_fac.to_feather(rf'{data_path}/factor_data/barra_fac.fea')

# 对于season季度的测试集，在季度开始前（test_start前）获取前month个月的日期列表，用这些日期来评测因子
def get_eval_date(all_date, season, month):
    test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2)
    start_date = datetime.strptime(test_start, "%Y%m")
    train_start = (start_date - relativedelta(months=month)).strftime("%Y%m")
    # 隔开10天防止泄露未来数据（同模型训练）
    train_date_list = [x for x in all_date if train_start <= x < test_start][:-10]
    # 不考虑极端日期（同模型训练）
    not_train_date = [x for x in date_list if (x >= "202402") & (x <= "20240223")]
    train_date_list = [x for x in train_date_list if x not in not_train_date]
    train_date_list.sort()
    return train_date_list, train_date_list[0], train_date_list[-1]


# 按ic调整多空头收益
# head是因子大的组，tail是因子小的组
# 如果ic大于0，认为head是多头组，tail是空头组
# 如果ic小于0，认为head是空头组，tail是多头组
# 经过这一调整以后默认多头收益为正，空头收益为负
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

# 计算每日各个因子与标的之间的互相关系数MI
# import torch
# from sklearn.feature_selection import mutual_info_regression

# class MIDataset(Dataset):
#     def __init__(self, date_list, fac, ret):
#         """
#         date_list: 需要处理的日期列表
#         """
#         self.date_list = date_list
#         self.fac = fac
#         self.ret = ret
        
#     def __len__(self):
#         return len(self.date_list)
    
#     def __getitem__(self, idx):
#         date = self.date_list[idx]
        
#         # 加载当日数据
#         fac_data = self.fac.loc[date].set_index('Code', drop=True).sort_index()
#         ret_data = self.ret.loc[date].T.sort_index()
        
#         fac_data = fac_data.dropna(how='all').fillna(0)
#         ret_data = ret_data.reindex(fac_data.index).fillna(0)
        
#         # 计算互信息
#         mi = mutual_info_regression(
#             fac_data.values,
#             ret_data.values,
#             n_neighbors=3,
#             random_state=3253
#         )
#         return pd.Series(mi, index=fac_data.columns, name=date)

# def collate_fn(batch):
#     """合并多个Series为DataFrame"""
#     return pd.concat(batch, axis=1).T

# cal_mi = DataLoader(
#     MIDataset(
#         date_list = date_list,  
#         fac = fac_data,         
#         ret = ret_data     
#     ),
#     batch_size=1,            # 每个batch处理一个日期
#     shuffle=False,           # 保持日期顺序
#     num_workers=64,     
#     collate_fn=collate_fn,    
#     persistent_workers=True 
# )

# mi_df = pd.concat(res for res in tqdm(cal_mi))
# mi_df = mi_df.stack().reset_index()
# mi_df.columns = ['date','fac_name','mi']
# fac_info_all = pd.merge(fac_info_all, mi_df, on=['date','fac_name'], how='left')


# # 使用机器学习的方法直接筛选因子
# import lightgbm as lgb

# class lgbmDataset(Dataset):
#     def __init__(self, date_list, fac, ret):
#         """
#         date_list: 需要处理的日期列表
#         """
#         self.date_list = date_list
#         self.fac = fac
#         self.ret = ret
#         self.model_params = {
#             'objective': 'regression',
#             'metric': 'rmse',
#             'num_leaves': 127,
#             'learning_rate': 0.01,
#             'feature_fraction': 0.6,
#             'bagging_freq': 5,
#             'verbosity': -1,
#             'seed': 3253
#         }

#     def __len__(self):
#         return len(self.date_list)
    
#     def __getitem__(self, idx):
#         date = self.date_list[idx]
        
#         # 加载当日数据
#         fac_data = self.fac.loc[date].set_index('Code', drop=True).sort_index()
#         ret_data = self.ret.loc[date]
        
#         fac_data = fac_data.dropna(how='all').fillna(0)
#         ret_data = ret_data.reindex(fac_data.index).fillna(0)
        
#         # 划分训练集（单日数据不划分验证集）
#         train_data = lgb.Dataset(fac_data, label=ret_data)
        
#         # 训练模型
#         model = lgb.train(
#             self.model_params,
#             train_data,
#             num_boost_round=100)
        
#         # 获取因子重要性
#         importance = pd.Series(
#             model.feature_importance(importance_type='gain'),
#             index=fac_data.columns,
#             name=date
#         )
        
#         # 归一化处理
#         return importance / importance.sum()


# def collate_fn(batch):
#     """合并每日重要性为DataFrame"""
#     valid_batch = [b for b in batch if not b.empty]
#     return pd.concat(valid_batch, axis=1).T

# cal_importance = DataLoader(
#     lgbmDataset(
#         date_list = date_list,  
#         fac = fac_data,         
#         ret = ret_data     
#     ),
#     batch_size=1,            # 每个batch处理一个日期
#     shuffle=False,           # 保持日期顺序
#     num_workers=64,     
#     collate_fn=collate_fn,    
#     persistent_workers=True 
# )

# lgbm_df = pd.concat(res for res in tqdm(cal_importance))
# lgbm_df = lgbm_df.stack().reset_index()
# lgbm_df.columns = ['date','fac_name','importance']
# fac_info_all = pd.merge(fac_info_all, lgbm_df, on=['date','fac_name'], how='left')


# # 在机器学习方法中集成SHAP
# import lightgbm as lgb
# import shap

# class shap_lgbmDataset(Dataset):
#     def __init__(self, date_list, fac, ret):
#         """
#         date_list: 需要处理的日期列表
#         """
#         self.date_list = date_list
#         self.fac = fac
#         self.ret = ret
#         self.model_params = {
#             'objective': 'regression',
#             'metric': 'rmse',
#             'num_leaves': 127,
#             'learning_rate': 0.01,
#             'feature_fraction': 0.6,
#             'bagging_freq': 5,
#             'verbosity': -1,
#             'seed': 3253
#         }

#     def __len__(self):
#         return len(self.date_list)
    
#     def __getitem__(self, idx):
#         date = self.date_list[idx]
        
#         # 加载当日数据
#         fac_data = self.fac.loc[date].set_index('Code', drop=True).sort_index()
#         ret_data = self.ret.loc[date]
        
#         fac_data = fac_data.dropna(how='all').fillna(0)
#         ret_data = ret_data.reindex(fac_data.index).fillna(0)
        
#         train_data = lgb.Dataset(fac_data, label=ret_data)
#         model = lgb.train(
#             self.model_params,
#             train_data,
#             num_boost_round=100)
        
#         # 因子重要性
#         importance = pd.Series(
#             model.feature_importance(importance_type='gain'),
#             index=fac_data.columns,
#             name=date)
        
#         # 计算SHAP值（优化计算量）
#         sample_size = min(500, len(fac_data))  # 控制计算样本量
#         sample_idx = np.random.choice(fac_data.index, size=sample_size, replace=False)
#         explainer = shap.TreeExplainer(model)
#         shap_values = explainer.shap_values(fac_data.loc[sample_idx])
        
#         # 计算方向一致性
#         direction_consistency = np.mean(np.sign(shap_values), axis=0)
        
#         # 返回复合结果
#         return ( 
#             importance,
#             pd.Series(np.abs(shap_values).mean(axis=0), 
#                         index=fac_data.columns, 
#                         name = date), 
#             pd.Series(np.abs(direction_consistency), 
#                         index=fac_data.columns, 
#                         name = date)            
#                          )

# cal_shap = DataLoader(
#     shap_lgbmDataset(date_list, fac_data, ret_data),
#     batch_size=1,
#     collate_fn=lambda x:x[0],
#     num_workers=64
# )

# res = [b for b in tqdm(cal_shap)]
# import_df = pd.concat([x[0] for x in res], axis=1).T
# shap_df = pd.concat([x[1] for x in res], axis=1).T
# direct_df = pd.concat([x[2] for x in res], axis=1).T

# import_df = import_df.stack().reset_index()
# import_df.columns = ['date','fac_name','importance']
# shap_df = shap_df.stack().reset_index()
# shap_df.columns = ['date','fac_name','shap_abs']
# direct_df = direct_df.stack().reset_index()
# direct_df.columns = ['date','fac_name','shap_direct']

# fac_info_all = pd.merge(fac_info_all, import_df, on=['date','fac_name'], how='left')
# fac_info_all = pd.merge(fac_info_all, shap_df, on=['date','fac_name'], how='left')
# fac_info_all = pd.merge(fac_info_all, direct_df, on=['date','fac_name'], how='left')

# 获取每个季度的因子列表
# sel_fac_season中记录的因子列表即为每个季度的筛选结果
sel_fac_season = dict()
for season in tqdm(season_list):
    # 取前两年的数据
    _, eval_start, eval_end = get_eval_date(date_list, season, 24)
    fac_info = fac_info_all.loc[fac_info_all['date'].between(eval_start, eval_end)]
    # 按照因子进行分组-as_index控制不把分组列作为索引
    fac_info = fac_info.sort_values(['fac_name', 'date']).groupby('fac_name', as_index=False)
    
    # 计算各项指标在前两年的均值
    fac_info_mean = []
    for fac_name in fac_name_all:
        res = fac_info.get_group(fac_name)
        
        # 补充1：计算多空收益的标准差用于计算夏普比率
        # head_std = res['head1.5e9'].std()
        # tail_std = res['tail1.5e9'].std()

        # 补充2：计算正向因子占比
        # def cal_direct(res):
        #     if len(res['rank_ic'].dropna()) == 0:
        #         direct = np.nan
        #     else:
        #         direct = len(res.loc[res['rank_ic']>0, 'rank_ic']) / len(res['rank_ic'].dropna())
        
        # # 补充3：因子稳定性——信息系数IR
        # res['ir'] = res['rank_ic'].rolling(window=60, min_periods=5).apply(lambda x:x.mean() / x.std())

        # res = res.loc[:, 'ic':].mean()
        res = res.loc[:, 'ic':].ewm(span=252, adjust=False).mean().iloc[-1,:]
        res['fac_name'] = fac_name
        
        # res['head_sharpe'] = ((res['head1.5e9'] - res['base_r']) / head_std) * np.sqrt(252)
        # res['tail_sharpe'] = ((res['tail1.5e9'] - res['base_r']) / tail_std) * np.sqrt(252)
        # res['head_sharpe'] = ((res['head1.5e9']) / head_std) * np.sqrt(252)
        # res['tail_sharpe'] = ((res['tail1.5e9']) / tail_std) * np.sqrt(252)
        # res['direct'] = direct
        # res['ir'] = ir
        
        fac_info_mean.append(res)
    fac_info_mean = pd.concat(fac_info_mean, axis=1).T.set_index('fac_name', drop=True)

    # 调整多空方向
    fac_info_mean = adjust_sign(fac_info_mean)
    
    # # 注意调整后需要对尾段夏普取负值以得到真正的夏普
    # fac_info_mean['tail_sharpe'] = -1 * fac_info_mean['tail_sharpe']

    # 筛选条件——base_绝对收益足够：head1.5e9足够大（多头足够强）或tail1.5e9足够小（空头足够强）
    sel_fac_long = fac_info_mean[fac_info_mean['head1.5e9'] > 3 / 10000].index.to_list()
    sel_fac_short = fac_info_mean[fac_info_mean['tail1.5e9'] < - 4 / 10000].index.to_list()
    sel_fac_absr = sel_fac_long + [x for x in sel_fac_short if x not in sel_fac_long]

    # sel_fac_long = fac_info_mean[fac_info_mean['head10p'] > 3 / 10000].index.to_list()
    # sel_fac_short = fac_info_mean[fac_info_mean['tail10p'] < - 4 / 10000].index.to_list()
    # sel_fac_absr = sel_fac_long + [x for x in sel_fac_short if x not in sel_fac_long]
    # 筛选条件——rankic(仅保留绝对值0.02以上的因子)
    fac_info_mean = fac_info_mean.loc[sel_fac_absr]
    icp = fac_info_mean[fac_info_mean['rank_ic'] >= 0.01].index.to_list()
    icn = fac_info_mean[fac_info_mean['rank_ic'] < -0.01].index.to_list()
    sel_fac_absr_ic = icp + [x for x in icn if x not in icp]

    # 筛选条件——rankic同向比例（正向比例大于0.7或小于0.3）
    # fac_info_mean = fac_info_mean.loc[sel_fac_absr]
    # directp = fac_info_mean[fac_info_mean['direct'] >= 0.6].index.to_list()
    # directn = fac_info_mean[fac_info_mean['direct'] <= 0.4].index.to_list()
    # sel_fac_absr_direct = directp + [x for x in directn if x not in directp]

    # # 筛选条件——相对夏普（market_base使用每日所有因子的均值作为代替）
    # # 多头夏普高于1. 空头夏普高于2
    # fac_info_mean = fac_info_mean.loc[sel_fac_absr]
    # long_sharpe = fac_info_mean[fac_info_mean['head_sharpe'] > 2].index.to_list()
    # short_sharpe = fac_info_mean[fac_info_mean['tail_sharpe'] > 4].index.to_list()
    # sel_fac_absr_sharpe = long_sharpe + [x for x in short_sharpe if x not in long_sharpe]
    
    # 筛选条件——互信息mi（仅保留互信息高于0.04的部分）
    # fac_info_mean = fac_info_mean.loc[sel_fac_absr]
    # sel_fac_absr_mi = fac_info_mean[fac_info_mean['mi'] >= 0.02].index.to_list()
    
    # 筛选条件——树模型因子重要性(保留重要性前50%的因子)
    # fac_info_mean = fac_info_mean.loc[sel_fac_absr]
    # sel_fac_lgbm = fac_info_mean[(fac_info_mean['importance'] >= fac_info_mean['importance'].quantile(0.5)) 
    #                              &
    #                              (fac_info_mean['importance'] > 0)].index.to_list()
    # fac_info_mean = fac_info_mean.loc[sel_fac_lgbm]
    # sel_fac_long = fac_info_mean[fac_info_mean['head1.5e9'] > 3 / 10000].index.to_list()
    # sel_fac_short = fac_info_mean[fac_info_mean['tail1.5e9'] < - 4 / 10000].index.to_list()
    # sel_fac_lgbm_absr = sel_fac_long + [x for x in sel_fac_short if x not in sel_fac_long]
    
    # # 集成思路——通过打分制定综合评价指标
    # fac_info_mean = fac_info_mean.drop(columns='base_r')
    # fac_info_mean = fac_info_mean.apply(lambda x:(x-x.mean()) / x.std())
    # fac_score = fac_info_mean.mean(axis=1)
    
    sel_fac_season[season] = sel_fac_absr_ic
for key in sel_fac_season.keys():
    print(len(sel_fac_season[key]))

def normed(df):
    df = np.clip(df, df.quantile(0.005), df.quantile(0.995), axis=1)
    df = (df - df.mean()) / df.std()
    df = df.fillna(0)
    return df

class sfacDataset(Dataset):
    def __init__(self, date_list, fac):
        """
        date_list: 需要处理的日期列表
        """
        self.date_list = date_list
        self.fac = fac
        
    def __len__(self):
        return len(self.date_list)
    
    def __getitem__(self, idx):
        date = self.date_list[idx]
        # 加载当日数据
        fac_data = self.fac.loc[date].set_index('Code', drop=True).sort_index()
        fac_data = normed(fac_data)
        fac_data = fac_data.reset_index()
        fac_data['date'] = date
        return fac_data

def collate_fn(batch):
    """合并多个Series为DataFrame"""
    return pd.concat(batch, axis=0)

# 存储因子数据
import os 
fac_name = rf'fac20250212_ewm252_base_ic'
os.makedirs(rf'{data_path}/factor_data/{fac_name}', exist_ok=True)
for key, value in sel_fac_season.items():
    # 提前进行标准化以缩短训练时间
    cdata = fac_data[['Code']+value]
    get_sfac = DataLoader(
        sfacDataset(
            date_list = date_list,  
            fac = cdata,            
        ),
        batch_size=1,            # 每个batch处理一个日期
        shuffle=False,           # 保持日期顺序
        num_workers=64,     
        collate_fn=collate_fn,    
        persistent_workers=True 
    )
    fea = pd.concat(res for res in tqdm(get_sfac))
    fea = fea.reset_index(drop=True)
    fea.to_feather(
        rf'{data_path}/factor_data/{fac_name}/{key}_num-{len(value)}.fea'
        )