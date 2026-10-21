import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# 数据存储目录
data_path = rf'/project/model_share/share_1'
fac_path = rf'{data_path}/factor_data'
fac_name = rf'fac20250212'
label_path = rf'{data_path}/label_data'
label_name = rf'label1'
liquid_path = rf'{data_path}/label_data'
liquid_name = rf'can_trade_amt1'

# 读取完整因子集和其他数据
fac_data = pd.read_feather(rf'{fac_path}/{fac_name}/{fac_name}.fea')
liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
season_list = ["2023q1", "2023q2", "2023q3", "2023q4", "2024q1", "2024q2"]
date_list = [x for x in fac_data['date'].unique() if x in ret_data.index and x in liquid_data.index]
date_list.sort()
fac_data = fac_data.set_index('date').sort_index()


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
cal_res = DataLoader(FacMetric(fac_data, date_list), collate_fn=lambda x: x, num_workers=64)
fac_info_all = pd.concat([res[0] for res in tqdm(cal_res)])
fac_info_all = fac_info_all.reset_index(drop=False).rename(columns={'index': 'fac_name'})
fac_name_all = fac_info_all['fac_name'].unique()


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


# 获取每个季度的因子列表
# sel_fac_season中记录的因子列表即为每个季度的筛选结果
sel_fac_season = dict()
for season in season_list:
    # 取前两年的数据
    _, eval_start, eval_end = get_eval_date(date_list, season, 24)
    fac_info = fac_info_all.loc[fac_info_all['date'].between(eval_start, eval_end)]
    fac_info = fac_info.sort_values(['fac_name', 'date']).groupby('fac_name', as_index=False)
    # 计算各项指标在前两年的均值
    fac_info_mean = []
    for fac_name in fac_name_all:
        res = fac_info.get_group(fac_name).loc[:, 'ic':].mean()
        res['fac_name'] = fac_name
        fac_info_mean.append(res)
    fac_info_mean = pd.concat(fac_info_mean, axis=1).T.set_index('fac_name', drop=True)
    # 筛选条件：head1.5e9足够大（多头足够强）或tail1.5e9足够小（空头足够强）
    fac_info_mean = adjust_sign(fac_info_mean)
    sel_fac_long = fac_info_mean[fac_info_mean['head1.5e9'] > 3 / 10000].index.to_list()
    sel_fac_short = fac_info_mean[fac_info_mean['tail1.5e9'] < - 4 / 10000].index.to_list()
    sel_fac_list = sel_fac_long + [x for x in sel_fac_short if x not in sel_fac_long]
    sel_fac_season[season] = sel_fac_list[:]
