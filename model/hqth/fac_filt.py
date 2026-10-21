import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# 数据存储目录
fac_name = rf'data_float'
label_path = rf'/home/user96/label_data1'
label_name = rf'label1'
liquid_path = rf'/home/user96/label_data1'
liquid_name = rf'can_trade_amt1'

# 读取完整因子集和其他数据
fac_data = pd.read_feather(rf'/project/model_share/share_1_new/factor_csy/int_or_float/data_float.fea')
liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
season_list = ["2023q1", "2023q2", "2023q3", "2023q4", "2024q1", "2024q2", "2024q3", "2024q4"]
date_list = [x for x in fac_data['date'].unique() if x in ret_data.index and x in liquid_data.index]
date_list.sort()
fac_data = fac_data.sort_values(by = ['date','Code']).set_index('date')



# 多进程计算每日因子的回测表现
class FacMetric(Dataset):

    def __init__(self, data, date):
        self.fac_data = data
        self.date_list = date

    def __getitem__(self, index):
        # 取date日的因子，计算表现
        date = self.date_list[index]
        fac_td = self.fac_data.loc[date].set_index('Code')
        fac_rank = fac_td.rank(pct=True, method='average')
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
        fac_info['head10p'] = head10p.dot(ret.reindex(fac_rank.index).fillna(0)) / (head10p.sum(axis=1) + 1e-15) # 2025.6.21 csy
        fac_info['tail10p'] = tail10p.dot(ret.reindex(fac_rank.index).fillna(0)) / (tail10p.sum(axis=1) + 1e-15) # 2025.6.21 csy

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


# 对于season季度的测试集，在季度开始前（test_start前）获取前day的日期列表，用这些日期来评测因子
from datetime import datetime, timedelta
def get_eval_date(all_date, season, day):
    test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2) + "01"
    start_date = datetime.strptime(test_start, "%Y%m%d")

    train_start = (start_date - timedelta(days=day)).strftime("%Y%m%d")
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


# 获取每个季度的因子列表
# sel_fac_season中记录的因子列表即为每个季度的筛选结果
sel_fac_season = dict()
fac_num_target = 1400
for season in tqdm(season_list):
    # 取前两年的数据
    _, eval_start, eval_end = get_eval_date(date_list, season, day=720)
    fac_info = fac_info_all.loc[fac_info_all['date'].between(eval_start, eval_end)]
    # 按照因子进行分组-as_index控制不把分组列作为索引
    fac_info = fac_info.sort_values(['fac_name', 'date']).groupby('fac_name', as_index=False)
    
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
    
    # 第一步：分正负 rank_ic，分别筛选前 50%
    fac_info_pos = fac_info_mean[fac_info_mean['rank_ic'] > 0]
    fac_info_neg = fac_info_mean[fac_info_mean['rank_ic'] < 0]
    # 正向因子中按 rank_ic 从大到小排序，保留前 50%
    fac_info_pos_sorted = fac_info_pos.sort_values('rank_ic', ascending=False)
    fac_info_pos_top = fac_info_pos_sorted.iloc[:len(fac_info_pos_sorted) // 2]
    # 负向因子中按 rank_ic 从小到大排序，保留前 50%
    fac_info_neg_sorted = fac_info_neg.sort_values('rank_ic', ascending=True)
    fac_info_neg_top = fac_info_neg_sorted.iloc[:len(fac_info_neg_sorted) // 2]
    # 合并
    fac_info_half = pd.concat([fac_info_pos_top, fac_info_neg_top])

    # 第二步：判断是否超过目标数量
    if len(fac_info_half) <= fac_num_target:
        sel_factors = fac_info_half.index.tolist()
    else:
        # 第三步：基于收益强度排序（头尾收益的绝对值的最大值）
        fac_info_half['abs_return_strength'] = fac_info_half[['head1.5e9', 'tail1.5e9']].abs().max(axis=1)
        fac_info_final = fac_info_half.sort_values('abs_return_strength', ascending=False).iloc[:fac_num_target]
        sel_factors = fac_info_final.index.tolist()
    
    sel_fac_season[season] = sel_factors

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
        # fac_data = normed(fac_data) 保持model.py不变，取模型训练里面标准化
        fac_data = fac_data.reset_index()
        fac_data['date'] = date
        return fac_data

def collate_fn(batch):
    """合并多个Series为DataFrame"""
    return pd.concat(batch, axis=0)

# 存储因子数据
import os 
fac_name = rf'fac20250708'
os.makedirs(rf'/project/model_share/share_1_new/factor_csy/{fac_name}', exist_ok=True)
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
        rf'/project/model_share/share_1_new/factor_csy/{fac_name}/{key}_num-{len(value)}.fea'
        )