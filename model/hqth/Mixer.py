import gc
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchmetrics import PearsonCorrCoef
from argparse import ArgumentParser
import warnings
import shutil
import copy
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler

warnings.filterwarnings("ignore")

cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.autograd.set_detect_anomaly(True)

'''参数设置要求'''
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)  # batch包含的天数
    parser.add_argument('--weight_decay', type=float, default=2e-2)  # weight_decay
    parser.add_argument('--seed', type=int, default=3253)  # 随机种子
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adam', 'adamw'])
    parser.add_argument('--loss', default='wpcc')  # 损失函数
    parser.add_argument('--lr', type=float, default=0.001)  # 初始学习率

    parser.add_argument('--max_epochs', type=int, default=30)  # 最大轮数
    parser.add_argument('--gpus', default=[0])
    parser.add_argument('--strategy', default='ddp')
    parser.add_argument('--find_unused_parameters', default=False)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--check_test_every_n_epoch', type=int, default=1)

    parser.add_argument('--log_every_n_steps', type=int, default=1)
    parser.add_argument('--early_stop', action='store_true')  # 早停设置
    parser.add_argument('--swa', action='store_true')  # swa设置
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', help='path to checkpoints (for test)')


    args, unknown = parser.parse_known_args()
    return args
args = parse_args()

# 模型文件存储目录
root_path = rf'/home/user96/20250812_trainmodel3'
# 因子
fac_path = rf'/project/model_share/share_1_new/factor_csy/'
fac_name = rf'fac20250613_ic1400'
# 标签
label_path = rf'/home/user96/label_data1/'
label_name = rf'label1'
# 流动性数据
liquid_path = rf'/home/user96/label_data1/'
liquid_name = rf'can_trade_amt1'


class params:
    model_path = rf'{root_path}/model_train'
    profiler_path = rf'{root_path}/logs'
    # 模型前缀
    model_prefix = rf'nn'
    liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
    # 是否使用dropout
    dropout = True
    # dropout率
    dropout_rate = 0.2
    # 因子标准化方法
    normed_method = 'zscore'


def get_basic_name():
    name = rf'{params.model_prefix}--{fac_name}--{label_name}'
    if params.dropout:
        name += rf'--dropout{params.dropout_rate}'
    return name


# 因子标准化
def normed_data(data, date, stage, normed_method=params.normed_method):
    data = data.reset_index()
    data = data.reindex(data[params.factor_list].dropna(thresh=int(0.1 * len(params.factor_list))).index)
    data = data.set_index('date')
    liquid_data = params.liquid_data.loc[date]
    data['liquid'] = liquid_data.reindex(data["Code"]).values
    ret_data = params.ret_data.loc[date]
    data['Label'] = ret_data.reindex(data["Code"]).values
    if stage == "train":
        data['Label'] = (data['Label'] - data['Label'].mean()) / data['Label'].std()
    data['Label'] = data['Label'].fillna(0)
    code_value = data['Code'].values
    data_X = data.drop(['Code', 'Label', 'liquid'], axis=1)
    data_y = data['Label']
    data_liquid = data['liquid']

    if normed_method == 'zscore':
        data_X = np.clip(data_X, data_X.quantile(0.005), data_X.quantile(0.995), axis=1)
        data_X = (data_X - data_X.mean()) / data_X.std()
        data_X = data_X.fillna(0)

    else:
        raise NotImplementedError

    return torch.tensor(data_X.values, dtype=torch.float32), \
        torch.tensor(data_y.values, dtype=torch.float32), \
        code_value, torch.tensor(data_liquid.values, dtype=torch.float32)


def collate_fn(datas):
    data_X = [temp for data in datas for temp in data[0]]
    data_y = [temp for data in datas for temp in data[1]]
    data_time = [temp for data in datas for temp in data[2]]
    code_value = [temp for data in datas for temp in data[3]]
    data_liquid = [temp for data in datas for temp in data[4]]
    return data_X, data_y, data_time, code_value, data_liquid


class DLDataset(torch.utils.data.Dataset):

    def __init__(self, date_list, stage='train'):
        self.date_list = date_list
        self.stage = stage

    def __getitem__(self, index):
        date = self.date_list[index]
        if date == 'out_sample':
            return ['out_sample'], ['out_sample'], ['out_sample'], ['out_sample'], ['out_sample']
        data = params.all_data.loc[date].copy()
        train_X = []
        train_y = []
        train_time = []

        train_code_value = []
        train_liquid = []
        data_X, data_y, code_value, data_liquid = normed_data(data, date, stage=self.stage)
        train_X.append(torch.nan_to_num(data_X, nan=0, posinf=0, neginf=0))
        train_y.append(torch.nan_to_num(data_y, nan=0, posinf=0, neginf=0))
        train_liquid.append(torch.nan_to_num(data_liquid, nan=0, posinf=0, neginf=0))
        train_time.append(date)
        train_code_value.append(code_value)

        return train_X, train_y, train_time, train_code_value, train_liquid

    def __len__(self):
        return len(self.date_list)


class DLDataModule(pl.LightningDataModule):
    def __init__(self, args, train_date_list, valid_date_list, test_date_list):
        super().__init__()
        self.args = args
        self.tr = DLDataset(train_date_list, stage='train')
        self.val = DLDataset(valid_date_list, stage='valid')
        self.test = DLDataset(test_date_list, stage='test')

    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.args.batch_size, collate_fn=collate_fn,
                          num_workers=4, shuffle=True,
                          persistent_workers=False,
                          drop_last=False,
                          pin_memory=False)

    def _val_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                          num_workers=0, persistent_workers=False,
                          pin_memory=False, drop_last=False)

    def val_dataloader(self):
        return self._val_dataloader(self.val)

    def test_dataloader(self):
        return self._val_dataloader(self.test)


'''损失函数设置要求——wpcc必须使用preds进行加权'''
def get_loss_fn(loss):
    def wpcc(preds, y):
        _,argsort = torch.sort(preds, descending=True,dim=0) 
        argsort = argsort.squeeze()
        weight = torch.zeros(preds.shape,device=preds.device)
        weight_new = torch.tensor([0.5**((i-1)/(preds.shape[0]-1)) for i in range(1,preds.shape[0]+1)]
                                  ,device=preds.device).unsqueeze(dim=1)
        weight[argsort,:] = weight_new
        wcov = (preds * y * weight).sum(dim = 0) / weight.sum(dim = 0) - ((preds * weight).sum(dim=0) / weight.sum(dim=0)) * ((y * weight).sum(dim=0) / weight.sum(dim=0))
        pred_std = torch.sqrt(((preds - preds.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        y_std = torch.sqrt(((y - y.mean(dim=0))**2 * weight).sum(dim=0) / weight.sum(dim=0))
        return -(wcov / (pred_std * y_std)).mean()
    def output(loss):
        return {
        'wpcc':wpcc
        }[loss]
    return output(loss)


'''模型设置要求'''
from src.model import *
from kan import KAN
class PredictModel(nn.Module):
    def __init__(self, args, time_steps=1, market=128):
        super(PredictModel, self).__init__()


        input_dim = params.factor_num

        # 输入层
        self.input_layter = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # 隐含层
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 混合层
        channels = 128
        scale_dim = 1
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1)
        self.dropout_conv = nn.Dropout(p=0.2)        # Conv 后
        self.dropout_channel = nn.Dropout(p=0.3)     # channel_fc 后

        # 输出层
        self.output_layer = KAN([128 * 3,1])
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


    def forward(self, inputs):

        inputs = inputs.float()
        inputs = self.input_layter(inputs)
        inputs = self.net(inputs)

        inputs = inputs.unsqueeze(1)
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = self.dropout_conv(x)
        x = x.permute(0, 2, 1)

        y = self.mixer(inputs, x)
        y = y.view(y.shape[0], -1)
        y = self.dropout_channel(y)

        y = self.output_layer(y)
        return y



'''训练验证测试部分相关设定要求：主要是val_wei以及学习率的调度'''
class DLLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PredictModel(args)
        self.test_pearson = PearsonCorrCoef()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_times = set()

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        tsdatas, rets, times, code_values, liquids = batch
        loss_fn = get_loss_fn(self.args.loss)
        loss_list = []

        # datamodule只是按照批次并行读取了数据，但是在实际处理过程中仍旧是一个截面一个截面地进行处理
        for i in range(len(tsdatas)):
            tsdata, ret, liquid = tsdatas[i], rets[i], liquids[i]
            ret = ret.unsqueeze(1)
            
            preds = self.forward(tsdata)
            loss = loss_fn(preds, ret)

            loss_list.append(loss)

        total_loss = sum(loss_list) / len(loss_list)
        self.log('train_loss', total_loss, prog_bar=True, on_step=True)
        return total_loss

    def _evaluate_step(self, batch, batch_idx, stage):
        def get_excess_return(preds, ret, liquid, money):
            _, sort = preds.sort(dim=0, descending=True, stable=True)
            sort = sort.squeeze(1)
            total_hold = torch.tensor(0.0, device=preds.device)
            total_earned = torch.tensor(0.0, device=preds.device)
            for num, idx in enumerate(sort):
                if num >= 500:
                    break
                if (money - total_hold) < 1:
                    break
                hold_money = min(money - total_hold, liquid[idx])
                total_hold += hold_money
                total_earned += ret[idx] * hold_money
            total_ret = total_earned / money
            return total_ret
        excess_return_list = []
        ic_list = []
        tsdatas, rets, times, code_values, liquids = batch
        model_cpu = copy.deepcopy(self.model)
        model_cpu = model_cpu.to('cpu')
        model_cpu.eval()
        save_model = False
        for i in range(len(tsdatas)):
            tsdata, ret, time, code_value, liquid = tsdatas[i], rets[i], times[i], code_values[i], liquids[i]
            if tsdata == 'out_sample':
                pass
            else:
                preds = self.forward(tsdata)
                if stage == "test":
                    self.test_times.add(time)
                    data_cpu = tsdata.to('cpu')
                    preds_cpu = model_cpu.forward(data_cpu)
                    res = pd.DataFrame(preds_cpu.cpu().detach().numpy(), index=code_value, columns=['value'])
                    res.index.name = 'Code'
                    res.to_pickle(f'{params.test_save_path}/{time}.pkl')

                excess_return = get_excess_return(preds, ret, liquid, money=1.5e9)
                excess_return_list.append(excess_return)
                ic_list.append(self.test_pearson(preds.squeeze(), ret))
        try:
            res_list = [sum(excess_return_list) / len(excess_return_list), sum(ic_list) / len(ic_list)]
        except:
            res_list = [np.nan, np.nan]
        if stage == 'val':
            self.validation_step_outputs.append(res_list)
        if stage == "test":
            self.test_step_outputs.append(res_list)
        return res_list

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'val')

    def on_validation_epoch_end(self):
        val_step_outputs = self.validation_step_outputs
        num_batch = len(val_step_outputs)
        self.log('val_ret', sum([(data[0] * 100) for data in val_step_outputs]) / num_batch, prog_bar=True,
                 sync_dist=True)
        self.log('val_icmean', sum([(data[1]) for data in val_step_outputs]) / num_batch, prog_bar=True,
                 sync_dist=True)
        self.log('val_wei', sum([(data[0] * 100 + data[1] * 0.7) for data in val_step_outputs]) / num_batch, prog_bar=True,
                 sync_dist=True)
        self.validation_step_outputs.clear()
        gc.collect()

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        num_batch = len(test_step_outputs)
        self.log('test_wei', sum([data[0] * 100 for data in test_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.log('test_icmean', sum([(data[1]) for data in test_step_outputs]) / num_batch, prog_bar=True,sync_dist=True)
        self.test_step_outputs.clear()

        self.test_times.clear()

    def configure_optimizers(self):
        kwargs = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay,
        }
        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]
        # 添加学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',          # 监控的指标模式（'max' 表示指标越大越好）
            factor=0.5,         # 学习率衰减因子（每次调整为原来的 0.5 倍）
            patience=3,         # 容忍的 epoch 数（如果 5 个 epoch 指标没有提升，则降低学习率）
            verbose=True,        # 打印学习率调整信息
            min_lr=5e-6,         # 学习率下限
            cooldown=2           # 下降后冷却两个epoch不会继续下降
        )
        optim_config = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_wei',  # 监控的指标（与 validation_step 中的 self.log('val_wei') 对应）
            },
        }
        return optim_config

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_wei', mode='max', save_top_k=10, save_last=False,
                            filename='{epoch}-{val_wei:.4f}')
        ]
        if self.args.swa:
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=0.7,
                                                       device='gpu'))
        if self.args.early_stop:
            callbacks.append(EarlyStopping(monitor='val_ret',
                                           mode='max', patience=6))
        return callbacks



# 模型训练
def train_single(args, name, seed, train_date_list, valid_date_list, test_date_list):
    torch.set_num_threads(args.threads)
    seed_everything(seed)
    logger = TensorBoardLogger(save_dir=params.model_path, name=name)
    profiler = SimpleProfiler(dirpath=params.profiler_path, filename=name)
    args_for_trainer = dict()
    for key, value in vars(args).items():
        try:
            Trainer(**{key: value})
            args_for_trainer[key] = value
        except:
            pass
    trainer = Trainer(**args_for_trainer,
                      num_sanity_val_steps=1,
                      profiler=profiler,
                      logger=logger,
                      deterministic=True)

    litmodel = DLLitModule(args)
    dm = DLDataModule(args, train_date_list, valid_date_list, test_date_list)
    trainer.fit(litmodel, dm)
    best_ckpt = trainer.checkpoint_callback.best_model_path
    test_result = trainer.test(ckpt_path=best_ckpt, datamodule=dm)
    print(test_result)


# 训练函数
def train(args, name, market, season, state='train'):
    # 保存最终模型的路径
    save_path = rf"{root_path}/model_test/{get_basic_name()}"
    try:
        os.makedirs(save_path, exist_ok=True)
        # 备份模型脚本
        current_file_path = os.path.abspath(__file__)
        shutil.copy(current_file_path, save_path)
    except Exception as e:
        print(e)

    params.model_name = f"{save_path}/{name[:len(market) + 19]}"
    params.test_save_path = f"{save_path}/{name[:len(market)] + name[len(market) + 6:len(market) + 19]}"
    os.makedirs(params.test_save_path, exist_ok=True)

    # 读取因子
    sel_fac_name_save = [x for x in os.listdir(rf'{fac_path}/{fac_name}') if season in x]
    assert len(sel_fac_name_save) == 1
    sel_fac_name_save = sel_fac_name_save[0]
    params.all_data = pd.read_feather(rf'{fac_path}/{fac_name}/{sel_fac_name_save}')
    
    date_list = list(params.all_data["date"].unique())
    date_list = [x for x in date_list if x in params.ret_data.index and x in params.liquid_data.index]
    date_list.sort()
    params.all_data = params.all_data.set_index("date").sort_index()

    from datetime import datetime, timedelta
    import random
    def get_train_date_split(kfold ,date_list , season , period , interval=10 , month=24):
        test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2)
        test_start = datetime.strptime(test_start, "%Y%m")
        test_end =test_start +relativedelta(months=3)
        # 设置训练起始时间
        if args.all_train:
            train_start = datetime.strptime(date_list[0], "%Y%m%d")
        else:
            train_start = test_start - relativedelta(months=month)

        print(f"test_start: {test_start}, test_end: {test_end}, train_start: {train_start}")
        test_date_list = [x for x in date_list if test_start <= datetime.strptime(x, "%Y%m%d") < test_end]
        print(f"test_date_list: {test_date_list}")
        # 极端行情不参与训练
        not_train_date = [x for x in date_list if (x >= "202402") & (x <= "20240223")]
        # 2024国庆后的下跌行情归入2024q3的测试集以及2024q4训练集
        if season == '2024q3':
            test_date_list += [x for x in date_list if '20241001' <= x <= '20241016']
        if season == '2024q4':
            test_date_list = [x for x in test_date_list if '20241017' <= x]
        
        # === 分支1：shuffle 切分逻辑 ===
        if args.shuffle_split:
            # 获取 train+valid 区间（提前 interval 天以防泄露）
            train_valid_end = test_start - timedelta(days=interval)
            train_valid_date_list = [x for x in date_list if train_start <= datetime.strptime(x, "%Y%m%d") <= train_valid_end]
            train_valid_date_list = [x for x in train_valid_date_list if x not in not_train_date]
            # 打乱并划分 KFold
            random.seed(args.seed)
            train_valid_date_list = sorted(train_valid_date_list)
            random.shuffle(train_valid_date_list)
            fold_size = len(train_valid_date_list) // kfold
            folds = [train_valid_date_list[i*fold_size : (i+1)*fold_size] for i in range(kfold - 1)]
            folds.append(train_valid_date_list[(kfold-1)*fold_size:]) 
            valid_date_list = folds[period - 1]
            train_date_list = [d for i, fold in enumerate(folds) if i != (period - 1) for d in fold]
            if season == '2024q4':
                train_date_list += [x for x in date_list if '20241001' <= x <= '20241016']
        # === 分支2：原时间序列划分逻辑 ===
        else:
            day_n = test_start - train_start
            day_num = day_n.days
            gap = day_num // kfold

            valid_date_split = [train_start]
            for i in range(1, kfold):
                valid_date_split.append(valid_date_split[i-1] + timedelta(days=gap))
            valid_date_split.append(test_start)
            valid_date_split.append(test_end)

            valid_start = valid_date_split[period - 1]
            valid_end = valid_date_split[period]
            test_end = valid_date_split[-1]

            valid_date_list = [x for x in date_list if valid_start <= datetime.strptime(x, "%Y%m%d") < valid_end][:-interval]

            # 训练集分两段，间隔interval
            train_date_list = (
                [x for x in date_list if train_start <= datetime.strptime(x, "%Y%m%d") < valid_start][:-interval] +
                [x for x in date_list if valid_end <= datetime.strptime(x, "%Y%m%d") < test_start][interval:-interval]
            )

            # 剔除异常区间
            train_date_list = [x for x in train_date_list if x not in not_train_date]

            if season == '2024q4':
                train_date_list += [x for x in date_list if '20241001' <= x <= '20241016']

        return train_date_list , valid_date_list , test_date_list

    args.all_train = True
    args.shuffle_split = True  
    train_date_list , valid_date_list , test_date_list = get_train_date_split(
        kfold=4, 
        date_list=date_list, 
        season=season, 
        period=int(params.test_save_path[-1]), 
        interval=10, 
        month=24)
    train_date_list.sort()
    valid_date_list.sort()
    test_date_list.sort()

    if len(test_date_list) == 0:
        test_date_list = ['out_sample']
    elif market == 'ALL':
        params.all_data = params.all_data.loc[:, params.all_data.replace(0, np.nan).dropna(how="all", axis=1).columns]
    else:
        raise NotImplementedError

    feature_map = list(params.all_data.columns[1:])
    params.factor_list = feature_map[:]
    with open(rf'{save_path}/{market}{name[len(market):len(market) + 6]}-feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(feature_map):
            file.write(rf'{factor_name}={idx}')
            file.write('\n')

    params.factor_num = params.all_data.shape[1] - 1

    # debug
    print(f"season: {season}, period: {params.test_save_path[-1]}")
    print(f"train_date_list: {train_date_list}")
    print(f"valid_date_list: {valid_date_list}")
    print(f"test_date_list: {test_date_list}")

    if state == 'train':
        train_single(args, name, args.seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError
