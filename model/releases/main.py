#!/usr/bin/env python
"""执行即可推演所有发布模型最近 10 个交易日，生成 DAILY-YYYYMMDD.md。

读取已训练权重、model/trainingdata 与本地冻结输入合同；按训练列名、顺序、方向评分。
各模型四折相加，再按发布清单冻结的方法和比例在共同有效股票中融合。
每天一份独立日报，同日增量刷新；输入未变时复用缓存，旧日报保留。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

# 在 numpy/torch 导入前设置；与发布模型的 CPU 交付口径保持一致。
os.environ.setdefault('MX_THREADS', '1')
os.environ['MX_LIVE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = os.environ['MX_THREADS']
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
STATE_NAME = '.daily_state.json'
BEGIN = '<!-- releases/main.py:begin -->'
END = '<!-- releases/main.py:end -->'
KEYS = ['trade_date', 'stock_code']
VERSION = 5
CONTRACT_PATH = ROOT / 'input_contracts.json'
CONFIG_PATH = ROOT / 'release_config.json'


def daily_path(directory, now=None):
    """按机器本地执行日期命名，交易日窗口单独以快照日期为准。"""
    return Path(directory) / f'DAILY-{(now or datetime.now()).strftime("%Y%m%d")}.md'


def say(message):
    tqdm.write(str(message))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def atomic_text(path, text):
    """内容未变不写；临时文件与目标同目录，替换时不留下半份日报。"""
    path = Path(path)
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.chmod(temp, 0o664)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return True


@contextmanager
def update_lock(directory):
    """避免两次执行同时覆盖缓存/日报。"""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.daily.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('另一个 releases/main.py 正在更新，请等待它完成') from exc
        yield


def discover_units(directory):
    units = sorted(p for p in Path(directory).iterdir()
                   if p.is_dir() and not p.name.startswith('.') and (p / 'model.py').is_file())
    if not units:
        raise ValueError(f'{directory} 没有含 model.py 的发布模型目录')
    return units


def discover_checkpoints(unit):
    """选择最新完整季度的四折。最新季度不完整时直接报错。"""
    quarters = sorted(p for p in (unit / 'model_train').glob('*')
                      if p.is_dir() and re.fullmatch(r'\d{4}Q[1-4]', p.name))
    if not quarters:
        raise ValueError(f'{unit.name} 没有已训练的季度权重')
    quarter = quarters[-1]
    paths = [quarter / f'fold{k}' / 'best.pt' for k in (1, 2, 3, 4)]
    missing = [str(p.relative_to(unit)) for p in paths
               if not p.is_file() or not (p.parent / 'complete.json').is_file()
               or not (p.parent / 'training_info.json').is_file()]
    if missing:
        raise ValueError(f'{unit.name} 最新季度 {quarter.name} 四折不齐：{missing}')
    return quarter.name, paths


def load_module(unit):
    name = '_release_daily_' + re.sub(r'\W', '_', unit.name)
    spec = importlib.util.spec_from_file_location(name, unit / 'model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for required in ('RECIPE', 'PredictModel', 'Panel', 'predict'):
        if not hasattr(module, required):
            raise ValueError(f'{unit.name}/model.py 缺少推演接口 {required}')
    return module


def validate_columns(unit, checkpoint, training_info, module, meta):
    """以 checkpoint 为准对表：多余列忽略、缺列报错、保存顺序严格一致。"""
    for required in ('model', 'features', 'ridge', 'input_dim', 'market_dim', 'horizons'):
        if required not in checkpoint:
            raise ValueError(f'{unit.name} 权重缺少 {required}，无法完整恢复交付模型')
    features = list(checkpoint['features'])
    # 现有 best.pt 未保存市场列名，必须取同折训练时保存的顺序，不能用新快照顺序代替。
    if 'feature_columns' not in training_info or 'market_columns' not in training_info:
        raise ValueError(f'{unit.name} training_info.json 缺少冻结的股票/市场列清单')
    market = list(training_info['market_columns'])
    if len(set(market)) != len(market):
        raise ValueError(f'{unit.name} 训练市场列清单重复')
    if 'market_features' in checkpoint and list(checkpoint['market_features']) != market:
        raise ValueError(f'{unit.name} checkpoint 与训练记录的市场列顺序不一致')
    if not features or len(set(features)) != len(features):
        raise ValueError(f'{unit.name} 权重特征清单为空或重复')
    expected = list(module.RECIPE['feature_names'])
    if features != expected or features != list(training_info['feature_columns']):
        raise ValueError(f'{unit.name} checkpoint、训练记录与模型 RECIPE 的特征列名/顺序不一致')
    if list(checkpoint['horizons']) != list(module.RECIPE['output_horizons']):
        raise ValueError(f'{unit.name} checkpoint.horizons 与模型输出顺序不一致')
    if (training_info.get('recipe') or {}).get('snapshot_meta_sha256') != module.RECIPE.get('snapshot_meta_sha256'):
        raise ValueError(f'{unit.name} 训练记录与模型的输入快照指纹不一致')
    missing = [n for n in features if n not in meta['columns']['features']]
    if missing:
        raise ValueError(f'{unit.name} 当前快照缺少权重必需的 {len(missing)} 个特征：{missing}')
    market_available = set((meta.get('market_factors') or {}).get('columns') or [])
    missing_market = [n for n in market if n not in market_available]
    if missing_market:
        raise ValueError(f'{unit.name} 当前快照缺少市场输入：{missing_market}')
    if checkpoint.get('input_dim', len(features)) != len(features):
        raise ValueError(f'{unit.name} checkpoint.input_dim 与特征数不一致')
    if checkpoint.get('market_dim', len(market)) != len(market):
        raise ValueError(f'{unit.name} checkpoint.market_dim 与市场列数不一致')
    return features, market


def direction_multipliers(unit, features, market, meta, training_sha, current_sha, contracts):
    """把 snapshot=z×当前方向 转为 model=z×训练方向；方向缺失时拒绝评分。"""
    current = (meta.get('columns') or {}).get('direction') or {}
    if not training_sha:
        raise ValueError(f'{unit.name} 未记录训练快照指纹，无法确认因子方向')
    if current_sha == training_sha:
        expected = current
    else:
        contract = (contracts.get('snapshots') or {}).get(training_sha)
        if not contract:
            raise ValueError(f'{unit.name} 快照已变化，但缺少训练方向合同 {training_sha}；不能只按列名加载')
        if contract.get('semantics') != meta.get('semantics'):
            raise ValueError(f'{unit.name} 训练方向合同与当前快照的标准化口径不一致')
        per_model = contract.get('market_columns_by_model')
        if per_model is not None:
            if unit.name not in per_model:
                raise ValueError(f'{unit.name} 缺少该模型的冻结市场列合同')
            frozen_market = per_model[unit.name]
        else:
            frozen_market = contract.get('market_columns') or []
        if list(frozen_market) != market:
            raise ValueError(f'{unit.name} 训练方向合同与权重的市场列名/顺序不一致')
        expected = contract.get('feature_direction') or {}
    missing = [n for n in features if n not in current or n not in expected]
    if missing:
        raise ValueError(f'{unit.name} 无法确认训练/当前因子方向：{missing}')
    invalid = [n for n in features if current[n] not in (-1, 1) or expected[n] not in (-1, 1)]
    if invalid:
        raise ValueError(f'{unit.name} 因子方向必须为 -1 或 1：{invalid}')
    return [int(expected[n]) * int(current[n]) for n in features]


def snapshot_axis(data, meta):
    days = []
    for year in sorted(meta['built_years']):
        path = data / 'factors' / f'year={year}' / 'data.parquet'
        col = pq.ParquetFile(path).read(columns=['trade_date'], use_threads=False)['trade_date']
        days.extend(sorted(str(d)[:10] for d in pc.unique(col).to_pylist()))
    if not days or days != sorted(set(days)):
        raise ValueError('快照日期轴为空、重复或分区顺序异常')
    codes = np.asarray(meta['axis']['codes'], dtype=object)
    if not len(codes) or len(set(codes)) != len(codes):
        raise ValueError('快照股票轴为空或重复')
    return np.asarray(days), codes


def read_block(data, kind, days, codes, columns, stock=True):
    """仅取窗口主键及权重需要的列；始终按给定主键对齐。"""
    parts = []
    for year in sorted({d[:4] for d in days}):
        path = data / kind / f'year={year}' / 'data.parquet'
        if not path.is_file():
            raise ValueError(f'快照缺少 {kind}/{year} 分区')
        required = (KEYS if stock else ['trade_date']) + list(columns)
        schema = pq.ParquetFile(path).schema_arrow.names
        missing = [c for c in required if c not in schema]
        if missing:
            raise ValueError(f'{path} 缺少列：{missing}')
        frame = pq.read_table(path, columns=required,
                              filters=[('trade_date', '>=', days[0]), ('trade_date', '<=', days[-1])],
                              use_threads=False).to_pandas()
        frame['trade_date'] = frame['trade_date'].astype(str).str[:10]
        if stock:
            frame['stock_code'] = frame['stock_code'].astype(str)
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True).set_index(KEYS if stock else ['trade_date'])
    if frame.index.has_duplicates:
        raise ValueError(f'{kind} 窗口内主键重复')
    index = (pd.MultiIndex.from_product([days, codes], names=KEYS) if stock
             else pd.Index(days, name='trade_date'))
    if len(frame) != len(index) or not frame.index.sort_values().equals(index):
        raise ValueError(f'{kind} 窗口日期/股票轴不一致，不能静默补行')
    frame = frame.reindex(index)
    if list(frame.columns) != list(columns):
        raise ValueError(f'{kind} 投影后列顺序不一致')
    values = frame.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError(f'{kind} 输入含无穷值')
    return values.reshape(len(days), len(codes), len(columns)) if stock else values


def make_panel(module, data, meta, days, codes, features, market, shared_x, feature_positions, shared_market,
               multipliers):
    """使用原 Panel 的 inputs/mask，读取窗口而非整段历史；不修改训练端的 pin。"""
    panel = module.Panel.__new__(module.Panel)
    panel.root, panel.meta = data, meta
    panel.days, panel.codes = np.asarray(days), codes
    panel.features, panel.market_features = features, market
    panel.market_dim = len(market)
    positions = [feature_positions[f] for f in features]
    # 显式把方向统一到训练口径。已按训练方向存储时乘 1，不重复翻转。
    signs = np.asarray(multipliers, dtype=np.float32)
    if signs.shape != (len(features),) or not np.isin(signs, [-1., 1.]).all():
        raise ValueError('模型输入的方向转换向量不合法')
    x = shared_x[:, :, positions] * signs
    panel.coverage = np.isfinite(x).mean(axis=2)
    for group in ('flow', 'fund', 'technical'):
        indices = module.RECIPE[f'{group}_indices']
        if not indices or max(indices) >= len(features):
            raise ValueError(f'{module.UNIT} {group} 特征索引不合法')
        setattr(panel, group + '_coverage', np.isfinite(x[:, :, indices]).mean(axis=2))
    panel.X = np.nan_to_num(x, nan=0.)
    panel.M = (np.clip(np.nan_to_num(shared_market[tuple(market)], nan=0.),
                       -module.RECIPE['market_clip'], module.RECIPE['market_clip'])
               if market else np.zeros((len(days), 0), np.float32))
    panel.Y = {}  # 推演不读取未来收益标签。
    panel.amount = None
    return panel


def make_prices(module, panel, data):
    """全池模型无需价格。有限热门池沿用发布模型自身的历史窗口算法。"""
    recipe = module.RECIPE['hot_pool']
    if recipe['frac'] >= 1.:
        return SimpleNamespace(hot_frac=recipe['frac'])
    panel.amount = read_block(data, 'amount', panel.days, panel.codes, ['amount'])[:, :, 0]
    price = module.Prices.__new__(module.Prices)
    columns = list(module.PRICE_COLUMNS)
    values = read_block(data, 'prices', panel.days, panel.codes, columns)
    price.raw = {n: values[:, :, i].astype(np.float64) for i, n in enumerate(columns)}
    pre = price.raw['pre_close']
    price.limit_up = np.floor(pre * 1.10 * 100 + .5) / 100
    price._hot_pool(panel)
    return price


def rank_order(values, codes):
    """有限分数按降序；并列时按股票代码稳定排序。"""
    valid = np.flatnonzero(np.isfinite(values))
    return valid[np.lexsort((codes[valid], -values[valid]))]


def top_rows(values, codes, names, count=10):
    return [dict(rank=i, code=str(codes[c]), name=names.get(str(codes[c]), ''), score=float(values[c]))
            for i, c in enumerate(rank_order(values, codes)[:count], 1)]


def fuse_scores(scores, method='rank_mean', weights=None):
    """在共同有效股票内变换各分数，再按已冻结发布权重融合。"""
    from scipy.stats import rankdata
    from scipy.special import ndtri
    if method not in ('rank_mean', 'zscore_clip3', 'gaussian_rank', 'rank_square'):
        raise ValueError('未知融合方法')
    if weights is None:
        weights = dict.fromkeys(scores, 1.)
    if set(weights) != set(scores):
        raise ValueError('融合权重与模型集合不一致')
    mass = np.asarray([weights[k] for k in scores], dtype=np.float64)
    if not np.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError('融合权重必须为有限正数')
    stack = np.stack(list(scores.values())).astype(np.float64)
    common = np.isfinite(stack).all(axis=0)
    fused = np.full(stack.shape[1], np.nan, np.float64)
    if not common.any():
        raise ValueError('模型没有共同有效的股票，无法生成融合排名')
    transformed = []
    for row in stack:
        values = row[common]
        q = (rankdata(values, method='average') - .5) / common.sum()
        if method == 'rank_mean':
            value = q
        elif method == 'rank_square':
            value = q * q
        elif method == 'gaussian_rank':
            value = ndtri(q)
        else:
            std = values.std(ddof=1) if len(values) > 1 else 0.
            value = np.clip((values - values.mean()) / std, -3., 3.) if std > 0 else np.zeros_like(values)
        transformed.append(value)
    total = np.zeros(int(common.sum()), dtype=np.float64)
    for weight, value in zip(mass, transformed):
        total += weight * value
    fused[common] = total / mass.sum()
    return fused, int(common.sum())


def release_config(units):
    """缺少清单时兼容原四折入口；有清单时严格绑定发布集合。"""
    if not CONFIG_PATH.is_file():
        return dict(method='rank_mean', weights=dict.fromkeys((u.name for u in units), 1.))
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if (config.get('status') != 'active' or len(config['models']) != 4
            or sorted(config['models']) != sorted(u.name for u in units)):
        raise ValueError('发布清单状态或四模型集合与目录不一致')
    fusion = config['fusion']
    # Validate before reading market blocks, including methods/positive weights.
    fuse_scores({u.name: np.asarray([0., 1.]) for u in units}, **fusion)
    return fusion


def verify_release_files(plans):
    """Bind operational code, checkpoints and training records to the active manifest."""
    if not CONFIG_PATH.is_file():
        return
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    required = {'main.py'}
    for plan in plans:
        if plan['quarter'] != config['quarter']:
            raise ValueError('实际权重季度与发布清单不一致')
        if 'seed_sets' in config and plan['seed_sets'] != config['seed_sets'][plan['unit'].name]:
            raise ValueError('实际种子集合与发布清单不一致')
        required.update(str((plan['unit'] / name).relative_to(ROOT))
                        for name in ('model.py', 'run.py', 'analysis.py'))
        for weight in plan['checkpoints']:
            required.update(str(path.relative_to(ROOT)) for path in
                            (weight, weight.parent / 'training_info.json', weight.parent / 'complete.json'))
    if not required.issubset(config['files']):
        raise ValueError('发布清单缺少必需源码、权重或训练记录指纹')
    for name, expected in config['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'发布产物指纹变化：{name}')


def references(project):
    """名称和未来交易日只用于展示，不参与模型评分。"""
    raw = project.parent / 'datadownload' / 'data'
    stocks = Path(os.environ.get('MX_STOCK_LIST', raw / 'stock_list/data.parquet'))
    calendar = Path(os.environ.get('MX_CALENDAR', raw / 'basic_calendar/data.parquet'))
    names, open_days = {}, []
    if stocks.is_file():
        frame = pd.read_parquet(stocks, columns=['stock_code', 'name'])
        names = dict(zip(frame.stock_code.astype(str), frame.name.astype(str)))
    if calendar.is_file():
        frame = pd.read_parquet(calendar, columns=['date', 'is_open'])
        open_days = sorted(set(frame.loc[frame.is_open.astype(str).isin(['1', '1.0', 'True']), 'date']
                               .astype(str).str[:10]))
    return names, open_days


def format_score(value):
    return f'{value:.6f}'


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def render_entry(entry):
    budget = entry.get('strategy_budget')
    units = entry['units']
    next_day = entry.get('next_day') or '待日历补齐'
    lines = [BEGIN, f"下一交易日：**{next_day}**；共同有效股票：{entry['common_count']} 只。", '',
             '**融合 Top10**', '', '| 名次 | 代码 | 名称 | 融合分 | ' + ' | '.join(units) + ' |',
             '|---:|---|---|---:|' + '---:|' * len(units)]
    for row in entry['fusion']:
        lines.append(f"| {row['rank']} | {row['code']} | {cell(row['name'])} | {format_score(row['score'])} | "
                     + ' | '.join(str(row['ranks'][u]) for u in units) + ' |')
    for unit in units:
        lines += ['', f'### {unit} · Top10', '', '| 名次 | 代码 | 名称 | 模型分 |', '|---:|---|---|---:|']
        for row in entry['models'][unit]:
            lines.append(f"| {row['rank']} | {row['code']} | {cell(row['name'])} | {format_score(row['score'])} |")
    lineages = '；'.join(f"{u}={entry['quarters'][u]} 四折相加"
                         + (f"（每折{len(entry['seed_sets'][u])}种子等权原始分数）"
                            if len(entry.get('seed_sets', {}).get(u, [])) > 1 else '') for u in units)
    fusion = entry.get('fusion_config', dict(method='rank_mean', weights=dict.fromkeys(units, 1.)))
    total = sum(fusion['weights'].values())
    lines += ['', f"融合：`{fusion['method']}`；" + '、'.join(f"{u} {fusion['weights'][u] / total:.0%}" for u in units) + '。',
              f"权重：{lineages}。", f"输入快照：`{entry['snapshot_sha256']}`；值口径：`{entry['semantics']}`。",
              '按训练方向转换：' + '；'.join(f"{u} {entry['direction_flips'][u]} 列" for u in units) + '。', END]
    if budget:
        lines.insert(2, f"事件风险概率：{budget['risk_probability']:.2%}；下一开盘目标仓位：**{budget['target_exposure']:.0%}**，最多{budget['target_names']}只；B40。目标清仓仍受涨跌停等成交条件限制。")
    return '\n'.join(lines)


def merge_markdown(previous, entries, latest_days, source_end):
    """同一日报按交易日更新受管理的推演块，保留窗口内人工备注。"""
    sections = {}
    matches = list(re.finditer(r'^## (\d{4}-\d{2}-\d{2})[^\n]*\n', previous, re.MULTILINE))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(previous)
        day = match.group(1)
        body = previous[match.end():end].strip()
        heading = match.group(0).strip()
        if day in sections:
            raise ValueError(f'日报已有重复日期 {day}，请先合并记录')
        sections[day] = (heading, body)
    # 保留自定义首页内容（自动生成的摘要单独替换）。
    prefix = previous[:matches[0].start()] if matches else previous
    prefix = re.sub(r'<!-- releases/main.py:header -->.*?<!-- releases/main.py:header-end -->\s*',
                    '', prefix, flags=re.DOTALL)
    if not prefix.strip():
        prefix = '# releases 每日推演（最新在上）\n\n'
    prefix = prefix.rstrip() + '\n\n'
    header = [ '<!-- releases/main.py:header -->',
               f'> 由 `python releases/main.py` 更新；最近 {len(latest_days)} 个交易日：{latest_days[0]} ~ {latest_days[-1]}；数据截至 {source_end}。',
               '> 每个发布模型使用已有四折权重；融合方法及比例见各日记录，发布选择见 `release_config.json`。',
               '> 以下是当前已训练权重对历史日期的推演；不自动生成或变更持仓。每天另存一份日报，同日更新此文件。',
               '<!-- releases/main.py:header-end -->', '' ]
    for day, entry in entries.items():
        generated = render_entry(entry)
        heading, body = sections.get(day, (f'## {day}', ''))
        if BEGIN in body:
            if body.count(BEGIN) != 1 or body.count(END) != 1:
                raise ValueError(f'日报 {day} 自动生成标记损坏')
            body = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), lambda _: generated,
                          body, count=1, flags=re.DOTALL)
        elif body:
            # 初次迁移旧日报：不把旧持仓作为更早日期的期初持仓，也不覆盖用户记录。
            body = generated + '\n\n<details>\n<summary>此前持仓及操作记录（原文保留）</summary>\n\n' + body + '\n\n</details>'
        else:
            body = generated
        sections[day] = (heading, body)
    lines = [prefix + '\n'.join(header)]
    for day in sorted(set(sections) & set(map(str, latest_days)), reverse=True):
        heading, body = sections[day]
        lines.append(heading + '\n\n' + body + '\n')
    return '\n'.join(lines).rstrip() + '\n'


def build_plan(units, meta, current_sha, contracts):
    import torch
    plans = []
    for unit in tqdm(units, desc='核对模型与权重', unit='模型', dynamic_ncols=True):
        module = load_module(unit)
        quarter, checkpoints = discover_checkpoints(unit)
        features = market = None
        checkpoint_hashes = []
        training_hashes = []
        for path in checkpoints:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            info_path = path.parent / 'training_info.json'
            info = json.loads(info_path.read_text(encoding='utf-8'))
            columns, markets = validate_columns(unit, ck, info, module, meta)
            if features is not None and (columns != features or markets != market):
                raise ValueError(f'{unit.name} 四折输入列名/顺序不一致')
            features, market = columns, markets
            checkpoint_hashes.append(sha256(path))
            training_hashes.append(sha256(info_path))
        training_sha = module.RECIPE.get('snapshot_meta_sha256')
        multipliers = direction_multipliers(unit, features, market, meta, training_sha, current_sha, contracts)
        flips = sum(s < 0 for s in multipliers)
        say(f'{unit.name}：股票输入 {len(features)} 列 · 市场输入 {len(market)} 列 · 按训练方向转换 {flips} 列')
        plans.append(dict(unit=unit, module=module, quarter=quarter, checkpoints=checkpoints,
                          features=features, market=market, checkpoint_hashes=checkpoint_hashes,
                          training_sha=training_sha, multipliers=multipliers,
                          training_hashes=training_hashes, source_hash=sha256(unit / 'model.py'),
                          seed_sets=list(module.RECIPE.get('replica_seed_sets', [module.RECIPE.get('seed')]))))
    return plans


def score_payload_hash(identity, raw, fused, common):
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
    for array,dtype in [(raw,'<f8'),(fused,'<f8'),(common,'?')]:
        value=np.asarray(array,dtype=dtype)
        digest.update(str(value.shape).encode());digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def archive_scores(directory,day,codes,scores,fused,next_day,fingerprint,provenance,now=None):
    """Append immutable full vectors; repeat payloads keep the earliest recorded timestamp."""
    directory=Path(directory)
    units=list(scores);codes=np.asarray(codes,dtype=str)
    raw=np.asarray([scores[name] for name in units],dtype=np.float64)
    fused=np.asarray(fused,dtype=np.float64)
    if raw.shape!=(len(units),len(codes)) or fused.shape!=(len(codes),) or np.isinf(raw).any() or np.isinf(fused).any():
        raise ValueError('完整分数归档的日期/模型/股票轴异常')
    common=np.isfinite(raw).all(axis=0)
    if not np.array_equal(np.isfinite(fused),common):raise ValueError('融合分数与共同有效股票池不一致')
    identity=dict(schema=1,trade_date=str(day),next_open_day=next_day,fingerprint=fingerprint,
        stock_codes=codes.tolist(),units=units,model_provenance=provenance,
        raw_score_meaning='sum of four frozen fold scores per released model; declared seed replicas averaged within fold; raw amplitudes retained',
        fusion_meaning=provenance.get('fusion', 'equal mean percentile rank on common finite stock pool'))
    payload=score_payload_hash(identity,raw,fused,common)
    path=directory/'signal_archive'/str(day)/f'{payload}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata'].item()))
            actual=score_payload_hash(metadata['identity'],saved['raw_scores'],saved['fused_rank'],saved['common_mask'])
            if (actual!=payload or metadata['payload_sha256']!=payload
                    or not np.array_equal(saved['stock_codes'],codes)
                    or saved['units'].tolist()!=units):
                raise RuntimeError('已有完整分数归档损坏；保留原文件并停止写入')
    else:
        created=now or datetime.now(timezone.utc)
        if created.tzinfo is None:raise ValueError('分数归档时间必须包含时区')
        created=created.astimezone(timezone.utc)
        next_open=datetime.fromisoformat(next_day+'T09:30:00').replace(tzinfo=ZoneInfo('Asia/Shanghai')).astimezone(timezone.utc) if next_day else None
        relation='unknown_next_open' if next_open is None else 'recorded_before_next_open' if created<next_open else 'retrospective_after_next_open'
        metadata=dict(identity=identity,payload_sha256=payload,recorded_at_utc=created.isoformat(),
            next_open_at_utc=next_open.isoformat() if next_open else None,time_relation=relation,
            interpretation='generation time only; earlier dates are retrospective, and this does not prove upstream point-in-time factor availability')
        stream=io.BytesIO()
        np.savez_compressed(stream,stock_codes=codes,units=np.asarray(units,dtype=str),raw_scores=raw,
            fused_rank=fused,common_mask=common,metadata=np.asarray(json.dumps(metadata,ensure_ascii=False,sort_keys=True)))
        path.parent.mkdir(parents=True,exist_ok=True)
        # Link a complete same-filesystem temporary file; the final archive is never partial or replaced.
        handle,temporary=tempfile.mkstemp(dir=path.parent,prefix='.scores-',suffix='.tmp')
        try:
            with os.fdopen(handle,'wb') as output:output.write(stream.getvalue())
            os.chmod(temporary,0o664)
            os.link(temporary,path)
        except FileExistsError:
            return archive_scores(directory,day,codes,scores,fused,next_day,fingerprint,provenance,now)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    return dict(path=str(path.relative_to(directory)),sha256=sha256(path),payload_sha256=payload,
        recorded_at_utc=metadata['recorded_at_utc'],time_relation=metadata['time_relation'])


def score_archive_valid(entry,directory):
    item=entry.get('score_archive') or {}
    if not item.get('path') or not item.get('sha256'):return False
    path=(Path(directory)/item['path']).resolve()
    if not path.is_relative_to((Path(directory)/'signal_archive').resolve()):return False
    try:return path.is_file() and sha256(path)==item['sha256']
    except OSError:return False


def predict_window(plans, data, meta, all_days, codes, requested, names, open_days,score_sink=None, fusion_config=None):
    import torch
    torch.set_num_threads(max(1, int(os.environ['MX_THREADS'])))
    lookback = max(max(p['module'].RECIPE['sequence_days'] - 1,
                       max(p['module'].RECIPE['hot_pool'].get(k, 0)
                           for k in ('amount_window', 'amp_window', 'limit_window'))
                       if p['module'].RECIPE['hot_pool']['frac'] < 1. else 0)
                   for p in plans)
    first = int(np.searchsorted(all_days, requested[0]))
    if first < max(p['module'].RECIPE['sequence_days'] - 1 for p in plans):
        raise ValueError('最新窗口前没有足够的历史交易日供序列模型预热')
    days = all_days[max(0, first - lookback):]
    selected = np.searchsorted(days, requested)
    features = sorted(set(n for p in plans for n in p['features']))
    positions = {f: i for i, f in enumerate(features)}
    say(f'读取推演窗口：{len(requested)} 日 + {first - max(0, first - lookback)} 日预热 · {len(codes)} 只股票 · {len(features)} 个模型输入特征')
    shared_x = read_block(data, 'factors', days, codes, features)
    shared_market = {}
    for columns in {tuple(p['market']) for p in plans if p['market']}:
        shared_market[columns] = read_block(data, 'market_factors', days, codes, columns, stock=False)
    scores = {}
    with tqdm(total=sum(len(p['checkpoints']) for p in plans), desc='模型推演', unit='折', dynamic_ncols=True) as progress:
        for plan in plans:
            module = plan['module']
            panel = make_panel(module, data, meta, days, codes, plan['features'], plan['market'],
                               shared_x, positions, shared_market, plan['multipliers'])
            prices = make_prices(module, panel, data)
            total = None
            for path in plan['checkpoints']:
                ck = torch.load(path, map_location='cpu', weights_only=False)
                net = module.PredictModel(len(plan['features']), len(plan['market'])).to('cpu')
                net.load_state_dict(ck['model'], strict=True)
                net.ridge = ck['ridge']
                net.eval()
                with torch.inference_mode():
                    result = np.asarray(module.predict(net, panel, prices, selected, torch.device('cpu')), dtype=np.float64)
                if result.shape != (len(requested), len(codes)) or np.isinf(result).any():
                    raise ValueError(f'{plan["unit"].name} 推演结果形状/数值异常：{result.shape}')
                total = result if total is None else total + result
                progress.set_postfix(model=plan['unit'].name, fold=path.parent.name)
                progress.update()
            scores[plan['unit'].name] = total
    units = [p['unit'].name for p in plans]
    entries = {}
    for i, day in enumerate(requested):
        model_scores = {u: scores[u][i] for u in units}
        fusion_config = fusion_config or dict(method='rank_mean', weights=dict.fromkeys(units, 1.))
        fused, common = fuse_scores(model_scores, **fusion_config)
        ranks = {u: {str(codes[c]): rank for rank, c in enumerate(rank_order(model_scores[u], codes), 1)}
                 for u in units}
        fusion = top_rows(fused, codes, names)
        for row in fusion:
            row['ranks'] = {u: ranks[u][row['code']] for u in units}
        following = next((d for d in open_days if d > str(day)), None)
        # 窗口内下个交易日优先使用快照真实日期。
        current = int(np.searchsorted(all_days, day))
        if current + 1 < len(all_days):
            following = str(all_days[current + 1])
        entries[str(day)] = dict(day=str(day), units=units, common_count=common, next_day=following,
                                 fusion=fusion, models={u: top_rows(model_scores[u], codes, names) for u in units},
                                 quarters={p['unit'].name: p['quarter'] for p in plans},
                                 direction_flips={p['unit'].name: sum(s < 0 for s in p['multipliers']) for p in plans},
                                 snapshot_sha256=sha256(data / 'meta.json'), semantics=meta['semantics'],
                                 fusion_config=fusion_config, seed_sets={p['unit'].name: p['seed_sets'] for p in plans})
        if score_sink is not None:score_sink(str(day),codes,model_scores,fused,following)
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=10, help='最近交易日数量（默认 10）')
    parser.add_argument('--data', type=Path, default=None, help='trainingdata 路径，默认脚本上一级的共享快照')
    parser.add_argument('--out', type=Path, default=None, help='日报路径，默认 releases/DAILY-YYYYMMDD.md（执行当天）；缓存跟随日报目录')
    parser.add_argument('--force', action='store_true', help='忽略评分缓存，重新推演最新窗口')
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error('--days 必须大于 0')
    data = (args.data or Path(os.environ.get('MX_DATA', PROJECT / 'trainingdata'))).expanduser().resolve()
    if data.name == 'factors' and (data.parent / 'meta.json').is_file():
        data = data.parent
    document = (args.out or daily_path(ROOT)).expanduser().resolve()
    try:
        with update_lock(document.parent):
            meta = json.loads((data / 'meta.json').read_text(encoding='utf-8'))
            if meta.get('semantics') != 'zscore_win1_99_v1':
                raise ValueError(f'模型要求 zscore_win1_99_v1，当前快照为 {meta.get("semantics")}')
            units = discover_units(ROOT)
            fusion = release_config(units)
            contracts = json.loads(CONTRACT_PATH.read_text(encoding='utf-8')) if CONTRACT_PATH.is_file() else {}
            plans = build_plan(units, meta, sha256(data / 'meta.json'), contracts)
            verify_release_files(plans)
            all_days, codes = snapshot_axis(data, meta)
            requested = all_days[-args.days:]
            earliest = max(str(pd.Period(p['quarter'], freq='Q').start_time.date()) for p in plans)
            if str(requested[0]) < earliest:
                raise ValueError(f'请求日期早于交付季度起点 {earliest}；请缩小 --days，避免以新季度权重回填旧季度')
            names, open_days = references(PROJECT)
            source_files = [data / 'meta.json']
            for kind in ['factors', 'market_factors', 'amount', 'prices']:
                source_files.extend(sorted((data / kind).glob('year=*/data.parquet')))
            model_files = [p['unit'] / 'model.py' for p in plans]
            if CONTRACT_PATH.is_file():
                model_files.append(CONTRACT_PATH)
            if CONFIG_PATH.is_file():
                model_files.append(CONFIG_PATH)
            for plan in plans:
                for path in plan['checkpoints']:
                    model_files.extend([path, path.parent / 'training_info.json', path.parent / 'complete.json'])
            fingerprint_data = dict(version=VERSION, runner_sha256=sha256(__file__), fusion=fusion,
                                    input=[signature(p) for p in source_files],
                                    model_files=[signature(p) for p in model_files],
                                    meta_sha256=sha256(data / 'meta.json'),
                                    models=[dict(unit=p['unit'].name, quarter=p['quarter'], source_hash=p['source_hash'],
                                                 checkpoint_hashes=p['checkpoint_hashes'], training_hashes=p['training_hashes'],
                                                 seed_sets=p['seed_sets'],
                                                 features=p['features'], market=p['market'], training_sha=p['training_sha'],
                                                 multipliers=p['multipliers'])
                                            for p in plans], names=names, open_days=open_days)
            fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
            state_path = document.parent / STATE_NAME
            state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
            previous_entries = dict(state.get('entries') or {}) if state.get('fingerprint') == fingerprint else {}
            cached = all(str(d) in previous_entries and score_archive_valid(previous_entries[str(d)],document.parent) for d in requested)
            score_payloads={}
            def collect_scores(day,axis,scores,fused,next_day):
                score_payloads[day]=(axis.copy(),{k:v.copy() for k,v in scores.items()},fused.copy(),next_day)
            if cached and not args.force:
                entries = {str(d): previous_entries[str(d)] for d in requested}
                say(f'缓存命中：{len(units)} 个模型 × {len(requested)} 个交易日；输入和权重未变')
            else:
                entries = predict_window(plans, data, meta, all_days, codes, requested, names, open_days,
                                         score_sink=collect_scores, fusion_config=fusion)
            if (sha256(data / 'meta.json') != fingerprint_data['meta_sha256']
                    or [signature(p) for p in source_files] != fingerprint_data['input']
                    or [signature(p) for p in model_files] != fingerprint_data['model_files']):
                raise RuntimeError('推演期间数据或模型发生变化，本次不写日报，请在更新完成后重跑')
            for day,(axis,scores,fused,next_day) in score_payloads.items():
                entries[day]['score_archive']=archive_scores(document.parent,day,axis,scores,fused,next_day,
                    fingerprint,fingerprint_data)
            import importlib.util
            strategy_spec = importlib.util.spec_from_file_location('release_budget', ROOT/'strategy.py')
            strategy = importlib.util.module_from_spec(strategy_spec)
            strategy_spec.loader.exec_module(strategy)
            budgets = strategy.budget_trace(data, str(requested[-1]), ROOT)
            for day in entries: entries[day]['strategy_budget'] = budgets[day]
            previous = document.read_text(encoding='utf-8') if document.is_file() else ''
            content = merge_markdown(previous, entries, requested, str(all_days[-1]))
            old_dates = set(re.findall(r'^## (\d{4}-\d{2}-\d{2})', previous, re.MULTILINE))
            updated = atomic_text(document, content)
            previous_entries.update(entries)
            if not cached or args.force:
                state = dict(version=VERSION, fingerprint=fingerprint, entries=previous_entries,
                             updated_at=datetime.now(timezone.utc).isoformat())
                atomic_text(state_path, json.dumps(state, indent=1, ensure_ascii=False) + '\n')
            say(f"{'已更新' if updated else '已是最新'} {document} · {len(units)} 个模型 · {requested[0]} ~ {requested[-1]} · "
                f"新增 {len(set(entries) - old_dates)} 日，刷新 {len(set(entries) & old_dates)} 日")
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        say(f'推演未完成：{exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
