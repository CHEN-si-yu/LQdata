"""新因子的单因子体检（**标签隔离修复版**，评审建议 S-03）。

## 这个文件与 `../20260921_new_factors/newfactor_ic.py` 的唯一区别

原脚本第 27/36 行：

    CUTOFF = '2025-06-30'
    df = df[df['trade_date'] <= CUTOFF]

**只按信号日截断，没有截标签末端。** 而标签是前视的：

    label_ret_5d(T) = open(T+1+5) / open(T+1) − 1        （SPEC.md §2 铁律一）

T = 2025-06-30 的标签要用到 **2025-07-08** 的开盘价 —— 落在评价窗
（2025-07-01 起）**里面**。于是最后 **6 个信号日**（06-23 ~ 06-30）的 RankIC
用到了评价窗的价格信息，而这张表**直接决定选列**（臂 D = V65 按 |ic_mean| 取前 34 名）。

## 修法（照抄平台自己的口径，不另立标准）

`V16/model.py::splits()` 的隔离判据是 `cutoff = first − h − 1`、`eligible = ix[:cutoff]`，
再加一条**严格不等式**断言 `max(train.max(), valid.max()) + h + 1 < first`
⇒ 最后一个可训练样本索引 = `first − h − 2`，其兑现日 = `first − 1`，**严格早于** first。

    评价窗首个交易日 first = 2025-07-01
    first − h − 1 = 2025-06-23 → 兑现日 2025-07-01  ✘（正好落进评价窗）
    first − h − 2 = 2025-06-20 → 兑现日 2025-06-30  ✔   ← 正确的 CUTOFF

★ **我第一版按 `first − (h+1)` 退，`_assert_isolated()` 当场就炸了** —— 差一天就是越界。
这条自检不是装饰：它把"算完之后再验一次兑现日"固化成运行时断言，算错就拒跑，
不会静默产出一张看起来正常的污染表。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY newfactor_ic_isolated.py                       # 写到本目录 newfactor_ic_isolated.csv
    $PY newfactor_ic_isolated.py --skip-redundancy     # 只算 IC（快）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

MODEL = Path('/autodl-fs/data/model')
SNAP = MODEL / 'trainingdata'
OLD337 = json.loads((MODEL / 'V62/features_old337.json').read_text())
META = json.loads((SNAP / 'meta.json').read_text())
ALL = META['columns']['features']
NEW = [f for f in ALL if f not in set(OLD337)]
LABEL = 'label_ret_5d'
H = 5                       # label_ret_5d 的 h
EVAL_START = '2025-07-01'   # 评价窗首个交易日（SPEC §4，不许改）
OUT_DIR = MODEL / 'research/20260921_s03_screening'


def trading_days(years=range(2018, 2027)) -> list[str]:
    days: set[str] = set()
    for y in years:
        p = SNAP / f'target/year={y}/data.parquet'
        if p.exists():
            days |= set(pq.read_table(p, columns=['trade_date']).column(0).to_pylist())
    return sorted(days)


def isolated_cutoff(days: list[str]) -> str:
    """最后一个**标签兑现日严格早于评价窗**的信号日。

    ★ 注意是 `first − h − 2`，不是 `first − h − 1`（**差一天就是越界**）。
    `V16/model.py::splits()` 的判据是：

        cutoff = first - horizon - 1
        eligible = ix[:cutoff]                      # 最后一个可训练样本索引 = first−h−2
        assert max(train.max(), valid.max()) + horizon + 1 < first      # ★ 严格小于

    ⇒ 最后一个安全样本的兑现日 = (first−h−2) + h + 1 = `first − 1`，**严格早于** first。
    我第一版按 `first − (h+1)` 退，自检当场就炸了（兑现日正好落在 first 那一天）——
    这正是 `_assert_isolated()` 存在的意义。h=5、first=2025-07-01 时：

        first − h − 1 = 2025-06-23 → 兑现日 2025-07-01  ✘（落进评价窗）
        first − h − 2 = 2025-06-20 → 兑现日 2025-06-30  ✔
    """
    j = int(np.searchsorted(days, EVAL_START))
    if days[j] != EVAL_START:
        raise SystemExit(f'✘ {EVAL_START} 不在交易日历里，无法定位评价窗起点')
    if j < H + 2:
        raise SystemExit('✘ 日历太短，退不出 h+2 个交易日')
    return days[j - (H + 2)]


def _assert_isolated(days: list[str], cutoff: str) -> None:
    """再验一遍：cutoff 的标签兑现日必须严格早于评价窗起点。"""
    j = days.index(cutoff)
    realize = days[j + H + 1]           # = T+1+h，T+1 是下一个交易日
    if not realize < EVAL_START:
        raise SystemExit(f'✘ 标签隔离失败：cutoff={cutoff} 的兑现日 {realize} '
                         f'不早于评价窗起点 {EVAL_START}')
    print(f'✔ 标签隔离自检通过：cutoff={cutoff} → 兑现日={realize} < {EVAL_START}')


def year_ic(year: int, cols: list[str], cutoff: str) -> dict:
    f = pq.read_table(SNAP / f'factors/year={year}/data.parquet',
                      columns=['trade_date', 'stock_code'] + cols).to_pandas()
    t = pq.read_table(SNAP / f'target/year={year}/data.parquet',
                      columns=['trade_date', 'stock_code', LABEL]).to_pandas()
    df = f.merge(t, on=['trade_date', 'stock_code'], how='left')
    df = df[df['trade_date'] <= cutoff]          # ★ 现在是**安全**的截断
    out: dict[str, list[float]] = {c: [] for c in cols}
    for day, g in df.groupby('trade_date', sort=True):
        y = g[LABEL].to_numpy(np.float64)
        ok_y = np.isfinite(y)
        if ok_y.sum() < 50:
            continue
        for c in cols:
            x = g[c].to_numpy(np.float64)
            m = ok_y & np.isfinite(x)
            if m.sum() < 50:
                continue
            xr = x[m]
            if np.all(xr == xr[0]):
                out[c].append(0.0)
                continue
            out[c].append(float(np.corrcoef(rankdata(xr), rankdata(y[m]))[0, 1]))
    del f, t, df
    return {c: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0., len(v))
            for c, v in out.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='newfactor_ic_isolated.csv')
    ap.add_argument('--skip-redundancy', action='store_true')
    ap.add_argument('--redundancy-sample-days', type=int, default=250)
    args = ap.parse_args()

    days = trading_days()
    cutoff = isolated_cutoff(days)
    _assert_isolated(days, cutoff)
    print(f'窗口：2018-01-02 ~ {cutoff}（原脚本用的是 2025-06-30，越界 {H + 1} 个交易日）')

    rows = []
    for year in range(2018, 2026):
        r = year_ic(year, NEW, cutoff)
        for c, (m, s, n) in r.items():
            rows.append(dict(factor=c, year=year, RankIC=m,
                             ICIR=(m / s if s else 0.0), days=n))
        print(f'{year} 完成：{len(r)} 个因子', flush=True)
    per_year = pd.DataFrame(rows)
    agg = per_year.groupby('factor').apply(
        lambda g: pd.Series(dict(
            ic_mean=g['RankIC'].mean(),
            icir=(g['RankIC'].mean() / g['RankIC'].std(ddof=1))
            if len(g) > 1 and g['RankIC'].std(ddof=1) else 0.0,
            ic_pos_years=int((g['RankIC'] > 0).sum()), years=len(g))),
        include_groups=False).reset_index()

    if not args.skip_redundancy:
        days_ix: list[str] = []
        for year in range(2020, 2025):
            d = pq.read_table(SNAP / f'factors/year={year}/data.parquet', columns=['trade_date'])
            days_ix.extend(sorted(set(d.column(0).to_pylist()))[::40])
        days_ix = days_ix[:args.redundancy_sample_days]
        by_year: dict[str, list[str]] = {}
        for d in days_ix:
            by_year.setdefault(d[:4], []).append(d)
        max_rho: dict[str, list[float]] = {c: [] for c in NEW}
        for year, ds in by_year.items():
            f = pq.read_table(SNAP / f'factors/year={year}/data.parquet',
                              columns=['trade_date'] + OLD337 + NEW).to_pandas()
            f = f[f['trade_date'].isin(ds)]
            for _day, g in f.groupby('trade_date'):
                old = g[OLD337].to_numpy(np.float64)
                new = g[NEW].to_numpy(np.float64)

                def rk(a):
                    a = np.where(np.isfinite(a), a, np.nan)
                    out = np.empty_like(a)
                    for j in range(a.shape[1]):
                        col = a[:, j]
                        m = np.isfinite(col)
                        out[:, j] = 0.5
                        if m.sum() > 10 and not np.all(col[m] == col[m][0]):
                            out[m, j] = (rankdata(col[m]) - .5) / m.sum()
                    return out

                ro = rk(old); rn = rk(new)
                ro = ro - ro.mean(0); rn = rn - rn.mean(0)
                den = np.sqrt((ro ** 2).sum(0))[:, None] * np.sqrt((rn ** 2).sum(0))[None, :]
                with np.errstate(invalid='ignore', divide='ignore'):
                    rho = np.abs((ro.T @ rn) / den)
                rho = np.nan_to_num(rho)
                for j, c in enumerate(NEW):
                    max_rho[c].append(float(rho[j].max()))
            del f
            print(f'冗余 {year} 完成', flush=True)
        agg['max_abs_rho_vs_old'] = agg['factor'].map(
            {c: float(np.mean(v)) for c, v in max_rho.items()})

    agg = agg.sort_values('ic_mean', key=lambda s: -s.abs())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / args.out
    agg.to_csv(out, index=False)
    (OUT_DIR / 'run_meta.json').write_text(json.dumps(
        dict(eval_start=EVAL_START, h=H, cutoff=cutoff, n_new=len(NEW),
             source='fixed copy of ../20260921_new_factors/newfactor_ic.py',
             original_cutoff='2025-06-30', isolated=True), ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(f'\n已写 {out}（{len(agg)} 个新因子）· cutoff={cutoff}')
    print(agg.head(25).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
