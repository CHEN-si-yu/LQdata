# -*- coding: utf-8 -*-
"""量级跳变护栏：逐列比"同日截面中位数 / 前一日截面中位数"，超过阈值就报。

为什么需要它（这不是假想）：2026-09-01 上游 `stock_finance` 整表（5,546 行）把
`total_share` 的单位从「股」切到「万」（比值 1.00009e−4），次日恢复。本工程自算市值，
于是 `bp` 0.58→2.157e4、`log_mv` 逐股 −9.194、`total_leverage_ratio` 0.039→387.9；
三个检测器取并集命中 **31 个因子**，其中 **18 个的 `rank`（模型可见列）被打破**，
**4 个滚动因子的污染一直持续到数据末日**。

★ 而 `main.py check` 当时是**全绿**的 —— `fea/validation.py` 只查格式（主键唯一、
  日期在轴内、不是 ±inf、行数对得上），**没有"值应该在什么量级"这一维**。
  所以"格式全绿"不等于"值没被换单位"。这条护栏补的就是这一维。

判据故意做到最钝（就是要钝）：只看**逐日截面中位数**，且**两个条件同时成立**才报：
  ① **倍数**：|log10(v_T / v_{T-1})| > log10(ratio)（默认 10 倍）；
  ② **离群**：这次跳幅 ≥ 窗口内逐日跳幅中位数的 `z_min` 倍（默认 20 倍）。
只满足①不报 —— 像指数的 `change`/`pct_chg` 这类**中位数本就接近 0** 的列，
逐日比值天天能到十几倍，但它的日常跳幅本来就那么大，不是事故。真事故（整表换单位）
会让**倍数**和**离群**同时爆表：中位数整体平移几个数量级，而日常波动远小于此。

用法：
  python guard_magnitude.py                 # 扫数据层全部日频表的数值列，最近 N 天
  python guard_magnitude.py --days 8 --ratio 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = Path('/root/autodl-fs/datadownload/data')

#: 表内哪一列是日期。默认 trade_date，几张非行情表用别的名字。
DATE_CANDIDATES = ('trade_date', 'ann_date', 'date', 'end_date')


def table_days(path):
    """读一张分年表的 (日期, 各数值列)。分年表取最后一年的分区。"""
    if path.is_dir():
        parts = sorted(path.glob('year=*'))
        if not parts:
            return None
        path = parts[-1] / 'data.parquet'
    if not path.is_file():
        return None
    try:
        df = pq.read_table(path).to_pandas()
    except Exception:
        return None
    col = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    if col is None:
        return None
    df[col] = df[col].astype(str).str.slice(0, 10)
    return df, col


def scan(days_window, ratio, z_min=20.0, min_rows=20, row_change_max=1.6):
    hits, scanned = [], 0
    for d in sorted(p for p in DATA.iterdir() if p.is_dir()):
        got = table_days(d)
        if not got:
            continue
        df, col = got
        got_days = sorted(u for u in df[col].unique() if u and u[0].isdigit())[-days_window:]
        if len(got_days) < 2:
            continue
        scanned += 1
        num = [c for c in df.columns
               if c != col and pd.api.types.is_numeric_dtype(df[c])]
        sub = df[df[col].isin(got_days)]
        med = sub.groupby(col)[num].median(numeric_only=True)
        cnt = sub.groupby(col)[num].count()
        med = med.reindex(got_days)
        cnt = cnt.reindex(got_days)
        for c in num:
            v = med[c].to_numpy(dtype='float64')
            k = cnt[c].to_numpy(dtype='float64')
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.abs(np.log10(np.abs(v[1:]) / np.abs(v[:-1])))
            # ② 离群：这次跳幅 ≥ 窗口内逐日跳幅中位数的 z_min 倍。
            #   ★ 只判 ① 会把"中位数本就接近 0 的列"全报成事故（实测 400 处噪声）：
            #     指数的 change/pct_chg 天天十几倍，但那是它的日常波动。
            #   ★ 另加最小样本数：季度表某天只来 1 行时，"中位数"就是那一行，毫无意义
            #     （实测 stock_balancesheet.total_share 被这样误报成 49.8× 事故）。
            dv = np.abs(np.diff(v))
            base = np.nanmedian(dv)
            z = dv / base if (np.isfinite(base) and base > 0) else np.full_like(dv, np.inf)
            # ③ 样本数不许突变：截面中位数会因**构成变化**而平移
            #   （实测 stock_pledge_stat 2026-04-30 行数 2247→4193，那些两天都在的股票
            #     比值恰好 = 1.0，纯粹是厂商多返回了一批 —— 报出来就是假阳性）。
            ratio_k = np.maximum(k[1:], k[:-1]) / np.maximum(1.0, np.minimum(k[1:], k[:-1]))
            bad = np.flatnonzero(np.isfinite(r) & (r > np.log10(ratio))
                                 & (z >= z_min) & (k[1:] >= min_rows) & (k[:-1] >= min_rows)
                                 & (ratio_k <= row_change_max))
            for i in bad:
                hits.append((d.name, c, got_days[i], v[i], got_days[i + 1], v[i + 1], 10 ** r[i]))
    return scanned, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=8, help='每张表看最近多少个交易日')
    ap.add_argument('--ratio', type=float, default=10.0, help='报错阈值：单日截面中位数比值')
    ap.add_argument('--z-min', type=float, default=20.0, help='跳幅至少是窗口内日常跳幅中位数的多少倍')
    ap.add_argument('--min-rows', type=int, default=20, help='两天的样本数都要 ≥ 这个值，避免单行中位数误报')
    ap.add_argument('--row-change-max', type=float, default=1.6, help='两天行数之比超过它就不判（构成变化会平移中位数）')
    a = ap.parse_args()
    n, hits = scan(a.days, a.ratio, a.z_min, a.min_rows, a.row_change_max)
    print(f'扫了 {n} 张表（各取最近 {a.days} 个交易日），判据 = 倍数 > {a.ratio:g}× '
          f'且 跳幅 ≥ 日常 {a.z_min:g} 倍 且 两天样本数 ≥ {a.min_rows}；命中 {len(hits)} 处')
    if hits:
        print(f'\n{"表":<34}{"列":<26}{"前一日":<12}{"当日":<12}{"倍数":>10}')
        print('-' * 100)
        for t, c, d0, v0, d1, v1, rat in hits:
            print(f'{t:<34}{c:<26}{d0:<12}{d1:<12}{rat:>10.1f}')
    return 1 if hits else 0


if __name__ == '__main__':
    sys.exit(main())
