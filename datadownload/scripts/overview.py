#!/usr/bin/env python
"""已落地数据总览：每个数据集的频率与时间范围。

为什么单独写一个：用户经常要一眼看清「盘上现在有多少种数据、各是什么频率、覆盖到哪天」，
而 DATA_CATALOG.md 是逐字段的字典、catalog.py 跑得慢，都不适合回答这个问题。

实现上刻意不读数据本体：优先用 parquet 的行组统计（row group statistics）取 min/max，
15G 数据秒级扫完；只有统计缺失时才回退到读那一列。频率则靠抽样若干行组、
取去重后的相邻时间戳间隔众数来判断，避免为了数天数把整表拉进来。

用法：
    python scripts/overview.py                # 全量，输出 markdown 表
    python scripts/overview.py --json         # 输出 JSON（给别的工具吃）
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1] / "data"

# 常见日期列，按优先级排；trade_date/trade_time 在最前是因为盘上绝大多数表用它俩
DATE_PREFERENCE = [
    "trade_date", "trade_time", "date", "datetime", "end_date", "ann_date",
    "report_date", "list_date", "in_date", "out_date", "change_date",
    "pub_date", "notice_date", "stat_date", "record_date", "float_date",
]

# 抽样上限：一个数据集最多读这么多行的时间列来判断频率，够用且内存有界
MAX_SAMPLE_ROWS = 1_500_000
MAX_SAMPLE_GROUPS = 24


def pick_date_col(names):
    lower = {n.lower(): n for n in names}
    for p in DATE_PREFERENCE:
        if p in lower:
            return lower[p]
    for n in names:
        ln = n.lower()
        if "date" in ln or "time" in ln:
            return n
    return None


def norm(v):
    """把各种写法的日期/时间统一成可比较的字符串。

    盘上同一个字段的格式并不统一（实测 stock_limit_up 就有 '14:02:36' 和 95947 混用），
    所以这里统一补零补分隔符，否则相邻间隔算出来是乱的。
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        s = str(int(v))
        if len(s) == 8:            # 20200101
            return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        if len(s) == 14:           # 20200102093500
            return f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:]}"
        if len(s) == 6:            # 093500
            return f"{s[:2]}:{s[2:4]}:{s[4:]}"
        if len(s) == 5:            # 95947 -> 09:59:47
            return f"0{s[0]}:{s[1:3]}:{s[3:]}"
        return s
    return str(v)


def parse_seconds(s):
    """把归一化后的字符串解析成「自纪元起的秒」，解析不了返回 None。"""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            d = datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
        # 纯时间（没有日期）用固定基准日，只用来算间隔
        if fmt == "%H:%M:%S":
            return d.hour * 3600 + d.minute * 60 + d.second
        return int(d.timestamp())
    return None


def human_freq(sec):
    """把相邻间隔秒数翻译成人话。"""
    if sec is None:
        return "?"
    if sec < 60:
        return f"{sec}秒"
    if sec % 86400 == 0:
        n = sec // 86400
        return "日线" if n == 1 else (f"约{n}天" if n < 27 else f"约{n // 30}个月")
    if sec % 3600 == 0:
        return f"{sec // 3600}小时"
    if sec % 60 == 0:
        return f"{sec // 60}分钟"
    return f"{sec}秒"


def scan(d, verbose=False):
    files = sorted(d.rglob("*.parquet"))
    info = {
        "dataset": d.name,
        "files": len(files),
        "rows": 0,
        "col": None,
        "min": None,
        "max": None,
        "freq": None,
        "layout": "flat",
        "note": "",
    }
    if not files:
        info["note"] = "空目录"
        return info

    # 分区形式：只是给人看的标签，取自路径里的第一个 'k=v' 片段
    names = {p.name for p in d.iterdir()}
    if any(n.startswith("year=") for n in names):
        info["layout"] = "按年分区"
    elif any(n.startswith("date=") for n in names):
        info["layout"] = "按日分区"
    elif len(files) == 1:
        info["layout"] = "单文件"
    else:
        info["layout"] = "多文件"

    # 列名取第一个文件的即可（同数据集 schema 一致）
    schema = pq.ParquetFile(files[0]).schema_arrow
    col = pick_date_col(schema.names)
    info["col"] = col
    if col is None:
        info["note"] = "无日期列（快照表）"
        for fp in files:
            info["rows"] += pq.ParquetFile(fp).metadata.num_rows
        return info

    # ---- 第一遍：行组统计拿全局 min/max（不读数据本体）----
    lo = hi = None
    groups_per_file = []   # [(file, [row_group_idx,...])]
    stats_missing = False
    for fp in files:
        pf = pq.ParquetFile(fp)
        md = pf.metadata
        info["rows"] += md.num_rows
        idxs = []
        for rgi in range(md.num_row_groups):
            rgm = md.row_group(rgi)
            idxs.append(rgi)
            for c in range(rgm.num_columns):
                cm = rgm.column(c)
                if cm.path_in_schema != col:
                    continue
                st = cm.statistics
                if st is None or not st.has_min_max:
                    stats_missing = True
                    continue
                a, b = norm(st.min), norm(st.max)
                if a is not None and (lo is None or a < lo):
                    lo = a
                if b is not None and (hi is None or b > hi):
                    hi = b
        groups_per_file.append((fp, idxs))

    # ---- 抽样读时间列，判断频率 ----
    all_groups = [(fp, rgi) for fp, idxs in groups_per_file
                  for rgi in idxs if rgi < len(idxs)]
    sampled = []
    if all_groups:
        step = max(1, len(all_groups) // MAX_SAMPLE_GROUPS)
        picks = all_groups[::step][:MAX_SAMPLE_GROUPS]
        seen = set()
        for fp, rgi in picks:
            if str(fp) + str(rgi) in seen:
                continue
            seen.add(str(fp) + str(rgi))
            pf = pq.ParquetFile(fp)
            # 单行组也可能超大，再截一刀保证内存有界
            t = pf.read_row_group(rgi, columns=[col])
            vals = t.column(0).to_pylist()[:MAX_SAMPLE_ROWS]
            sampled.extend(norm(v) for v in vals if v is not None)
            if len(sampled) >= MAX_SAMPLE_ROWS:
                break

    if sampled and (lo is None or hi is None):
        lo, hi = min(sampled), max(sampled)
    info["min"], info["max"] = lo, hi

    # 频率 = 去重后相邻间隔的众数；同时记一下单日去重条数供参考
    uniq = sorted(set(sampled))
    if len(uniq) >= 3:
        secs = []
        for a, b in zip(uniq, uniq[1:]):
            sa, sb = parse_seconds(a), parse_seconds(b)
            if sa is not None and sb is not None and sb > sa:
                secs.append(sb - sa)
        if secs:
            info["freq"] = human_freq(Counter(secs).most_common(1)[0][0])
    elif len(uniq) == 1:
        info["freq"] = "单一时点"
    if stats_missing and info["freq"] is None:
        info["note"] = "统计缺失且抽样不足"

    if verbose:
        print(f"  {d.name}: {info['rows']:,} 行 / {len(files)} 文件 / {info['freq']}",
              file=sys.stderr)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    out = []
    for i, d in enumerate(dirs, 1):
        if args.verbose:
            print(f"[{i}/{len(dirs)}] {d.name}", file=sys.stderr)
        out.append(scan(d))

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"| 数据集 | 频率 | 起始 | 结束 | 行数 | 分区 | 备注 |")
    print("|:--|:--|:--|:--|--:|:--|:--|")
    for r in out:
        print(f"| `{r['dataset']}` | {r['freq'] or '-'} | {r['min'] or '-'} | "
              f"{r['max'] or '-'} | {r['rows']:,} | {r['layout']} | {r['note']} |")
    total = sum(r["rows"] for r in out)
    print(f"\n共 {len(out)} 个数据集，{total:,} 行")


if __name__ == "__main__":
    main()
