#!/usr/bin/env python
"""全库完整性审计 —— **纯本地对账，零 API 请求**。

这是 QUANT_PLATFORM.md §16.4 / TASKS.md §六「任务 1」要求的固化脚本。
和 `verify.py` 的分工：verify.py 是**打服务端抽样比对**（要发 API、占限速额度、
只能在下载停的时候跑）；audit.py 是**纯本地**自洽性检查，随时能跑、秒级到分钟级。

审计四个维度（前两个是 §16.3 已做过一遍的，后两个是任务 1 要求补的）：

  A. partitions vs 磁盘         —— state 声称每个分区有多少行 / 日期范围，实际是多少
  B. coverage vs 实际交易日     —— **全部年份**（不只 2026）：coverage 说已覆盖的区间里，
                                   哪些交易日实际一行数据都没有  ← 这是最要命的一类
  C. suspect 复核               —— manifest 里挂着的可疑区间，现在补上了吗
  D. 日频表新鲜度               —— 每张日频表的 max_date 相对基准表晚几个交易日
                                   （任务 3 要固化的 delay，这里给出每日观测值）

**为什么 B 最重要**：`engine._plan_ranges` 按 coverage 找缺口 —— coverage 说"覆盖了"
它就不再请求。所以只要某轮抓取把"空响应"错记成了"已覆盖"，那个缺口就**永远不会被补**，
而且**全程无报错**。`index_daily` 缺 159 个交易日就是这样潜伏至今的（§16.3 问题 1）。

判定按 `conf/frequency.yaml` 的 freq 分类区分，**不会把不定期表误判成缺失**：
只有 `daily_full` 才做逐交易日缺口检查；`daily_sparse` 只做新鲜度；其余只对账 coverage。

    python scripts/audit.py                    # 全库审计
    python scripts/audit.py -d index_daily     # 只审一个
    python scripts/audit.py --json report.json # 额外输出机器可读结果
    python scripts/audit.py --deep             # 追加主键唯一性 + schema 漂移（慢）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date as _date
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATE = ROOT / "state"
DATA = ROOT / "data"


def d(s: str) -> _date:
    return _date.fromisoformat(str(s)[:10])


# ---------------------------------------------------------------- 基础读取
def load_freq() -> dict:
    with open(ROOT / "conf" / "frequency.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calendar_days() -> list[str]:
    """交易日历（来自 data/basic_calendar）。

    ⚠️ 这张表存的是**全部自然日**，靠 `is_open` 列区分（1=开市）。
    第一版忘了过滤 is_open，结果把周末全当成"缺失交易日"（stock_daily 虚报 2045 天）。
    """
    f = DATA / "basic_calendar" / "data.parquet"
    if not f.exists():
        return []
    t = pq.read_table(f, columns=["date", "is_open"])
    days, opens = t.column("date").to_pylist(), t.column("is_open").to_pylist()
    vals = [str(dv)[:10] for dv, op in zip(days, opens) if dv is not None and op == 1]
    return sorted(set(vals))


def partition_files(name: str) -> list[tuple[str, Path]]:
    """返回 [(分区键, 文件)]。年份分区键是 '2010'；单文件表键是 '0'。"""
    dpath = DATA / name
    if not dpath.is_dir():
        return []
    out = []
    for p in sorted(dpath.glob("year=*")):
        f = p / "data.parquet"
        if f.exists():
            out.append((p.name.split("=", 1)[1], f))
    flat = dpath / "data.parquet"
    if flat.exists():
        out.append(("0", flat))
    return out


def nrows(f: Path) -> int:
    """从 parquet 元数据取行数 —— 不读数据本体，大表也秒回。"""
    try:
        return pq.ParquetFile(f).metadata.num_rows
    except Exception:
        return -1


def date_stats(f: Path, col: str) -> tuple[list[str], str | None, str | None]:
    """只读日期列，返回 (去重后的日期, min, max)。"""
    try:
        t = pq.read_table(f, columns=[col])
    except Exception:
        return [], None, None
    vals = [str(x)[:10] for x in t.column(col).to_pylist() if x is not None]
    if not vals:
        return [], None, None
    uniq = sorted(set(vals))
    return uniq, uniq[0], uniq[-1]


# ---------------------------------------------------------------- 审计主体
class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, ds: str, check: str, level: str, msg: str, detail=None) -> None:
        self.rows.append({"dataset": ds, "check": check, "level": level,
                          "msg": msg, "detail": detail or []})

    def problems(self) -> list[dict]:
        return [r for r in self.rows if r["level"] in ("error", "warn")]


def audit_one(name: str, spec, fq: dict, cal: list[str], rep: Report,
              baseline_max: str | None, deep: bool) -> dict:
    """审计单个数据集，返回它的汇总信息。"""
    info = {"dataset": name, "freq": fq.get("freq", "?"), "rows": 0,
            "partitions": 0, "max_date": None, "delay_days": fq.get("delay_days", 0)}
    st = STATE / f"{name}.json"
    if not st.exists():
        rep.add(name, "A", "error", "state 文件不存在（但 spec 是启用的）")
        return info
    man = json.loads(st.read_text(encoding="utf-8"))
    files = partition_files(name)
    if not files:
        rep.add(name, "A", "error", "data/ 下没有任何 parquet")
        return info

    info["partitions"] = len(files)
    col = fq.get("date_field") or spec.date_field
    part_claim = man.get("partitions") or {}

    # ---- A. partitions 声明 vs 磁盘实际 ----
    disk_rows_total = 0
    actual_dates: list[str] = []
    for key, f in files:
        n_disk = nrows(f)
        disk_rows_total += max(n_disk, 0)
        claim = part_claim.get(key)
        if claim is None:
            rep.add(name, "A", "warn", f"分区 {key} 在磁盘上存在但 state 里没有记录")
            continue
        n_claim = int(claim.get("rows", 0))
        if n_disk != n_claim:
            rep.add(name, "A", "error",
                    f"分区 {key} 行数不符：state 说 {n_claim:,}，磁盘实际 {n_disk:,}",
                    {"partition": key, "claimed": n_claim, "actual": n_disk})
        # 日期范围也要对
        if fq.get("freq") not in ("snapshot",) and col:
            uniq, mn, mx = date_stats(f, col)
            actual_dates.extend(uniq)
            c_mn, c_mx = claim.get("min_date"), claim.get("max_date")
            if mn and c_mn and str(mn) < str(c_mn):
                rep.add(name, "A", "warn", f"分区 {key} 实际最早 {mn} 早于 state 记录的 {c_mn}")
            if mx and c_mx and str(mx) != str(c_mx):
                rep.add(name, "A", "error",
                        f"分区 {key} 最晚日期不符：state 说 {c_mx}，磁盘实际 {mx}")
    for key in part_claim:
        if key not in {k for k, _ in files} and key != "0":
            rep.add(name, "A", "error", f"state 声明了分区 {key} 但磁盘上没有对应文件")

    info["rows"] = disk_rows_total
    info["max_date"] = max(actual_dates) if actual_dates else None

    # ---- B. coverage vs 实际交易日（全部年份）----
    coverage = man.get("coverage") or []
    # 有 delay 的表，最后 delay_days 个交易日**上游还没发布**，不算缺失
    # （否则 stock_margin_detail 这类"滞后 1 个交易日"的表每天都会被误报）
    exempt: set[str] = set()
    lag0 = fq.get("delay_days", 0) or 0
    if lag0 and baseline_max and cal:
        upto = [x for x in cal if x <= baseline_max]
        exempt = set(upto[-lag0:])
    if fq.get("freq") == "daily_full" and coverage and cal and actual_dates:
        have = set(actual_dates)
        missing: list[str] = []
        for a, b in coverage:
            for day in cal:
                if a <= day <= b and day not in have and day not in exempt:
                    missing.append(day)
        if missing:
            rep.add(name, "B", "error",
                    f"coverage 声称已覆盖、但实际无数据的交易日共 {len(missing)} 天",
                    {"missing_days": missing[:40], "total_missing": len(missing),
                     "exempt_last_n": lag0})

    # ---- B2. 日期连续性（**不依赖 coverage**）----
    # ★ 为什么要另做这一道：`per_entity` 模式的表（如 index_ths_daily）根本不用 coverage，
    #   走的是 done.entities —— 只查 coverage 会把它们整个漏掉。
    #   实测 index_ths_daily 缺 2026-09-08 / 09-09 两天，第一版审计就是这么漏过去的。
    #   这里改成直接看数据本身：在 [最早, 最新] 之间的交易日里，哪些一天数据都没有。
    if fq.get("freq") == "daily_full" and cal and actual_dates:
        have = set(actual_dates)
        lo, hi = min(actual_dates), max(actual_dates)
        gaps = [x for x in cal if lo <= x <= hi and x not in have and x not in exempt]
        if gaps:
            rep.add(name, "B2", "error",
                    f"数据区间内有 {len(gaps)} 个交易日完全没有数据（不依赖 coverage 查出）",
                    {"missing_days": gaps[:40], "total_missing": len(gaps)})
    elif fq.get("freq") == "daily_sparse" and coverage and cal and actual_dates:
        # 稀疏表每个交易日**不保证**有行，所以只查"coverage 覆盖了但整段一天都没有"的极端情况
        have = set(actual_dates)
        span_days = [x for x in cal if x >= min(actual_dates)]
        if span_days:
            covered_any = sum(1 for x in span_days if x in have)
            if covered_any == 0:
                rep.add(name, "B", "error", "实际有数据的天数为 0，但 coverage 非空")

    # ---- C. suspect 复核 ----
    sus = man.get("suspect") or {}
    if sus:
        have = set(actual_dates)
        still = []
        for rng, cnt in sus.items():
            parts = str(rng).split("~")
            if len(parts) != 2:
                continue
            a, b = parts
            got = sum(1 for x in have if a <= x <= b)
            if got == 0:
                still.append({"range": rng, "retries": cnt})
        if still:
            rep.add(name, "C", "warn",
                    f"suspect 里仍有 {len(still)} 个区间一条数据都没有（共 {len(sus)} 条记录）",
                    {"outstanding": still[:20], "total": len(still)})

    # ---- D. 日频表新鲜度（任务 3 的 delay 观测）----
    if fq.get("freq") in ("daily_full", "daily_sparse") and info["max_date"] and baseline_max:
        lag = len([x for x in cal if info["max_date"] < x <= baseline_max])
        info["observed_lag"] = lag
        expect = fq.get("delay_days", 0)
        if lag != expect:
            lvl = "warn" if lag > expect else "error"
            rep.add(name, "D", lvl,
                    f"滞后 {lag} 个交易日（frequency.yaml 声明 {expect}）",
                    {"observed_lag": lag, "declared": expect})

    # ---- E. 深检：主键唯一性 + schema 漂移 ----
    if deep:
        schema_sets = defaultdict(list)
        for key, f in files:
            try:
                names = tuple(pq.ParquetFile(f).schema_arrow.names)
            except Exception:
                continue
            schema_sets[names].append(key)
        if len(schema_sets) > 1:
            groups = [list(v) for v in schema_sets.values()]
            rep.add(name, "E", "warn",
                    f"不同分区的列集合不一致，有 {len(schema_sets)} 种 schema",
                    {"groups": [g[:6] for g in groups], "n_groups": len(groups)})
        if spec.keys:
            for key, f in files:
                try:
                    t = pq.read_table(f, columns=[c for c in spec.keys])
                except Exception:
                    continue
                n = t.num_rows
                if n == 0:
                    continue
                seen = len(set(zip(*(t.column(c).to_pylist() for c in spec.keys))))
                if seen != n:
                    rep.add(name, "E", "error",
                            f"分区 {key} 主键不唯一：{n:,} 行只有 {seen:,} 个唯一主键",
                            {"partition": key, "rows": n, "unique": seen})
                    break

    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", action="append", help="只审指定数据集")
    ap.add_argument("--json", help="把机器可读结果写到这个路径")
    ap.add_argument("--deep", action="store_true", help="追加主键唯一性 + schema 漂移检查（慢）")
    args = ap.parse_args()

    from lingqi import spec as S
    from main import ordered_specs

    fqc = load_freq()
    ds_cfg = fqc["datasets"]
    cal = calendar_days()
    print(f"交易日历：{len(cal)} 天（{cal[0]} ~ {cal[-1]}）" if cal else "⚠ 交易日历为空！")
    print()

    specs = [s for s in ordered_specs(None) if s.enabled]
    if args.dataset:
        specs = [s for s in specs if s.name in set(args.dataset)]

    # 先定基准（stock_daily）的 max_date，供新鲜度检查用
    base = fqc.get("baseline", "stock_daily")
    base_files = partition_files(base)
    baseline_max = None
    if base_files:
        col = ds_cfg.get(base, {}).get("date_field", "trade_date")
        ds_all = []
        for _, f in base_files:
            u, _, _ = date_stats(f, col)
            ds_all.extend(u)
        baseline_max = max(ds_all) if ds_all else None
    print(f"基准表 {base} 的最新日期：{baseline_max}")
    print()

    rep = Report()
    infos = []
    for i, spec in enumerate(specs, 1):
        name = spec.name
        if name not in ds_cfg:
            rep.add(name, "-", "warn", "frequency.yaml 里没有它的分类（新数据集？）")
            fq = {"freq": "?", "date_field": spec.date_field, "delay_days": 0}
        else:
            fq = ds_cfg[name]
        infos.append(audit_one(name, spec, fq, cal, rep, baseline_max, args.deep))
        print(f"\r  审计中 {i}/{len(specs)} …", end="", flush=True)
    print("\r" + " " * 40 + "\r", end="")

    # ------------------------------------------------------------ 输出
    print("=" * 96)
    print(f"{'数据集':34s} {'频率':15s} {'行数':>13s} {'分区':>4s} {'最新':11s} {'滞后':>4s}")
    print("-" * 96)
    for x in infos:
        lag = x.get("observed_lag")
        fqy = x["freq"]
        if fqy in ("snapshot", "monthly", "quarterly", "irregular", "minute", "weekly_monthly", "?"):
            lag_s = "—"
        else:
            lag_s = str(lag) if lag is not None else "?"
        print(f"{x['dataset']:34s} {fqy:15s} {x['rows']:>13,} {x['partitions']:>4} "
              f"{str(x['max_date'] or '—'):11s} {lag_s:>4s}")
    print("-" * 96)
    print(f"合计 {sum(x['rows'] for x in infos):,} 行 / {len(infos)} 个启用数据集")
    print()

    probs = rep.problems()
    if not probs:
        print("✅ 全部检查通过：partitions 对账一致、coverage 无假覆盖、无悬空 suspect。")
    else:
        errs = [p for p in probs if p["level"] == "error"]
        warns = [p for p in probs if p["level"] == "warn"]
        print(f"❌ {len(errs)} 个错误 / ⚠️ {len(warns)} 个警告：")
        print()
        for p in probs:
            icon = "❌" if p["level"] == "error" else "⚠️"
            print(f"{icon} [{p['dataset']}] ({p['check']}) {p['msg']}")
            det = p["detail"]
            if isinstance(det, dict) and det.get("missing_days"):
                print(f"      缺失日示例：{', '.join(det['missing_days'][:12])} …")
            elif isinstance(det, dict) and det.get("outstanding"):
                print(f"      区间示例：{', '.join(o['range'] for o in det['outstanding'][:6])}")
    print()

    if args.json:
        out = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "calendar_days": len(cal),
            "baseline": base,
            "baseline_max": baseline_max,
            "datasets": infos,
            "findings": rep.rows,
            "n_error": len([p for p in probs if p["level"] == "error"]),
            "n_warn": len([p for p in probs if p["level"] == "warn"]),
        }
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"机器可读结果已写入 {args.json}")

    return 1 if any(p["level"] == "error" for p in probs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
