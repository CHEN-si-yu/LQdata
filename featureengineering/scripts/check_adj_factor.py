"""`stock_adj_factor` 复权因子体检 —— 用户 2026-09-15 交办（每月跑一次）。

用户原话：「确认一下目前的复权 adj_factor 是合理正确的…将这个问题记录一下，
我们可以隔一个月的时候检查一次」。

## 为什么要专门体检这张表

`hfq(t) = 未复权价(t) × adj_factor(t)` 是所有价格类因子（动量/波动/标签…）的地基。
它有个隐蔽性质：**厂商可以回溯改写它** —— 改动某个历史日期的因子之后，所有吃 hfq 的
因子在那天的值就跟着变，而且不报任何错。所以不能只凭"数据在库里"就认定它对，
必须拿**另一个独立字段**（供应商的 `pct_chg`，按除权后基准价 `pre_close` 算的真实收益）对账。

## 三条判据（全部只用本地已落盘数据，零 API 请求）

1. **完整性**：主键 `(trade_date, stock_code)` 不重复；`adj_factor` 不缺失、不为 0/负；
2. **逐股单调性**：累计因子只增不减（除权只让它向上跳）。判据用**相对跌幅 > 0.1%** ——
   比 0.1% 更小的"下跌"是 float32 存储 + 4 位小数的舍入噪声（实测 12,619 格里 99% 都在这档）；
3. ★ **hfq 收益率 vs 交易所口径收益**（核心，判据同义于"因子跳变比例 = 1/参考价比例"）：
   `hfq_ret = (close×af)_t / (close×af)_{t-1} − 1` 必须等于 `pct_chg/100`。
   容差默认 **1e-3**：`pct_chg` 只有 2 位小数（舍入 5e-5）、`adj_factor` 4 位小数
   （小因子时相对误差 ~1e-4），两者叠加约 1.5e-4 —— 取 1e-3 留 6 倍余量。
   脚本会同时打印 2e-4 / 5e-4 / 1e-3 三档计数，便于看"是不是整体在漂"。
   实测基线（全历史 14.37M 格）：2e-4 → 12,651；5e-4 → 1,848；**1e-3 → 453**。
4. **除权日一致性**（附带）：价格侧有事件（`|pre_close/prev_close − 1| > 0.1%`）时，
   因子侧的跳变比例必须与之互补：**`r_fac × r_ref ≈ 1`**（容差 2%，留给低价股 0.01 元的
   参考价舍入）。实测全历史 47,560 个除权日里 99.4% 通过。

## 与"已知异常"清单的关系

`state/known_adj_anomalies.json` 存**基线**（首建于 2026-09-15，全历史 ~800 条）。
脚本会把基线**扣除**，只报**新增**：没有新增 → 退出码 0；有新增 → 退出码 1 并逐条打印。

用法：
    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/check_adj_factor.py                    # 全历史（每月一次）
    $PY scripts/check_adj_factor.py --years 2026 2026  # 只看某几年
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT.parent / "datadownload" / "data"
KNOWN = ROOT / "state" / "known_adj_anomalies.json"
MAIN_BOARD = ("600", "601", "603", "605", "000", "001", "002", "003")


def _read(ds: str, cols: list[str], years: list[int] | None) -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA / ds / "year=*" / "data.parquet")))
    if years:
        files = [f for f in files if int(Path(f).parent.name.split("=")[1]) in set(years)]
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f, columns=cols))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ⚠️ 跳过 {f}：{str(exc)[:60]}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def run(years: list[int] | None, tol: float):
    prob: list[str] = []
    d = _read("stock_daily", ["trade_date", "stock_code", "close", "pre_close", "pct_chg"], years)
    a = _read("stock_adj_factor", ["trade_date", "stock_code", "adj_factor"], years)
    for x in (d, a):
        x["td"] = x["trade_date"].astype(str).str[:10]
    print(f"  价格 {len(d):,} 行 / 因子 {len(a):,} 行")
    for nm, x in (("stock_daily", d), ("stock_adj_factor", a)):
        dup = len(x) - len(x.drop_duplicates(["td", "stock_code"]))
        if dup:
            prob.append(f"✘ {nm} 主键重复 {dup} 行")
    m = d.merge(a[["td", "stock_code", "adj_factor"]], on=["td", "stock_code"], how="left")
    bad_af = int(m["adj_factor"].isna().sum()) + int((m["adj_factor"] <= 0).sum())
    if bad_af:
        prob.append(f"✘ adj_factor 缺失或非正：{bad_af} 行")

    m = m.sort_values(["stock_code", "td"], kind="stable").reset_index(drop=True)
    g = m.groupby("stock_code", sort=False)
    m["af_chg"] = g["adj_factor"].diff()
    m["af_rel"] = m["af_chg"] / m["adj_factor"]
    m["prev_close"] = g["close"].shift(1)
    m["hfq_ret"] = ((m["close"] * m["adj_factor"])
                    .groupby(m["stock_code"], sort=False).pct_change())
    m["diff"] = (m["hfq_ret"] - m["pct_chg"] / 100.0).abs()
    m["r_ref"] = m["pre_close"] / m["prev_close"]
    m["r_fac"] = 1.0 + m["af_chg"] / (m["adj_factor"] - m["af_chg"])

    # ② 单调性（相对跌幅 > 0.1% 才算真违反）
    viol = m[m["af_rel"] < -1e-3]
    print(f"  单调性违反（相对跌幅>0.1%）：{len(viol)} 格 / {viol['stock_code'].nunique()} 只股")
    if len(viol):
        prob.append(f"⚠️ 单调性违反 {len(viol)} 格：{sorted(viol['stock_code'].unique())[:8]}")

    # ③ 核心：hfq vs 交易所口径
    g2 = m.dropna(subset=["diff"])
    print(f"  hfq 收益 vs pct_chg：{len(g2):,} 格；超 2e-4 的 {int((g2['diff']>2e-4).sum())}、"
          f"5e-4 的 {int((g2['diff']>5e-4).sum())}、{tol:g} 的 {int((g2['diff']>tol).sum())} 格")
    bad = m[m["diff"] > tol].copy()

    # ④ 除权日一致性 r_fac × r_ref ≈ 1
    ex = m[((m["r_ref"] - 1).abs() > 1e-3) & m["r_fac"].notna()].copy()
    ex["cons"] = (ex["r_fac"] * ex["r_ref"] - 1).abs()
    exbad = ex[ex["cons"] > 0.02]
    print(f"  除权日一致性：{len(ex):,} 个除权日，其中不满足 r_fac×r_ref≈1（>2%）的 {len(exbad)} 个")

    # ⑤ 跳变与厂商变更表对账（信息性）
    try:
        c = _read("stock_adj_factor_changes", ["stock_code", "date"], years)
        ck = set(zip(c["date"].astype(str).str[:10], c["stock_code"]))
        jumps = m[m["af_chg"] > 1e-9]
        miss = int(sum(1 for _, r in jumps.iterrows() if (r["td"], r["stock_code"]) not in ck))
        print(f"  跳变 {len(jumps):,} 次，变更表无记录 {miss} 次")
    except Exception:                                              # noqa: BLE001
        ck = set()

    # 分类（A/B/C/D）
    def kind(r) -> str:
        prc = pd.notna(r["r_ref"]) and abs(r["r_ref"] - 1) > 1e-3
        fj = r["af_chg"] > 1e-9
        if fj and not prc:
            return "A 因子跳了、价格没跳"
        if prc and not fj:
            return "B 价格跳了、因子没跳"
        return "C 两边都跳但比例不符"

    rows = []
    for _, r in bad.iterrows():
        rows.append({"stock_code": r["stock_code"], "date": r["td"], "kind": kind(r),
                     "hfq_minus_exchange": round(float(r["hfq_ret"] - r["pct_chg"] / 100), 6),
                     "in_change_table": (r["td"], r["stock_code"]) in ck,
                     "main_board": str(r["stock_code"]).startswith(MAIN_BOARD)})
    for _, r in viol.iterrows():
        rows.append({"stock_code": r["stock_code"], "date": r["td"],
                     "kind": "D 因子相对前值下跌（累计因子不该减）",
                     "hfq_minus_exchange": None, "af_rel": round(float(r["af_rel"]), 6),
                     "in_change_table": (r["td"], r["stock_code"]) in ck,
                     "main_board": str(r["stock_code"]).startswith(MAIN_BOARD)})
    return prob, pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, default=None, help="默认全历史")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out", default="", help="可选独立导出路径；默认记入 README.md")
    a = ap.parse_args()
    print(f"[{datetime.now():%H:%M:%S}] adj_factor 体检（年份：{a.years or '全历史'}，容差 {a.tol:g}）")
    prob, bad = run(a.years, a.tol)

    known = {}
    if KNOWN.exists():
        kj = json.loads(KNOWN.read_text(encoding="utf-8"))
        known = {(x["stock_code"], x["date"]) for x in kj.get("anomalies", [])}
        print(f"  已知基线：{len(known)} 条（{KNOWN.name}，建于 {kj.get('created_at','?')}）")
    new = bad[~bad.apply(lambda r: (r["stock_code"], r["date"]) in known, axis=1)] if len(bad) else bad

    lines = [f"# adj_factor 体检报告 · {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             f"- 范围：{a.years or '全历史'}；容差 {a.tol:g}（红线）",
             f"- 异常 {len(bad)} 个（基线内 {len(bad)-len(new)} / **新增 {len(new)}**）",
             f"- 结构性问题：{prob or '无'}", ""]
    if len(new):
        lines += ["## ★ 新增异常", "", "| 股票 | 日期 | 类型 | hfq−交易所 | 变更表 | 主板 |",
                  "|:--|:--|:--|--:|:--|:--|"]
        for _, r in new.sort_values(["kind", "stock_code"]).iterrows():
            v = r["hfq_minus_exchange"]
            lines.append(f"| {r['stock_code']} | {r['date']} | {r['kind']} | "
                         f"{'—' if pd.isna(v) else f'{v:+.4f}'} | "
                         f"{'有' if r['in_change_table'] else '**无**'} | {'是' if r['main_board'] else '否'} |")
    else:
        lines.append("## 新增异常：无 ✔（全部落在已知基线内，历史没有被悄悄改写）")
    txt = "\n".join(lines) + "\n"
    print("\n" + txt)
    if a.out:
        out = Path(a.out)
        if out.resolve() == (ROOT / "README.md").resolve():
            raise ValueError("不要通过 --out 覆盖总手册；省略该参数即可追加报告")
        out.write_text(txt, encoding="utf-8")
    else:
        from fea.documentation import append_report
        out = append_report(ROOT, "adj-check", "复权因子体检", txt)
    print(f"（报告已写入 {out}）")
    return 1 if (len(new) or prob) else 0


if __name__ == "__main__":
    raise SystemExit(main())
