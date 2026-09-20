#!/usr/bin/env python
"""生成 / 复核**冻结股票池**名单（用户 2026-09-18 拍板）。

背景与动机
----------
原来的池子是**动态**的：`fea/universe.py::code_master()` 取主板代码全集（含已退市），
每只票按自己的 `[list_date, delist_date]` 进出。这带来一个每晚都会发作的问题：

  **一只新股上市 → 全库所有历史截面的行数 +1、rank / cs_zscore 全部平移
   → 单日 MD5 台账 2177 条全红**（2026-09-18 实测：`log0918_eod` 的比对，
   1772/1866 格剔除新股后逐位复现，另 94 格全部归因到"新股入池"）。

冻结成固定名单后，池子不再随新股/退市变动，历史值**永久稳定**。

用户拍板的口径（2026-09-18，四选一问题逐项确认）
--------------------------------------------
  ① 退市股：**剔除**，只要至今存续的
  ② 优质判据：**从未 ST** + 上市满 N 年（流动性门槛本轮未启用）
  ③ 规模：**上市 ≤ 2018-01-01**
  ④ 变更后**立即全量重建**

⇒ 最终 = 主板 ∩ 存续 ∩ 从未ST ∩ 上市≤2018-01-01

⚠️ 三个必须知道的代价（用户已确认接受，此处如实留档）
--------------------------------------------------
1. **幸存者偏差**：要求"至今存续"= 用"事后知道谁活下来"筛样本，
   历史上的差公司被系统性剔除 → **回测收益会被系统性高估**。
   原实现特意保留 289 只退市股正是为了避免这一点。这是本次改动**有意**付出的代价。
2. **"从未 ST" 是事后判据**：某票 2019 年 ST 过 → 它 2012–2018 的正常数据也被一起剔除。
   池子里因此只剩"整个样本期都没出过事"的公司。
3. **ST 判据的覆盖盲区**：采用 `stock_st_info`（官方口径，784 只），
   但它的覆盖从 **2016-08-09** 起 ⇒ 本口径的完整含义是
   **"2016-08-09 以来从未 ST"**。2012–2016-08 期间 ST 过、之后恢复且再未 ST 的票
   会被漏掉，现有数据无法判定。

   ⚠️⚠️ **绝不能用 `stock_daily.stock_name` 判 ST（2026-09-18 实测踩到）**：
   厂商把**历史 `stock_name` 全量改写成了当前名**。实例 `000004.SZ`：
   `stock_st_info` 说它 2022-05-06 起是 `ST国华`，而 `stock_daily` 里
   **全部 3724 行的名称都是 `国华退`**（当前名）—— 逐日名称**不是历史名称**。
   实测：名称判据只找到 284 只，官方口径 784 只，**差 502 只**。
   脚本仍打印名称口径，但**只作诊断、不参与筛选**。

用法
----
    python scripts/build_frozen_universe.py            # 生成 conf/universe_frozen.tsv
    python scripts/build_frozen_universe.py --check    # 只复核现有名单是否仍然成立（不写）
    python scripts/build_frozen_universe.py --force    # 覆盖已存在的名单（★ 慎用）

★ **名单一旦冻结就不该再改**：`fea/engine.py` 把内容哈希写进 `universe_fp`，
  改了会触发**全部因子的全量重建**（这是有意的，不是 bug）。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UPSTREAM = ROOT / ".." / "datadownload" / "data"
OUT = ROOT / "conf" / "universe_frozen.tsv"

MAIN_BOARD = ("600", "601", "603", "605", "000", "001", "002", "003")
LIST_CUTOFF = "2018-01-01"


def _read_all_years(name: str, columns: list[str]) -> pd.DataFrame:
    """把 `data/<name>/year=*/data.parquet` 全部读出来（平铺表则读单文件）。"""
    d = UPSTREAM / name
    parts = sorted(d.glob("year=*/data.parquet"))
    if not parts:
        f = d / "data.parquet"
        if not f.exists():
            raise FileNotFoundError(f"找不到上游数据集 {name}: {d}")
        return pd.read_parquet(f, columns=columns)
    return pd.concat([pd.read_parquet(p, columns=columns) for p in parts], ignore_index=True)


def _st_codes() -> tuple[set[str], set[str], list[str]]:
    """ST 代码集合。返回 `(官方口径, 名称口径, 异常名称样例)`。

    · **官方口径（采用）**：`stock_st_info` 的 `stock_code`（`type='ST'`，784 只）。
    · **名称口径（仅诊断）**：`stock_daily.stock_name` 以 `ST`/`*ST` 开头。
      ⚠️ **不可用于筛选** —— 厂商把历史名称改写成了当前名，它只反映"当前是 ST"。
    """
    st = _read_all_years("stock_st_info", ["stock_code", "type"])
    st_info = set(st.loc[st["type"].astype(str) == "ST", "stock_code"].astype(str))

    df = _read_all_years("stock_daily", ["stock_code", "stock_name"])
    nm = df["stock_name"].astype(str).str.upper().str.replace(" ", "", regex=False)
    hit = nm.str.startswith("ST") | nm.str.startswith("*ST")
    st_names = set(df.loc[hit, "stock_code"].astype(str).unique())
    odd = sorted(set(nm[nm.str.contains("ST", regex=False) & ~hit].unique()))
    return st_info, st_names, odd


def compute() -> pd.DataFrame:
    """按口径算出名单。返回按代码排序的 DataFrame（列：code / ld / dd）。"""
    sl = _read_all_years("stock_list", ["stock_code", "list_date", "delist_date"])
    sl = sl.drop_duplicates("stock_code").copy()
    sl["code"] = sl["stock_code"].astype(str)
    sl["ld"] = sl["list_date"].astype(str).str[:10]
    sl["dd"] = sl["delist_date"].astype(str).replace(
        {"None": "", "nan": "", "NaT": ""}).fillna("")

    st_info, st_names, odd = _st_codes()

    steps: list[tuple[str, int]] = []
    m = sl["code"].str.startswith(MAIN_BOARD)
    steps.append((f"主板前缀 {MAIN_BOARD}", int(m.sum())))
    m &= sl["dd"].str.len() < 10
    steps.append(("∩ 至今存续（delist_date 为空）", int(m.sum())))
    m &= sl["ld"] <= LIST_CUTOFF
    steps.append((f"∩ 上市日 ≤ {LIST_CUTOFF}", int(m.sum())))
    m &= ~sl["code"].isin(st_info)
    steps.append(("∩ 从未 ST（stock_st_info 官方口径）", int(m.sum())))

    keep = sl.loc[m, ["code", "ld", "dd"]].sort_values("code").reset_index(drop=True)

    print("筛选过程（逐级）：")
    prev = None
    for lbl, n in steps:
        drop = "" if prev is None else f"   −{prev - n}"
        print(f"  {lbl:46} {n:5}{drop}")
        prev = n
    print()
    print("ST 判据复核（名称口径**不参与筛选**，只作诊断）：")
    print(f"  ★ stock_st_info（采用） : {len(st_info):5} 只  （覆盖 2016-08-09 起，官方口径）")
    print(f"    stock_daily 名称      : {len(st_names):5} 只  （厂商改写历史名 → 只反映"
          f"『当前是 ST』，两者差 {len(st_info - st_names)} 只）")
    if odd:
        print(f"    ⚠️ 含 'ST' 但不以 ST/*ST 开头的名称: {odd[:6]}")
    print()
    return keep


def codes_of(path: Path) -> list[str]:
    return [ln.split("\t")[0] for ln in path.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]


def build(force: bool) -> int:
    keep = compute()
    if keep.empty:
        print("✘ 筛选结果为空，拒绝写出")
        return 1
    if OUT.exists() and not force:
        print(f"✘ {OUT} 已存在。冻结名单不该被反复重写；"
              f"确需重建请显式加 --force（并意识到会触发全量重建）。")
        return 1

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 冻结股票池（factor engineering · universe）★ 一经冻结不要修改",
        f"# generated_at={stamp}",
        "# 口径：主板 ∩ 至今存续 ∩ 从未ST(2016-08-09起) ∩ 上市日≤2018-01-01",
        f"# 只数={len(keep)}",
        "#",
        "# ⚠️ 代价（用户 2026-09-18 确认接受）：",
        "#   1) 要求『至今存续』→ 引入幸存者偏差（回测收益会被系统性高估）；",
        "#   2) 『从未ST』是事后判据（2019 年 ST 过的票，其 2012–2018 数据也一并剔除）；",
        "#   3) ST 判据 = stock_st_info（type='ST'，官方口径）；",
        "#      ⚠️ 其覆盖只到 2016-08-09 ⇒ 完整含义是『2016-08-09 以来从未 ST』。",
        "#      ⚠️ 不可改用 stock_daily 的 stock_name —— 厂商已把历史名称改成当前名。",
        "#",
        "# ★ 内容哈希会进 fea/engine.py 的 universe_fp：改它 = 触发全部因子全量重建。",
        "#   重新生成：python scripts/build_frozen_universe.py --force",
        "#",
        "# stock_code\tlist_date\tdelist_date",
    ]
    lines += [f"{r.code}\t{r.ld}\t{r.dd}" for r in keep.itertuples()]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest = hashlib.sha1("\n".join(codes_of(OUT)).encode()).hexdigest()
    print(f"✔ 已写出 {OUT}")
    print(f"  只数 = {len(keep)} · 代码清单 sha1 = {digest[:16]}")
    return 0


def check() -> int:
    if not OUT.exists():
        print(f"✘ {OUT} 不存在")
        return 1
    cur = codes_of(OUT)
    print(f"现有名单：{len(cur)} 只 · sha1 = "
          f"{hashlib.sha1(chr(10).join(cur).encode()).hexdigest()[:16]}\n")
    keep = compute()
    new = list(keep["code"])
    added = sorted(set(new) - set(cur))
    removed = sorted(set(cur) - set(new))
    print(f"用当前上游重算：{len(new)} 只 · 新增 {len(added)} · 剔除 {len(removed)}")
    if added:
        print(f"  ＋ 新增（如新股上市）—— 名单已冻结，**不会自动纳入**：{added[:10]}")
    if removed:
        print(f"  － 剔除（如新被 ST / 退市）—— 名单已冻结，**不会自动剔除**：{removed[:10]}")
    if not added and not removed:
        print("✅ 名单与当前数据完全一致")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="生成/复核冻结股票池")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的名单")
    ap.add_argument("--check", action="store_true", help="只复核，不写文件")
    a = ap.parse_args()
    return check() if a.check else build(a.force)


if __name__ == "__main__":
    raise SystemExit(main())
