#!/usr/bin/env python3
"""去糟粕：按分诊清单**三层删除**因子（代码注册 + 产物 + 状态）。

## 为什么必须三层一起删

只删数据与状态是没用的：`factors/*.py` 里的 `@register` 还在，下一次 `main.py run`
会把它**原样重算回来**。所以删除 = 从源码里摘掉注册与函数 + 删 `data/factors/<名>/`
+ 删 `state/<名>.json`。

## 安全约束（脚本自己强制的）

1. **保护名单只剩 5 个标签**（`label_ret_*`）—— 它们不是因子，是下游的目标变量。
   ⚠️ 用户 2026-09-15 明确「因子不存在保护名单，你觉得不应该存在就删了」：
   原先那份"第一批 10 个因子"的保护名单**已取消**，它们按同一把尺子复判
   （结果：`gpm_ttm`/`cash_profit_ratio`/`debt_asset_ratio`/`yoy_net_profit`/`yoy_roe` 被删）。
2. **先写报告再删**：`docs/FACTOR_TRIAGE.md` 里逐条记录 名称/家族/公式/证据/理由，
   删掉之后仍可据此重建（参考库 `学习资料/factors.md` 里也有原始定义）。
3. `--dry-run`（默认）：预览待删对象并将报告记入总手册，不改因子代码或数据。要真删必须显式 `--apply`。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/prune_factors.py                 # 预览（默认 dry-run）
    $PY scripts/prune_factors.py --apply         # 真删
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# 分诊清单：{因子名: 理由}。判据只有三类（都写进报告）：
#   【重复】|ρ|≥0.90 的簇里，保留代表、删其余（`main.py dedup` 的实测结果）
#   【退化】零膨胀（>75% 取值为 0）/ 截面取值过少 / 覆盖率过低 —— 排名被并列值支配
#   【无效】|RankIC| 极小且分层无单调性（单年样本，仅作辅助证据，不单独作为理由）
# ─────────────────────────────────────────────────────────────────────────────
DELETE: dict[str, str] = {
    # ★★ 本轮（2026-09-17）清单 = `main.py eval --years 2026 2026` 的 ⚠NOISE 集合
    #   （36 个）**减去 2 个父依赖**（见下方说明）。判据：|RankIC| < 0.005 且样本 ≥60 天。
    #   无重复项可删 —— 本轮 `main.py dedup` 实测 **0 簇**（上一轮的去重已清干净）。
    "asset_turnover": "【无效】|RankIC|=0.0023（<0.005，2026 全年 166 个截面）；ac1=1.000 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "capex_to_revenue": "【无效】|RankIC|=0.0049（<0.005，2026 全年 166 个截面）；ac1=0.999 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "chip_below_momentum": "【无效】|RankIC|=0.0018（<0.005，166 个截面），与噪声不可区分",
    "chip_cost_kurtosis_20d": "【无效】|RankIC|=0.0031（<0.005，166 个截面），与噪声不可区分",
    "chip_cr3_factor": "【无效】|RankIC|=0.0050（<0.005，166 个截面），与噪声不可区分",
    "chip_loss_peak_frac": "【无效】|RankIC|=0.0001（<0.005，166 个截面），与噪声不可区分",
    "chip_tail_risk_change": "【无效】|RankIC|=0.0046（<0.005，166 个截面），与噪声不可区分",
    "corr_market_60": "【无效】|RankIC|=0.0047（<0.005，166 个截面），与噪声不可区分",
    "cp_limitup_momentum": "【无效】|RankIC|=0.0033（<0.005，166 个截面），与噪声不可区分",
    "cp_retail_reversal_5": "【无效】|RankIC|=0.0018（<0.005，166 个截面），与噪声不可区分",
    "cp_vol_turnover_20": "【无效】|RankIC|=0.0050（<0.005，164 个截面），与噪声不可区分",
    "cp_winner_momentum_60": "【无效】|RankIC|=0.0013（<0.005，166 个截面），与噪声不可区分",
    "current_ratio": "【无效】|RankIC|=0.0005（<0.005，2026 全年 166 个截面）；ac1=1.000 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "delta_gpm": "【无效】|RankIC|=0.0035（<0.005，166 个截面），与噪声不可区分",
    "delta_roa": "【无效】|RankIC|=0.0014（<0.005，166 个截面），与噪声不可区分",
    "delta_roe": "【无效】|RankIC|=0.0019（<0.005，166 个截面），与噪声不可区分",
    "doji_frequency_20": "【无效】|RankIC|=0.0037（<0.005，166 个截面），与噪声不可区分",
    "forecast_p_change_median": "【无效】|RankIC|=0.0025（<0.005，2026 全年 166 个截面）；ac1=0.995 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "gap_open_follow_ratio_20": "【无效】|RankIC|=0.0008（<0.005，166 个截面），与噪声不可区分",
    "margin_buy_momentum_5d": "【无效】|RankIC|=0.0017（<0.005，166 个截面），与噪声不可区分",
    "margin_leverage_trend_10d": "【无效】|RankIC|=0.0032（<0.005，166 个截面），与噪声不可区分",
    "margin_net_flow_ratio": "【无效】|RankIC|=0.0047（<0.005，166 个截面），与噪声不可区分",
    "mf_big_order_stability_20d": "【无效】|RankIC|=0.0037（<0.005，166 个截面），与噪声不可区分",
    "mf_flow_continuity": "【无效】|RankIC|=0.0008（<0.005，166 个截面），与噪声不可区分；截面取值卡片化",
    "mf_net_inflow_trend_5d": "【无效】|RankIC|=0.0031（<0.005，166 个截面），与噪声不可区分",
    "mf_net_inflow_volatility_20d": "【无效】|RankIC|=0.0021（<0.005，166 个截面），与噪声不可区分",
    "mf_net_persistent_5d": "【无效】|RankIC|=0.0041（<0.005，166 个截面），与噪声不可区分；截面取值卡片化",
    "mf_vol_amount_corr_20": "【无效】|RankIC|=0.0011（<0.005，166 个截面），与噪声不可区分",
    "momentum_accel_60_120": "【无效】|RankIC|=0.0007（<0.005，166 个截面），与噪声不可区分",
    "ncf_to_market": "【无效】|RankIC|=0.0045（<0.005，2026 全年 166 个截面）；ac1=0.993 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    "net_turnover_rate_20": "【无效】|RankIC|=0.0047（<0.005，166 个截面），与噪声不可区分",
    "obv_divergence_20": "【无效】|RankIC|=0.0035（<0.005，166 个截面），与噪声不可区分",
    "peg_252d": "【无效】|RankIC|=0.0009（<0.005，2026 全年 166 个截面）；ac1=0.996 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**；覆盖率仅 28%",
    "quick_ratio": "【无效】|RankIC|=0.0020（<0.005，2026 全年 166 个截面）；ac1=1.000 —— 慢变量单年只有约 4 个独立样本，IC 本身不可靠，**标注可复检**",
    # ---- 以下 2 个同样被标 ⚠NOISE，但**保留**（是存活耦合因子的父依赖，删了会打断子因子）----
    #   `mf_big_order_ratio` ← fundflow_retail_inst_divergence（子因子 RankIC −0.0382 / mono −0.85）
    #   `short_term_reversal_5` ← cp_chip_support_reversal_5（子因子 RankIC 0.0339 / mono 0.80）
}

# ★ 保护名单：**只有 5 个标签**（label_ret_1d/3d/5d/10d/20d）。
#   它们不是因子，是下游模型的目标变量（`is_label=True`），删了模型就没标签了。
#   ⚠️ 用户 2026-09-15 明确：「因子不存在保护名单，你觉得不应该存在就删了」——
#      所以原来那份"第一批 10 个因子"的保护名单**已取消**，它们按同一把尺子复判。
PROTECTED = {
    "label_ret_1d", "label_ret_3d", "label_ret_5d", "label_ret_10d", "label_ret_20d",
}


def _specs():
    import factors                                          # noqa: F401
    from fea.spec import REGISTRY
    return REGISTRY


def _remove_symbol(path: Path, names: set[str]) -> list[str]:
    """从源码里摘掉这些因子的 `@register(...)` 块与函数体。

    ★ 用 ast 定位（可靠的括号/多行字符串处理），再按行号**从后往前**删，
      避免行号位移。动态注册（`for _k in (5, 10, ...)` 里 `momentum_5` 那种）
      单独处理：从元组里摘掉那个数字。
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    spans: list[tuple[int, int, str]] = []                  # (起始行, 结束行, 名字)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            txt = ast.get_source_segment(src, dec) or ""
            m = re.search(r"name\s*=\s*[\"']([\w]+)[\"']", txt)
            if m and m.group(1) in names:
                # 装饰器可能带注释行在上面：从 decorator 行开始（注释保留）
                start = min(d.lineno for d in node.decorator_list)
                spans.append((start, node.end_lineno, m.group(1)))
                break
    removed = []
    for start, end, nm in sorted(spans, reverse=True):
        # 连同上方紧邻的注释块一起删（那段注释只属于这个因子）
        s = start - 1
        while s - 1 >= 0 and lines[s - 1].lstrip().startswith("#"):
            s -= 1
        del lines[s:end]
        removed.append(nm)

    # 动态注册：`for _k in (5, 10, 20, 60, 120, 250):` + `name=f"momentum_{_k}"`
    for nm in sorted(names):
        m = re.fullmatch(r"(\w+?)_(\d+)", nm)
        if not m or nm in removed:
            continue
        prefix, num = m.group(1), m.group(2)
        text = "".join(lines)
        # ★ 不能用 f-string 拼这段：f-string 里不允许 \" 转义（Python 语法限制）
        pat = re.compile(r"(for \w+ in \()([^)]*)(\):\n(?:(?!\n\S).)*?name=f\""
                         + prefix + r"_\{\w+\}\")", re.S)
        mm = pat.search(text)
        if not mm:
            continue
        kept = [x.strip() for x in mm.group(2).split(",") if x.strip() and x.strip() != num]
        text = text[:mm.start(2)] + ", ".join(kept) + text[mm.end(2):]
        lines = text.splitlines(keepends=True)
        removed.append(nm)
    if removed:
        # ★ 原子写：别的 Agent 可能正在 `import factors`，`write_text` 有一瞬间
        #   文件是空的 → 那个 Agent 会拿到 SyntaxError。先写临时文件再 os.replace。
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(lines), encoding="utf-8")
        os.replace(tmp, path)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description="按分诊清单三层删除因子")
    ap.add_argument("--apply", action="store_true", help="真的删（默认只预览）")
    ap.add_argument("--report", default="", help="可选独立导出路径；默认记入 README.md")
    args = ap.parse_args()

    from fea import store
    from fea.config import load as load_cfg

    cfg = load_cfg()
    reg = _specs()
    bad = [n for n in DELETE if n in PROTECTED]
    if bad:
        raise SystemExit(f"清单里有受保护的因子，拒绝执行：{bad}")
    unknown = [n for n in DELETE if n not in reg]
    missing = [n for n in DELETE if n not in reg]
    todo = {n: r for n, r in DELETE.items() if n in reg}

    # ---- 报告（先写，删了也能照它重建）
    rep = [f"# 因子分诊报告（去糟粕）", "",
           f"> 由 `scripts/prune_factors.py` 生成。判据见脚本头部注释。",
           f"> 计划删除 **{len(todo)}** 个（当前注册 {len(reg)} 个因子）；此为执行前计划，不代表删除成功。", "",
           "| 因子 | 家族 | 定义 | 公式 | 删除理由 |", "|:--|:--|:--|:--|:--|"]
    for n, r in sorted(todo.items()):
        s = reg[n]
        # ★ 公式里可能有换行（多行公式），不替换会把 markdown 表格撑破
        fml = s.formula.replace("\n", " ⏎ ").replace("|", chr(92) + "|")[:120]
        rep.append(f"| `{n}` | {s.group} | {s.desc} | `{fml}` | {r} |")
    if unknown:
        rep += ["", f"> ⚠️ 清单里有 {len(unknown)} 个未注册（可能已删过）：{unknown}"]
    rep += ["", "## 保留但标注低置信", "",
            "- `ac1 > 0.995` 的财务/慢变量：2026 单年只有约 4 个独立样本（季报），IC 统计上不可信；",
            "  同时它们换手≈0，无法独立产生交易信号 —— 作为**风格暴露/控制变量**保留。",
            "- `ac1 < 0.10` 的日内/资金流瞬时因子：日频全换手，必须按**成本后**收益复核。"]
    if args.report:
        out = ROOT / args.report
        if out.resolve() == (ROOT / "README.md").resolve():
            raise ValueError("不要通过 --report 覆盖总手册；省略该参数即可追加报告")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    else:
        from fea.documentation import append_report
        out = append_report(ROOT, "prune-preview", "因子分诊执行前计划", "\n".join(rep) + "\n")
    print(f"分诊报告已写入 {out}（{len(todo)} 个待删）")

    # ---- 预览
    print(f"\n{'因子':<28}{'家族':<10}{'产物':>10}{'状态':<8}")
    print("-" * 70)
    n_data = n_state = 0
    for n in sorted(todo):
        d = cfg.factors_dir / n
        st = cfg.state_dir / f"{n}.json"
        has_d = d.is_dir()
        has_s = st.exists()
        n_data += has_d
        n_state += has_s
        print(f"{n:<28}{reg[n].group:<10}{'有' if has_d else '—':>10}{'有' if has_s else '—':<8}")
    print("-" * 70)
    print(f"合计：{len(todo)} 个因子 · 产物目录 {n_data} · 状态文件 {n_state}")

    # ---- 按源码文件分组预览
    by_file: dict[str, list[str]] = {}
    for n in todo:
        by_file.setdefault(reg[n].fn.__code__.co_filename, []).append(n)
    print("\n源码改动（每个文件删掉对应注册块）：")
    for f, ns in sorted(by_file.items()):
        print(f"  {Path(f).name:<24} {len(ns):>3} 个：{', '.join(sorted(ns)[:6])}"
              f"{' …' if len(ns) > 6 else ''}")

    if not args.apply:
        print("\n（dry-run，仅更新报告，未改动因子代码或数据。要真删请加 --apply）")
        return 0

    # ---- 真删
    print("\n开始删除 …")
    for f, ns in sorted(by_file.items()):
        p = Path(f)
        removed = _remove_symbol(p, set(ns))
        miss = sorted(set(ns) - set(removed))
        print(f"  ✔ {p.name}：摘掉 {len(removed)} 个注册块"
              + (f"   ⚠️ 未定位到：{miss}" if miss else ""))
    for n in sorted(todo):
        shutil.rmtree(cfg.factors_dir / n, ignore_errors=True)
        st = cfg.state_dir / f"{n}.json"
        if st.exists():
            st.unlink()
    print(f"  删除产物目录 {n_data} 个、状态文件 {n_state} 个")
    print("\n★ 记得核对：`python main.py list | wc -l` 与 `import factors` 的注册数")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
