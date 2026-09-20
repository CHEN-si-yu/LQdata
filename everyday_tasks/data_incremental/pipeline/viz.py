"""可视化（纯 stdlib：ASCII + 内联 SVG；环境里没有 matplotlib）。

最重要的一个是 `render_gate()` —— 用户可能盯着它等最多 4 小时，
所以要能一眼看出"还差哪几张表、卡在哪一天"。
"""
from __future__ import annotations

import json
from datetime import datetime

from .. import paths
from ..core.progress import bar, fmt_duration, is_tty


# ================================================================ 闸门视图
def render_gate(round_no: int, elapsed: float, budget: float, T: str | None,
                rows: list[dict], interval: float, note: str = "") -> str:
    """闸门实时视图。

    rows: [{"name","tier","delay","expected","actual","ready","note"}, ...]

    ★ 2026-09-15 深夜：「闸门只管日频」改成按档（wait 阻塞 / watch 观测 / skip 不参与）。
      · 进度条与统计**只数 wait 档** —— watch 档恒 ready，混进来会虚高。
      · watch 档单独一段列出，标 `·`，并把"落后于服务端"的那几行放在最前。
    """
    wait_rows = [r for r in rows if r.get("tier", "wait") == "wait"]
    watch_rows = [r for r in rows if r.get("tier") == "watch"]
    ok = sum(1 for r in wait_rows if r["ready"])
    total = len(wait_rows) or 1
    holes = [r for r in wait_rows if r.get("hole")]
    behind = [r for r in watch_rows if "落后于服务端" in (r.get("note") or "")]
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"  ⏳ 闸门 · 第 {round_no} 轮 · 已等 {fmt_duration(elapsed)} "
                 f"/ 预算 {fmt_duration(budget)}")
    lines.append(f"  上游最新交易日 T = {T or '（未探测到）'}"
                 + (f"   下一轮 {int(interval)}s 后再查" if ok < total else ""))
    lines.append("=" * 78)
    if note:
        lines.append(f"  ℹ️  {note}")
    lines.append(f"  {bar(ok / total, 32)}  {ok}/{total} 张等待档表已到齐"
                 + (f"（其中 {len(holes)} 张记为**已知厂商洞**，不阻塞："
                    f"{', '.join(r['name'] for r in holes)}）" if holes else ""))
    lines.append("")
    lines.append(f"  {'数据集':32} {'档':4} {'delay':>5} {'应到':11} {'服务端':11} 状态")
    lines.append("  " + "-" * 78)
    for r in sorted(wait_rows, key=lambda x: (x["ready"], x["name"])):
        # ★ 2026-09-17：`❓` = 探测失败（unknown，不阻塞但也没确认到齐）。
        #   与 `✔`（确认到齐）必须能一眼分开，否则"看不清"会被当成"已到齐"。
        mark = ("⏭" if r.get("hole") else
                "❓" if r.get("unknown") else
                "✔" if r["ready"] else "⏳")
        act = str(r.get("actual") or "—")
        extra = f"  {r['note']}" if r.get("note") and (not r["ready"] or r.get("hole")) else ""
        lines.append(f"  {r['name']:32} {'等':4} {r['delay']:>5} {r['expected']:11} {act:11} {mark}{extra}")
    if watch_rows:
        lines.append("")
        lines.append(f"  ── 观测档（{len(watch_rows)} 张，**不阻塞闸门**：月频/季频/不定期/周月线"
                     f"，判定见 registry.gate_tier）" + (f"，其中 {len(behind)} 张落后于服务端 ⚠️" if behind else "") + " ──")
        for r in sorted(watch_rows, key=lambda x: ("落后于服务端" not in (x.get("note") or ""), x["name"])):
            mark = "⚠️" if "落后于服务端" in (r.get("note") or "") else "·"
            act = str(r.get("actual") or "—")
            lines.append(f"  {r['name']:32} {'看':4} {r['delay']:>5} {'—':11} {act:11} {mark}  {r.get('note') or ''}")
    lines.append("=" * 78)
    return "\n".join(lines)


def print_gate(*args, **kw) -> None:
    text = render_gate(*args, **kw)
    if is_tty():
        import sys
        sys.stdout.write("\033[2J\033[H" + text + "\n")
        sys.stdout.flush()
    else:
        print(text, flush=True)


# ================================================================ 汇总表
def render_summary(report: dict) -> str:
    """跑完的汇总：按频率分组 + 一行结论。"""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 92)
    lines.append(f"  📋 每日增量汇总 · {report.get('finished_at','')} · "
                 f"T={report.get('T','?')} · 用时 {fmt_duration(report.get('seconds',0))} · "
                 f"请求 {report.get('requests',0):,}（重试 {report.get('retries',0)}）")
    lines.append("=" * 92)

    groups = report.get("groups") or {}
    for gname, items in groups.items():
        if not items:
            continue
        lines.append(f"  【{gname}】{len(items)} 个")
        lines.append(f"    {'数据集':32} {'新增行数':>12} {'请求':>6} {'耗时':>8}  状态")
        lines.append("    " + "-" * 76)
        for it in items:
            st = it.get("status", "✔")
            lines.append(f"    {it['name']:32} {it.get('rows',0):>12,} "
                         f"{it.get('requests',0):>6} {fmt_duration(it.get('seconds',0)):>8}  {st}")
        lines.append("")

    # ★ 厂商改历史（单日 MD5 台账）
    rev = report.get("revisions") or []
    if rev:
        lines.append(f"  🔴 检测到上游修改了历史数据（{len(rev)} 处）—— 单日 MD5 与台账不符：")
        for r in rev[:20]:
            cols = ", ".join(r.get("changed_cols") or []) or "(整行)"
            lines.append(f"    {r['dataset']:30} {r['date']}  行数 {r.get('old_rows','?')}→{r.get('new_rows','?')}  "
                         f"变化列: {cols}")
        if len(rev) > 20:
            lines.append(f"    … 另有 {len(rev)-20} 处")
        lines.append("")

    plans = report.get("plans") or []
    if plans:
        lines.append(f"  🧪 执行计划（dry-run，{len(plans)} 项）：")
        for p in plans[:40]:
            lines.append(f"    {p['name']:32} {p['note']}")
        if len(plans) > 40:
            lines.append(f"    … 另有 {len(plans)-40} 项")
        lines.append("")

    alerts = report.get("alerts") or []
    if alerts:
        # ★★ 2026-09-17：**必须让真告警先出现**。
        #    ℹ️ 是记录性的（上游撤数/厂商重算/已知厂商洞），⚠️🔴✘ 才要人处理；
        #    旧实现直接取前 20 条，而 report._judge 把 🔴"上游改历史"排在**最后** ——
        #    实测一次 62 条假 ℹ️ 可以把当轮全部 🔴 顶出屏幕（真信号被假信号灭掉）。
        #    这里按"是否要处理"稳定排序（同类内保持原顺序），并在被截断时说清。
        actionable = [a for a in alerts if not a.startswith("ℹ️")]
        info = [a for a in alerts if a.startswith("ℹ️")]
        ordered = actionable + info
        lines.append(f"  ⚠️ 需要注意（{len(alerts)} 条，其中**要处理 {len(actionable)} 条**）：")
        for a in ordered[:20]:
            lines.append(f"    · {a}")
        if len(ordered) > 20:
            lines.append(f"    … 另有 {len(ordered)-20} 条（其中要处理 "
                         f"{max(0, len(actionable)-20)} 条）；完整清单见 state/last_report.json")
        lines.append("")

    lines.append(f"  {report.get('verdict','')}")
    lines.append("=" * 92)
    return "\n".join(lines)


# ================================================================ 趋势图
def render_trend(days: int = 30) -> str:
    """从 state/run_log.jsonl 画纯 ASCII 趋势（请求数 / 重试率 / 耗时）。"""
    if not paths.RUN_LOG.exists():
        return "（还没有运行记录：state/run_log.jsonl 不存在）"
    recs = []
    for ln in paths.RUN_LOG.read_text(encoding="utf-8").splitlines()[-days:]:
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    if not recs:
        return "（运行记录为空）"
    lines = [f"  近 {len(recs)} 次运行趋势（state/run_log.jsonl）", ""]
    mx_req = max((r.get("requests") or 0) for r in recs) or 1
    lines.append(f"  {'日期':12} {'请求':>7} {'重试率':>7} {'耗时':>8}  请求量")
    lines.append("  " + "-" * 72)
    for r in recs:
        req = r.get("requests") or 0
        # ★ 2026-09-17：口径与日报对齐 —— 剔除「空响应自旋」（expect_rows=True 的
        #   接口返回空时 client 自己重试，那是我们主动探测，不是服务端抖动）。
        #   旧实现用原始 retries，会出现"趋势图打 ⚠、当天日报说无异常"的自相矛盾。
        ret = max(0, (r.get("retries") or 0) - (r.get("empty_retries") or 0))
        rate = ret / req if req else 0
        flag = " ⚠" if rate > 0.02 else ""
        lines.append(f"  {str(r.get('finished_at',''))[:10]:12} {req:>7} {rate*100:>6.1f}% "
                     f"{fmt_duration(r.get('seconds',0)):>8}  {bar(req/mx_req, 24)}{flag}")
    return "\n".join(lines)


# ================================================================ HTML
def write_html(report: dict, path=None) -> str:
    """单文件 HTML 报告：内联 CSS + 内联 SVG，无外部依赖。"""
    p = path or (paths.STATE / "report.html")
    r = report
    rows_html = []
    for gname, items in (r.get("groups") or {}).items():
        for it in items:
            rows_html.append(
                f"<tr><td>{gname}</td><td>{it['name']}</td><td class=n>{it.get('rows',0):,}</td>"
                f"<td class=n>{it.get('requests',0)}</td><td>{it.get('status','✔')}</td></tr>")
    rev_html = "".join(
        f"<tr class=bad><td>{x['dataset']}</td><td>{x['date']}</td>"
        f"<td>{x.get('old_rows','?')} → {x.get('new_rows','?')}</td>"
        f"<td>{', '.join(x.get('changed_cols') or []) or '(整行)'}</td></tr>"
        for x in (r.get("revisions") or []))
    html = f"""<!doctype html><html lang=zh><meta charset=utf-8>
<title>每日增量报告 {r.get('finished_at','')}</title>
<style>
body{{font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
     margin:32px;color:#1a1a1a;background:#fafafa}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#666;margin-bottom:20px}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:12px 0 24px;
       box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:6px;overflow:hidden}}
th,td{{padding:7px 12px;text-align:left;border-bottom:1px solid #eee;font-size:13px}}
th{{background:#f4f4f6;font-weight:600}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
tr.bad td{{background:#fff1f0}} tr:last-child td{{border-bottom:none}}
.ok{{color:#137333}} .warn{{color:#b06000}} .bad{{color:#c5221f}}
</style>
<h1>每日增量报告</h1>
<div class=sub>{r.get('finished_at','')} · T = {r.get('T','?')} ·
  用时 {fmt_duration(r.get('seconds',0))} · 请求 {r.get('requests',0):,}（重试 {r.get('retries',0)}）</div>
<div>{r.get('verdict','')}</div>
<h2 style="font-size:15px;margin-top:24px">数据集明细</h2>
<table><tr><th>分组</th><th>数据集</th><th>新增行数</th><th>请求</th><th>状态</th></tr>
{''.join(rows_html) or '<tr><td colspan=5>无</td></tr>'}</table>
{'<h2 style="font-size:15px">🔴 上游修改了历史数据</h2><table><tr><th>数据集</th><th>日期</th><th>行数</th><th>变化的列</th></tr>'+rev_html+'</table>' if rev_html else ''}
</html>"""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p)
