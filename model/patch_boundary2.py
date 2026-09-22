"""台地边界检查（续）：把 `exit_rank` 的形状补完到上界（2026-09-22，RD11 第四轮）。

**为什么还要加**：第三轮加 800/1000 做边界检查，结果是 `g_re1000` 在三粒种子上
**最差 +41.05%、极差仅 9.9pp**，远稳于 700（最差 +16.76%）。这可能是"安全区越宽越稳"
的单调机制，也可能只是**在评价窗上又挑了一次**（`R35` 警告的那件事）。

两件事必须分开：
  (a) **形状**：`exit_rank` 一直加下去是继续变好，还是掉头？——加 1500 / 2000 补完。
  (b) **选择**：即便形状好，**按评价窗挑最好那格就是选择偏差** ⇒ 报告必须写明
      "1000 是评价窗上挑出来的"。机制解释（安全区宽度 ⇒ 名次抖动不敏感）要先立住。

★ 宇宙只有 2115 只，`exit_rank=2000` 已接近"只要不掉到尾部 5% 就继续持有"。
  再加 `hold_all`（不设 exit_rank、纯周期性换仓）作为**形状的右端点**做对照。
"""
import sys
from pathlib import Path

ROOT = Path('/autodl-fs/data/model')
UNITS = ['V76', 'V77', 'V78', 'V79', 'V80', 'V81']

OLD = """STRATEGIES += [dict(name=f'buffer_r{r}_t{n}', topn=n, exit_rank=r, min_hold=1)"""
NEW = """# --- ★ 2026-09-22 RD11 第四轮：**把 exit_rank 的形状补到上界**（形状检查，不是挑参数）---
# 第三轮发现 g_re1000 三粒种子最差 +41.05%、极差 9.9pp，远稳于 700。要分清是
# 「安全区越宽越稳」的单调机制，还是「在评价窗上又挑了一次」（R35）。
# ⇒ 补 1500 / 2000 两点看是否掉头。
# ★ **不加** `period=5` 那类「固定周期 + 门槛」：固定周期只在调仓日重算 `desired`，
#   risk_off 落在非调仓日时它会**继续持有**，与门槛语义（risk_off 即清仓）不符 ——
#   会产出一个看着合理但口径错的数字。排名退出族在 risk_off 日 `ranked` 为空、
#   持仓一律出局，语义正确，所以形状检查只用排名退出族。
STRATEGIES += [
    dict(name='g_re1500_t5_h5', topn=5, exit_rank=1500, min_hold=5, gate=True),
    dict(name='g_re2000_t5_h5', topn=5, exit_rank=2000, min_hold=5, gate=True),
    dict(name='re1500_t5_h5', topn=5, exit_rank=1500, min_hold=5),
]

STRATEGIES += [dict(name=f'buffer_r{r}_t{n}', topn=n, exit_rank=r, min_hold=1)"""

for unit in UNITS:
    p = ROOT / unit / 'analysis.py'
    if not p.exists():
        print(f'· {unit} 不存在，跳过')
        continue
    src = p.read_text(encoding='utf-8')
    n = src.count(OLD)
    if n != 1:
        print(f'✘ {unit}: 命中 {n} 次，期望 1 —— 中止', file=sys.stderr)
        raise SystemExit(1)
    p.write_text(src.replace(OLD, NEW, 1), encoding='utf-8')
    print(f'✔ {unit}/analysis.py 加好形状补完格')
