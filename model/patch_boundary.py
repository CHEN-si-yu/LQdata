"""台地边界检查补丁（2026-09-22，RD11 第三轮）。

**为什么加**：`R26` 登记的台地是 `exit_rank 400~700`，注册表只放 400/500/600/700。
本轮实测**无门槛**时 `re600` 最好、**有门槛**时 `re700` 最好 —— 而 700 是**已登记台地的上界**。
上界处变好有两种解释，读数完全不同：

  (a) **台地**：700 附近是一片平的，换个点也差不多 ⇒ 结论稳。
  (b) **山脊**：还在往上爬 ⇒ "台地"这个说法在无门槛下就不成立，得改口径。

**这是边界稳健性检查，不是参数加密** —— 只往外加 800/1000 两点看是否掉头，
**不据此挑最好那格**（选点仍以预登记的 400~700 为准）。

★ 同时补一条：`g_re700_t5_h5_sl8_tp20` —— 有门槛时的主实验格（因为带门槛时最优点移到了 700）。
"""
import sys
from pathlib import Path

ROOT = Path('/autodl-fs/data/model')
UNITS = ['V76', 'V77', 'V78', 'V79']

OLD = """STRATEGIES += [dict(name=f'buffer_r{r}_t{n}', topn=n, exit_rank=r, min_hold=1)"""
NEW = """# --- ★ 2026-09-22 RD11 第三轮：**台地边界检查**（不是参数加密）---
# 已登记台地是 400~700（R26）。本轮无门槛时 re600 最好、有门槛时 re700 最好，
# 而 700 是台地上界 ⇒ 必须分清"台地"还是"山脊"。只往外加 800/1000 两点看是否掉头。
# ★ 选点仍以预登记的 400~700 为准；这两点**只用于判断形状**，不用来挑最好那格。
STRATEGIES += [
    dict(name='re800_t5_h5', topn=5, exit_rank=800, min_hold=5),
    dict(name='re1000_t5_h5', topn=5, exit_rank=1000, min_hold=5),
    dict(name='g_re800_t5_h5', topn=5, exit_rank=800, min_hold=5, gate=True),
    dict(name='g_re1000_t5_h5', topn=5, exit_rank=1000, min_hold=5, gate=True),
    dict(name='g_re700_t5_h5_sl8_tp20', topn=5, exit_rank=700, min_hold=5, stop_loss=.08, take_profit=.20, gate=True),
]

STRATEGIES += [dict(name=f'buffer_r{r}_t{n}', topn=n, exit_rank=r, min_hold=1)"""

for unit in UNITS:
    p = ROOT / unit / 'analysis.py'
    src = p.read_text(encoding='utf-8')
    n = src.count(OLD)
    if n != 1:
        print(f'✘ {unit}: 命中 {n} 次，期望 1 —— 中止', file=sys.stderr)
        raise SystemExit(1)
    p.write_text(src.replace(OLD, NEW, 1), encoding='utf-8')
    print(f'✔ {unit}/analysis.py 加好边界检查格')
