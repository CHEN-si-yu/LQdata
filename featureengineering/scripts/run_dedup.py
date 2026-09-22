#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量因子冗余审计 runner（评审建议 S-02 的因子侧半，2026-09-22）。

## 为什么单独写一个 runner，而不是直接 `main.py dedup`

`main.py` 的每个命令入口在启动时都会调 `fea/resources.py::configure()`，把进程的
`RLIMIT_AS` 压到 **24 GiB** 作为共享内存保护。2026-09-21 的全量审计两次都卡在这里：
相关矩阵与簇已经算出、屏幕上打印过，**写报告时申请 34 MB 都失败** —— 那是
**地址空间**耗尽，不是真内存不够（审计本身按年分块累加 `G = XᵀX`，不可回收内存只有 GB 量级）。

本 runner 与 `main.py dedup` 走**同一个实现**（`fea.dedup.cmd_dedup`），唯一区别是
**不调用 `configure()`**，并把 `RLIMIT_AS` 放开到**现读的 cgroup 上限**：

- 不写死 24 GiB / 90 GiB 这类数字 —— 容器上限换台机器就会变，脚本一律现读
  （`/sys/fs/cgroup/memory.max`，v1 回退到 `memory/memory.limit_in_bytes`）；
- 只**放宽**、不收紧：拿到比现状更小的上限时保持原样，绝不让 runner 自己成为限制来源；
- 真实的兜底仍是 cgroup 硬上限 —— 超了会 OOM-Kill，不会拖垮同容器的别的进程。

★ 资源纪律：审计是一次**单进程**重活（全库因子读一遍 + 656² 的 Gram 累加），
跑之前确认没有训练/分析在并行（平台的「两个重活绝不并行」）。

## 用法

    PY=/autodl-fs/data/miniconda3/bin/python
    $PY scripts/run_dedup.py                       # 全库、全历史、|ρ| ≥ 0.95
    $PY scripts/run_dedup.py --threshold 0.90      # 换阈值（报告覆盖写 state/dedup/report.json）
    $PY scripts/run_dedup.py --years 2025 2026     # 只审计这两年

产物：`state/dedup/report.json`（簇清单 + 完整相关矩阵）。
下一步（交付物）是 `scripts/dedup_representatives.py`，把机读报告转成保守代表清单。
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ★ BLAS 线程数必须**在 import numpy 之前**设定（numpy/BLAS 在导入时读这些变量）。
#   因子侧的并行模型是多进程，单进程内的多线程只会互相踩踏；这里沿用 main.py 的口径
#   （默认 1），需要时用环境变量覆盖而不要改代码 —— 不同机器的核数不一样。
_THREADS = os.environ.get("FEA_BLAS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _THREADS


def _cgroup_limit() -> int | None:
    """现读 cgroup 内存上限（字节）；读不到返回 None。"""
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            txt = Path(p).read_text().strip()
        except OSError:
            continue
        if txt != "max":                       # cgroup v2 的「无限制」写法
            try:
                return int(txt)
            except ValueError:
                continue
    return None


def _relax_address_space() -> None:
    """把 `RLIMIT_AS` 放宽到 cgroup 上限，并打印前后值（审计留痕）。

    只**放宽**、不收紧 —— 已经比 cgroup 上限宽松（例如 unlimited）时原样保留：
    真正的兜底是 cgroup 硬上限，runner 不该反过来成为新的限制来源。
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = _cgroup_limit()
    fmt = lambda v: "unlimited" if v == resource.RLIM_INFINITY else f"{v / 1024**3:.1f} GiB"
    if cap is None:
        print(f"地址空间上限 RLIMIT_AS = {fmt(soft)}（读不到 cgroup 上限，保持原值）")
        return
    want = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
    if soft == resource.RLIM_INFINITY or (want != resource.RLIM_INFINITY and soft >= want):
        print(f"地址空间上限 RLIMIT_AS = {fmt(soft)}（已不低于 cgroup 上限 {fmt(cap)}，保持原值）")
        return
    resource.setrlimit(resource.RLIMIT_AS, (want, hard))
    print(f"地址空间上限 RLIMIT_AS: {fmt(soft)} → {fmt(want)}"
          f"（cgroup 上限 {fmt(cap)}，BLAS 线程 {_THREADS}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="全量因子冗余审计（绕过 24 GiB RLIMIT_AS）")
    ap.add_argument("factors", nargs="*", help="只检查指定因子（默认全库）")
    ap.add_argument("--years", nargs=2, type=int, default=None, metavar=("Y0", "Y1"),
                    help="只检查这些年（默认全部）")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="判定重复的 |ρ| 阈值（默认 0.95）")
    ap.add_argument("--matrix", action="store_true",
                    help="老的一次性长矩阵路径（只用于与分块路径对拍，全库跑会吃几十 GB）")
    ap.add_argument("--out", default="",
                    help="报告写到这个路径（默认 state/dedup/report.json）；试跑时别覆盖生产报告")
    args = ap.parse_args()

    # ★ `configure()` 在 main.py 里承担两件事：压 RLIMIT_AS + 固定 BLAS 线程。
    #   线程已在文件头用环境变量固定；RLIMIT_AS 在这里换成「放宽」，所以不再调它。
    import factors                              # noqa: F401  注册即完成因子登记
    from fea import config as cfg_mod
    from fea.dedup import cmd_dedup

    cfg_mod.setup_logging(False)
    cfg = cfg_mod.load()
    _relax_address_space()
    return cmd_dedup(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
