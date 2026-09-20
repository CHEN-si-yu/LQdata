"""数据可得性延迟表 —— **单一真相源**（不在这里硬编码）。

## 为什么要有这个文件

「上游表 T 日的数据要到 T+delay 才拿得到」这件事，会让**同一段历史在不同日子算出不同值**：
不处理的话，T 日的因子值当天看着完全正确，第二天数据到了就悄悄变成另一个值 ——
没有任何报错，`check` 也查不出来。所以 `fea/spec.py` 的 `register()` 会在注册期
硬拦「依赖了延迟表却没声明 `lagged_ok`」的因子。

拦的名单不能靠人记、也不能靠某个模块里的一个常量 —— 那迟早会和实际观测脱节。
模块① 的日更工程（`everyday_tasks`）**每天都在实测**每张表的服务端最新日期，
结果落在 `state/delay_history.json` 里；`datadownload/conf/frequency.yaml`
是静态声明表。

## 取值规则（★★ 2026-09-15 深夜定稿：**以声明常数为准**，观测只做安全网）

对每张表：

1. **声明常数**（主）：`datadownload/conf/frequency.yaml` 的 `delay_days`
   —— 它由人维护、与日更侧 `registry.py::_DELAY` **逐表对齐**，代表"这张表与厂商的约定"。
2. **观测安全网**（只上不下）：只有当 `delay_history.json` 里**最近 ≥3 次观测连续一致地
   比声明更差（更大）**时，才临时按更大的用，并记一条 warning 提示人工复核。
   —— 与日更侧 `delay.absent_streak`（连续 3 轮探测不到就按 delay+1 处理）同一套哲学。
3. `_BUILTIN` 兜底（两个来源都读不到时，至少不静默放过）。

**最后统一过滤掉 `<=0` 的表** —— 0 表示"当天可得"，本就不该进滞后名单。

### ⚠️ 为什么不是"按最新一次观测取值"（我 2026-09-15 晚写错过一版，同日被证伪）

那天晚上我一度改成"最近一次观测赢"，当天深夜就被现实打脸：
`ths_hot` / `stock_st_info` / `index_ths_daily` 在同一晚的 7 次采样里滞后序列是
`0,1,1,1,0,0,0`（新→旧）—— **21:22 还观测到滞后 1，23:32 服务端就出现了当天数据**。
也就是说：
- 它们的常数 delay **就是 0**（当天深夜发布），不是 1；
- 但"发布晚于我们跑批的时间"会让**早采样**系统性地高估滞后；
- 拿单次采样当常数 → 一晚之内就会翻来覆去，因子值跟着抖动。

**判据的正确形态**：delay 是**厂商的契约常数**，观测是用来**发现契约变化**的证据，
不是常数本身。要么多采几次（同晚多次 / 跨日），要么看"发布时点是否晚于跑批时点"，
再人工把结论写进 `frequency.yaml` + `registry._DELAY`（两边必须同时改）。

### 为什么从"取最大值"改成"取最新观测"

旧实现三路全部 `max()`、且入口就把 `delay<=0` 丢掉，于是**方向不对称：加严能传播，
放宽永远传播不了**。2026-09-15 上游实测确认 `dc_daily` / `stock_top_list` /
`stock_dragon_tiger` / `stock_adj_factor_changes` **当天就有数据**（`delay_obs: 0`，
见 registry.py 的 `_DELAY` 注释），但旧实现里它们被历史最大值 + `_BUILTIN` 钉死在 1，
因子侧继续白丢一天信息。用户口径：「**服务端当天有数据就必须拿当天**」。

⚠️ 这条规则的前提（别改坏）：**因子在 D 日用的是 D 日收盘后到当晚发布的数据，
下游按 D+1 及以后交易** —— 所以"D 当晚发布"对因子不构成未来函数。
若哪天改成"用 D 日数据在 D 日盘中做决策"，这里必须退回保守的 `max` 口径。

## 注意

- 模块② 的因子**只读**这两个文件，绝不写。
- 读不到不是错误：上游目录被挪走/换机器时回退到兜底表并记一条 warning。
  宁可少拦一个，也不要因为「另一个模块的文件格式变了」而让整个因子工程 import 失败。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("fea.delay")

# 兜底表：只在**两个上游来源都读不到**时生效（正常运行时它不参与取值）。
# ⚠️ 这里可能过期 —— 别拿它当真相，真相在 delay_history.json 的最近观测里。
_BUILTIN: dict[str, int] = {"stock_margin_detail": 1}


def _sources(upstream_root: Path | None) -> tuple[Path | None, Path | None]:
    """返回 (delay_history.json, frequency.yaml) 的路径（可能不存在）。"""
    if upstream_root is None:
        # 默认：本文件在 <平台根>/featureengineering/fea/ 下 → parents[2] = 平台根
        parent = Path(__file__).resolve().parents[2]
    else:
        parent = Path(upstream_root).resolve().parent      # ../  = 平台根目录
    hist = parent / "everyday_tasks" / "state" / "delay_history.json"
    freq = parent / "datadownload" / "conf" / "frequency.yaml"
    return (hist if hist.exists() else None, freq if freq.exists() else None)


def _recent_by_table(p: Path, n: int = 3) -> dict[str, list[int]]:
    """每张表**最近 n 次有值的观测**（日期倒序，只收 `delay_obs` 非 None 的）。

    ⚠️ 仅供"安全网"用（见 load 的注释）：单次观测**不是**常数本身 ——
       早采样的观测会系统性高估"当晚才发布"的表的滞后。
    """
    out: dict[str, list[int]] = {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:                                  # noqa: BLE001
        log.warning("读 delay_history.json 失败（忽略）：%s", exc)
        return out
    if not isinstance(raw, dict):
        return out
    for day_key in sorted(raw, reverse=True):                 # 日期倒序 = 从最新往回
        tables = (raw.get(day_key) or {}).get("tables") or {}
        for name, rec in tables.items():
            d = (rec or {}).get("delay_obs")
            if not isinstance(d, (int, float)):
                continue                                      # 该次没观测出来，跳过
            lst = out.setdefault(name, [])
            if len(lst) < n:
                lst.append(int(d))
        if all(len(v) >= n for v in out.values()) and len(out) >= len(tables):
            break
    return out


def _from_frequency(p: Path) -> dict[str, int]:
    """静态声明表（yaml）。保留 0（0 是"当天可得"，对优先级合并有意义）。"""
    out: dict[str, int] = {}
    try:
        import yaml                                          # 平台环境已装 pyyaml
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:                                  # noqa: BLE001
        log.warning("读 frequency.yaml 失败（忽略）：%s", exc)
        return out
    for name, rec in (raw.get("datasets") or {}).items():
        d = (rec or {}).get("delay_days")
        if isinstance(d, (int, float)):
            out[name] = int(d)
    return out


_CACHE: dict[str, int] | None = None


def load(upstream_root: Path | str | None = None, refresh: bool = False) -> dict[str, int]:
    """合并后的延迟表 `{数据集: 延迟交易日数}`（**只含 delay>0 的**）。"""
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    hist_p, freq_p = _sources(Path(upstream_root) if upstream_root else None)
    freq = _from_frequency(freq_p) if freq_p is not None else {}
    recent = _recent_by_table(hist_p) if hist_p is not None else {}

    # ① 声明常数为主：yaml 覆盖 _BUILTIN（_BUILTIN 只是两个来源都读不到时的兜底）
    declared: dict[str, int] = {**_BUILTIN, **freq}
    src = ["builtin", "frequency.yaml"] + (["delay_history.json"] if hist_p is not None else [])
    out: dict[str, int] = {}
    picked: dict[str, str] = {}
    for name, d in declared.items():
        obs = recent.get(name) or []
        # ② 安全网：连续 ≥3 次观测一致地比声明更差 → 临时按更大的用（只上不下）
        if len(obs) >= 3 and all(o > d for o in obs[:3]):
            log.warning("⚠️ %s：连续 3 次观测（%s）都比声明 delay=%d 更差 → 临时按 %d 用；"
                        "请人工复核 frequency.yaml / registry._DELAY", name, obs[:3], d, max(obs[:3]))
            d, picked[name] = max(obs[:3]), "观测上调"
        else:
            picked[name] = "yaml" if name in freq else "builtin"
        out[name] = d
    out = {k: v for k, v in out.items() if v > 0}             # ★ 最后统一过滤 0
    _CACHE = out
    log.info("延迟表已加载（来源：%s，生效 %d 张）：%s", "+".join(src), len(out),
             "、".join(f"{k}={v}({picked[k]})" for k, v in sorted(out.items())))
    return out
