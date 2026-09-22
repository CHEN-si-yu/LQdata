"""股票池（universe）—— 只保留主板。

★ 用户的硬性约束：下游是 **A 股日横断面回归排序**，只考虑 **主板 ~3000+ 只**。
主板 = 沪市 600/601/603/605 + 深市 000/001/002/003（002 原中小板，2021-04 并入主板）。
排除创业板 300/301/302、科创板 688/689、北交所 832/833/920。

把过滤放在**网格构造**这一步，而不是每个因子里各写一遍：
这样每个因子的输入、计算、截面 rank 天然只在主板内，不会有人漏掉。
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from .dates import series_to_int
from .panel import Panel

log = logging.getLogger("fea.universe")

_NO_END = 99_999_999


def is_main_board(codes: np.ndarray, prefixes: tuple[str, ...]) -> np.ndarray:
    c = np.asarray(codes, dtype=str)
    out = np.zeros(c.shape, dtype=bool)
    for p in prefixes:
        out |= np.char.startswith(c, p)
    return out


def frozen_codes(cfg) -> np.ndarray | None:
    """冻结股票池的代码数组；`None` = 未启用（conf 的 `universe.frozen_list` 为空）。

    文件格式见 `conf/universe_frozen.tsv`：`#` 开头是注释，其后每行
    `stock_code<TAB>list_date<TAB>delist_date`。只取第一列。
    """
    fx = cfg.frozen_universe
    if fx is None:
        return None
    codes = []
    for ln in fx.read_text(encoding="utf-8").splitlines():
        if not ln or ln.startswith("#"):
            continue
        c = ln.split("\t")[0].strip()
        if c:
            codes.append(c)
    if not codes:
        raise RuntimeError(f"冻结股票池名单为空：{fx}")
    return np.sort(np.array(sorted(set(codes)), dtype=str))


def frozen_fingerprint(cfg) -> str:
    """冻结名单的指纹 —— 进 `Engine.universe_fp`，改了它就会触发全量重建。

    ★ 用**内容哈希**而不是文件路径/修改时间：文件被同内容重写不该触发重建，
      而只要代码清单变了一只，就必须重算全部因子的历史值（rank 依赖池子）。
    """
    codes = frozen_codes(cfg)
    if codes is None:
        return "off"
    h = hashlib.sha1("\n".join(codes.tolist()).encode()).hexdigest()
    return f"{h[:12]}:n={len(codes)}"


def code_master(up, cfg) -> np.ndarray:
    """股票代码全集（已排序）。

    · 未配置冻结名单 → **主板全集（含已退市）**，各票按 `[list_date, delist_date]` 进出。
      用 stock_list 而不是「当前上市的股票」：它带 `delist_date`，
      289 只已退市股票也在里面，所以那样**没有幸存者偏差**。
      （灵启因子库文档禁用了依赖 stock_list 的因子，理由是「当前快照非 PIT」；
      我们只拿它做**板块归属**和**上市窗口**——这两者不随快照时点变化，
      而且用代码前缀判定板块根本不依赖快照。）

    · 配置了冻结名单 → 在上面基础上**取交集**（★★ 2026-09-18 用户拍板的口径）。
      名单本身已经保证是主板 + 存续 + 从未ST + 上市≤2018-01-01，
      但这里仍先过一遍主板前缀：万一名单被手工改脏，也不会把创业板塞进面板。
    """
    sl = up.read("stock_list", columns=["stock_code", "list_date", "delist_date"])
    if sl.empty:
        raise RuntimeError("上游 stock_list 不存在，无法确定股票池")
    codes = np.sort(sl["stock_code"].astype(str).unique())
    codes = codes[is_main_board(codes, cfg.board_prefixes)]

    want = frozen_codes(cfg)
    if want is None:
        return codes
    missing = sorted(set(want.tolist()) - set(codes.tolist()))
    if missing:
        raise RuntimeError(
            f"冻结股票池里有 {len(missing)} 只不在主板全集内（名单与上游口径不一致）："
            f"{missing[:10]}\n  冻结名单由 scripts/build_frozen_universe.py 生成，"
            f"请重新生成或在 conf/config.yaml 里置空 frozen_list。")
    return codes[np.isin(codes, want)]


def listed_mask(panel: Panel, up, cfg, st_ev=None) -> np.ndarray:
    """(T, C) bool —— 每只股票在其 [list_date, delist_date] 窗口内。

    未上市/已退市的格子直接是 False，因此不会产生「未来才上市」的因子值。
    """
    sl = up.read("stock_list", columns=["stock_code", "list_date", "delist_date"])
    sl = sl.drop_duplicates("stock_code")

    pos = {c: i for i, c in enumerate(panel.codes)}
    cols = np.array([pos.get(c, -1) for c in sl["stock_code"].astype(str)], dtype=np.int64)
    keep = cols >= 0
    cols = cols[keep]

    ld_raw = sl["list_date"].astype("string").fillna("").to_numpy()[keep]
    dd_raw = sl["delist_date"].astype("string").fillna("").to_numpy()[keep]
    lo = series_to_int(pd.Series(ld_raw)).astype(np.int64)
    hi = series_to_int(pd.Series(dd_raw)).astype(np.int64)
    lo[lo == 0] = 0                       # 空 list_date -> 从最早算起
    hi[hi == 0] = _NO_END                 # 空 delist_date -> 至今

    a = np.clip(np.searchsorted(panel.dates, lo, side="left"), 0, panel.T)
    b = np.clip(np.searchsorted(panel.dates, hi, side="right"), 0, panel.T)

    # ★ 差分数组 + 前缀和，代替「逐列切片赋值」的 Python 循环。
    #   原来是 for c in range(3484): mask[a:b, c] = True —— 单次 1.86s，
    #   而它是 (日期范围 × 股票池) 的纯函数，在 15 年 × 10 因子里被重算 150 次。
    T, C = panel.shape
    flat_add = a.astype(np.int64) * C + cols
    flat_sub = b.astype(np.int64) * C + cols
    n = (T + 1) * C
    diff = (np.bincount(flat_add, minlength=n) - np.bincount(flat_sub, minlength=n))
    mask = diff.reshape(T + 1, C).cumsum(axis=0)[:T] > 0

    if cfg.min_listed_days:
        n_skip = int(cfg.min_listed_days)
        starts = mask.argmax(axis=0)                     # 每列首个 True 的行号
        has = mask.any(axis=0)
        for c in np.flatnonzero(has):
            mask[starts[c]:starts[c] + n_skip, c] = False

    if cfg.exclude_st:
        mask &= ~st_mask(panel, st_ev if st_ev is not None else st_events(up, panel.codes))

    return mask


def st_events(up, codes: np.ndarray, strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """把 ST 记录预处理成 `(列号, 日期)` 两个数组 —— **与面板日期范围无关**。

    单独抽出来是为了只算一次：原来每次 `st_mask()` 都要对 32.7 万行做
    `isin(dict)` + `map(dict)`，实测单次 917ms，而它在一次全量重建里会被调用上百次。
    用 `reindex` 代替 `map(dict)` 也快得多。

    ★★ 2026-09-21：读失败**不再无条件静默返回空**（S-01）。两条分支的理由不同，
    别把它们合并成一条：

    · `strict=True` —— 由 `Engine.universe_for` 在**股票池动态**时传入 → 直接抛错。
      池子动态进出时，「读不到 ST」与「真的没有 ST」会算出**不同的因子值**，
      而静默降级正是平台纪律明令禁止的 bug 形态（崩了你会去修，降级了你以为它在工作）。

    · `strict=False` —— 当前**冻结池**走这条 → 仍然返回空，但**必须留痕**。
      冻结名单的口径是「主板 ∩ 至今存续 ∩ 从未ST(2016-08-09 起)」，池内本来就不该有
      ST 事件，所以读不到**不会改变任何一个因子值**；此时抛错只会变成每天必响的假警报，
      而假警报会把真信号淹掉（同 HISTORY 里「增量丢行」那条告警的教训）。
      留痕落在 `st_pool_fingerprint()["source"]`，可事后审计；内容一变则
      `Engine.st_gate()` 会**直接拦下**整轮运行。
    """
    empty = (np.zeros(0, np.int32), np.zeros(0, np.int32))
    try:
        st = up.read("stock_st_info", columns=["stock_code", "trade_date"])
    except Exception as exc:                                  # noqa: BLE001
        if strict:
            raise RuntimeError(
                "✘ 读 stock_st_info 失败，而 universe.exclude_st=true 且股票池是**动态**的 —— "
                "此时「读不到 ST」与「没有 ST」会算出不同的因子值，拒绝静默按「无 ST」继续。\n"
                f"  原始错误：{exc!r}\n"
                "  修法：先修上游 stock_st_info；或显式把 universe.exclude_st 置 false 后重建。") from exc
        log.warning("⚠️ 读 stock_st_info 失败，按「无 ST」继续（冻结池口径下池内无 ST 事件，"
                    "不改变结果；已记入 state/universe_st.json 的 source 字段）：%r", exc)
        return empty
    if st is None or st.empty:
        if strict:
            raise RuntimeError(
                "✘ stock_st_info 为空，而 universe.exclude_st=true 且股票池是**动态**的 —— "
                "拒绝静默按「无 ST」继续（空表与读取失败在结果上无法区分）。")
        log.warning("⚠️ stock_st_info 为空，按「无 ST」继续（冻结池口径下不影响结果，已留痕）")
        return empty

    lut = pd.Series(np.arange(len(codes), dtype=np.int32), index=np.asarray(codes))
    cc = lut.reindex(st["stock_code"].astype(str)).to_numpy()
    ok = ~pd.isna(cc)
    if not ok.any():
        return empty
    dd = series_to_int(st["trade_date"])
    return cc[ok].astype(np.int32), dd[ok]


def st_mask(panel: Panel, events: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """(T, C) bool —— ST/*ST 期间。仅当 config.universe.exclude_st=true 时启用。"""
    cc, dd = events
    if cc.size == 0:
        return np.zeros(panel.shape, dtype=bool)
    return panel.scatter(cc, dd, weights=np.ones(cc.size)) > 0


def st_pool_fingerprint(up, codes: np.ndarray) -> dict:
    """ST 事件在**冻结池之内**的内容指纹 —— 落 `state/universe_st.json`，变了就拦。

    两个设计要点（都是权衡过的，别顺手改）：

    1. **只统计池内**。厂商每天都在给池外股票记 ST（全表 32.8 万行）；
       把池外也算进来的话这个指纹会**天天变**，闸门立刻退化成噪声源 ——
       而没人再看的闸门等于没有闸门（同 `delay.py` 里「假警报会淹没真信号」那一课）。

    2. **不并进 `Engine.universe_fp`**。那个指纹一变就是**全部 653 个因子全量重建**
       （几十 GB 重写，还会跟正在跑的模型训练抢共享盘）。ST 内容变化确实会让历史值变，
       但「现在要不要重建」必须由人拍板（可能正在训练、可能只是厂商补了一条无关记录），
       所以这里选择**拦下来并报告**，而不是替用户按下重建按钮。

    `source` 三态是为了事后能审计「当时到底读到了没有」：
      ok / empty（表在但没有池内事件）/ unavailable（读失败）。
    """
    no = {"error": None, "sha1": None, "n_events": None, "max_date": None}
    try:
        st = up.read("stock_st_info", columns=["stock_code", "trade_date"])
    except Exception as exc:                                  # noqa: BLE001
        return {"source": "unavailable", **{**no, "error": repr(exc)}}
    if st is None or st.empty:
        return {"source": "empty", **no, "n_events": 0}
    sc = st["stock_code"].astype(str)
    sub = st.loc[sc.isin(set(np.asarray(codes, dtype=str).tolist())).to_numpy()]
    if sub.empty:
        return {"source": "ok", **no, "sha1": "pool-empty", "n_events": 0}
    key = (sub["stock_code"].astype(str) + "|" + sub["trade_date"].astype(str)).sort_values()
    return {
        "source": "ok", "error": None,
        "sha1": hashlib.sha1("\n".join(key.tolist()).encode()).hexdigest(),
        "n_events": int(len(sub)),
        "max_date": str(sub["trade_date"].astype(str).max()),
    }
