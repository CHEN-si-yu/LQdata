"""因子面板加载 —— 从 `trainingdata/`（初加工产物）读成 (T·C, F) 的特征矩阵 + y。

## 分工（2026-09-18 重构后）

    上游单因子 parquet  ──preparingdata.py──▶  trainingdata/  ──本文件──▶  PanelData

初加工阶段已经做完的事（**本层不再重复做**，也不再碰上游因子文件）：
  · 把 230 个因子对齐到同一张**权威网格**（日历 ∩ 产物端点 × Engine.codes），缺格 NaN；
  · 特征只留 `rank` 列，并把方向统一成"越大越好"（需翻的存 `1 − rank`）；
  · 标签留 `value` 原值；股票池掩码与模块② 逐格一致，一并落盘。

所以本层只剩三件事：
  ① 按日期区间 / 年份选文件读进来（列裁剪 + 行过滤，均为 pyarrow 下推）；
  ② 对齐成一张 `(T·C, F)` float32 矩阵 + `(T·C,)` 的标签 + `(T·C,)` 的池掩码；
  ③ **校验**：X / Y / 掩码三者的行数、首末键、日期集合、代码集合必须一致
     —— 这三份是同一个构建器写出来的，一旦不一致就是产物坏了，必须立刻报错而不是静默错位。

★ `start` / `end` 只做**日期切片**（audit-pit 的"截断复算"就靠它：`load(cfg, end=cut)`），
  切片按整日进行，所以「每天都是完整截面」这个性质不会被破坏。
★ 内存：全历史 X 约 12.4M 行 × 225 列 × 4B ≈ **11.4 GB**；单年约 0.8 GB。
  按年加载 + 逐折训练是既定纪律（见 README 的内存纪律）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import panel_io, store
from .config import Cfg

UK = panel_io.UK
XK = panel_io.XK
YK = panel_io.YK


@dataclass
class PanelData:
    dates: list[str]                    # (T,) 交易日（字符串，升序）
    codes: np.ndarray                   # (C,) 股票代码（升序，与初加工产物一致）
    feats: list[str]                    # (F,) 特征名
    X: np.ndarray                       # (T*C, F) float32，方向已统一成"越大越好"
    y: dict[str, np.ndarray]            # label -> (T*C,) float32
    universe: np.ndarray                # (T*C,) bool（模块② 口径的股票池）
    row_day: np.ndarray                 # (T*C,) int32 行号→日索引
    row_code: np.ndarray                # (T*C,) int32 行号→股票索引
    dates_arr: np.ndarray = None         # (T,) object 数组（按行号取日期用）
    meta: dict = field(default_factory=dict)
    _nn: np.ndarray = None               # 每行非空特征数（懒算 + 缓存，见 n_feats）
    _lu: np.ndarray = None               # 每行"T 日收盘涨停"（懒算 + 缓存，见 limit_up_mask）
    _lu_thr: float = None                # 上面那份缓存用的阈值（换阈值会重算）
    _min_cov: float = 0.5                # 特征覆盖率门槛（由 conf 的 data.min_feature_coverage 定）
    clipped: list | None = None          # X 已在载入时夹过的区间（见 preprocess.clip_features）

    def __post_init__(self):
        if self.dates_arr is None:
            self.dates_arr = np.asarray(self.dates, dtype=object)

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    def n_feats(self) -> np.ndarray:
        """每行的非空特征数（**缓存**：全历史 12.4M 行扫一遍 ~1 s，别在标签循环里重复扫）。"""
        if self._nn is None:
            self._nn = np.isfinite(self.X).sum(axis=1).astype(np.int32)
        return self._nn

    def avail_mask(self, ratio: float | None = None) -> np.ndarray:
        """(n_rows,) bool —— 该行的**特征覆盖率达标**（"能和模型打交道"的行）。

        ★ 与 `sample_mask` 用**同一把尺子**。2026-09-18 修：落盘打分处原先写的是
          `n_feats() >= 1`，比训练样本的 `min_feature_coverage` 松得多 —— 同一份数据两把尺子，
          落盘的打分里就会混进训练时根本不认的行。
          实测这条**只影响 2 个格子**（README §8.4「归因修正」）：是个纯粹的正确性收口，
          **不是**性能改进 —— 别拿它当"提升"来报。
        """
        ratio = self._min_cov if ratio is None else float(ratio)
        if ratio <= 0:
            return np.ones(self.n_rows, dtype=bool)
        return self.n_feats() >= max(1, int(round(ratio * len(self.feats))))

    def limit_up_mask(self, thr: float = 0.98,
                      feature: str = "consecutive_limit_up") -> np.ndarray:
        """(n_rows,) bool —— **T 日收盘涨停**（PIT 安全：只用当日截面内的相对秩）。

        `thr` 是 `feature` 的**当日截面 rank** 阈值（初加工已把该因子存成 rank，方向已翻正）。
        实测（2025-09 ~ 2026-09，88.8 万格）：`thr = 0.98` 时
        **精确率 99.6% · 召回 91.4%**（当日真实涨停格占 1.795%）。

        ★ 为什么能用 rank 判涨停：该特征在涨停行上均值 **0.969**、非涨停行 **0.491**，AUC **0.988**；
          而 rank 是**逐日截面内的相对量** ⇒ 天然自校准，换年份不用改阈值。
        ★ 为什么不去读真实价格：价格层在因子侧（本模块只读、且快照已冻结），
          能从 `trainingdata` 自造就不新增耦合。
        ★ 特征缺失（早年 / 上游没这个因子）时判为 **False = 不拦** —— 宁可放过，不可误杀。
        """
        thr = float(thr)
        if self._lu is None or self._lu_thr != thr:
            j = self.feats.index(feature) if feature in self.feats else None
            if j is None:
                self._lu = np.zeros(self.n_rows, dtype=bool)
            else:
                self._lu = np.nan_to_num(self.X[:, j], nan=-1.0) >= thr
            self._lu_thr = thr
        return self._lu

    def buyable_mask(self, limit_up_rank: float | None = 0.98) -> np.ndarray:
        """(n_rows,) bool —— T 日**可交易**：特征覆盖率达标 **且** 当日未收盘涨停。

        ★ 只用在**训练/验证样本**上（见 `mx/train.py`）。**测试集保持全截面**：
          "能不能买"由策略层在执行时刻用真实 `entry_ok` 把关，那才是真实的可执行性；
          在测试集上预先剔掉，等于用"事后才知道买不进"来美化指标。
        ★ 两个判据都只用 T 日（含）之前的信息，T 日收盘后就已知，T+1 开盘下单时才用，
          不构成前视。
        """
        m = self.avail_mask()
        if limit_up_rank is not None and float(limit_up_rank) > 0:
            m = m & ~self.limit_up_mask(float(limit_up_rank))
        return m

    def sample_mask(self, label: str, min_ratio: float | None = None) -> np.ndarray:
        """该标签下的**可用样本**：票池内 ∩ 标签非空 ∩ 特征覆盖率达标。

        ★ 为什么必须取交集：模块② 的**标签没有做 universe 掩码**（实测池外有值、
          池内无值都存在），直接用 `y.notna()` 会把"池外 / 无特征"的格子喂进模型。

        ★ 为什么用"覆盖率比例"而不是"≥1 个特征"：权威网格是**日历 ∩ 上游端点**，
          上游补齐前（或早年底层因子缺失时）会有大量格子只有零星几个特征 ——
          那种行喂进模型只是噪声。默认门槛见 `data.min_feature_coverage`。
        """
        ratio = self._min_cov if min_ratio is None else float(min_ratio)
        ok = self.universe & np.isfinite(self.y[label])
        if ratio > 0:
            min_n = max(1, int(round(ratio * len(self.feats))))
            ok &= self.n_feats() >= min_n
        return ok

    def sub(self, rows: np.ndarray) -> pd.DataFrame:
        """把一批行号还原成带 `trade_date/stock_code` 的 DataFrame（索引与 X 同行序）。"""
        d = self.dates_arr[self.row_day[rows]]
        c = self.codes[self.row_code[rows]]
        return pd.DataFrame({"trade_date": d.astype(str), "stock_code": c})


# ---------------------------------------------------------------- 内部工具
def _pick_features(cfg: Cfg, avail: list[str], features: list[str] | None = None) -> list[str]:
    """从产物里的特征清单中选出本次要用的列（显式清单 ∩ 产物 − 排除清单）。"""
    want = features if features is not None else cfg.features
    have = set(avail)
    feats = [f for f in (want or avail) if f in have]
    excl = set(cfg.exclude_features)
    return [f for f in feats if f not in excl]


def _years_touching(meta: dict, start: str | None, end: str | None) -> list[int]:
    """挑出与 [start, end] 有交集的年份分区（不读文件，用 meta 里的逐年端点）。"""
    out = []
    for y, v in (meta.get("years") or {}).items():
        if start and str(v["end"]) < str(start)[:10]:
            continue
        if end and str(v["start"]) > str(end)[:10]:
            continue
        out.append(int(y))
    return sorted(out)


def _row_index(df: pd.DataFrame, dates: list[str], codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """行号 → 日索引 / 股票索引。网格完整时走 repeat/tile 快路（免掉 1200 万次 map）。"""
    C = len(codes)
    n_days = len(dates)
    if len(df) == n_days * C:
        cnt = df["trade_date"].value_counts(sort=False).reindex(dates)
        if len(cnt) == n_days and bool((cnt.to_numpy() == C).all()):
            rd = np.repeat(np.arange(n_days, dtype=np.int32), C)
            rc = np.tile(np.arange(C, dtype=np.int32), n_days)
            return rd, rc
    t_of = {d: i for i, d in enumerate(dates)}
    c_of = {c: i for i, c in enumerate(codes)}
    rd = df["trade_date"].map(t_of).to_numpy(dtype=np.int32, na_value=-1)
    rc = df["stock_code"].map(c_of).to_numpy(dtype=np.int32, na_value=-1)
    if (rd < 0).any() or (rc < 0).any():
        raise SystemExit("✘ 面板里有落在日期轴/股票轴之外的键 —— 产物坏了，请重跑 preparingdata.py")
    return rd, rc


def _keys_ok(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    """三份产物必须逐行同构：行数 + 首末键 + 日期集合 + 代码集合。"""
    if len(a) != len(b):
        raise SystemExit(f"✘ 面板网格不一致：{what} 行数 {len(a):,} ≠ {len(b):,}")
    if len(a) == 0:
        return
    if (a["trade_date"].iloc[0], a["stock_code"].iloc[0]) != \
       (b["trade_date"].iloc[0], b["stock_code"].iloc[0]):
        raise SystemExit(f"✘ 面板网格不一致：{what} 首行键不同")
    if not np.array_equal(np.sort(a["trade_date"].unique()), np.sort(b["trade_date"].unique())):
        raise SystemExit(f"✘ 面板网格不一致：{what} 的日期集合不同")
    if not np.array_equal(np.sort(a["stock_code"].unique()), np.sort(b["stock_code"].unique())):
        raise SystemExit(f"✘ 面板网格不一致：{what} 的股票集合不同")


def feature_set(cfg: Cfg, name: str, sample: bool = False) -> list[str]:
    """取 `meta.json:feature_sets` 里的**预制特征集**（由 `preparingdata.py` 算好）。

    动机：上游 230 个因子的时间覆盖长短不一（40 个晚起，最晚的 `rd_intensity` 到 2019-05 才有），
    所以"从某一年起可用的因子有哪些"是个**每次都要问的问题**。与其让每个单元自己去扫
    `per_factor_years`，不如在初加工时一次性算好、写进 meta —— 单元只写 `DATA={"features":"core_2018"}`。
    """
    root = panel_io.root_of(cfg, sample=sample)
    meta = panel_io.load_meta(root) or {}
    sets = meta.get("feature_sets") or {}
    if name not in sets:
        raise SystemExit(f"✘ 产物里没有特征集 {name!r}（可选：{sorted(sets) or '无'}）—— "
                         f"跑一次 `python preparingdata.py` 生成")
    return list(sets[name])


def load_for_unit(cfg: Cfg, unit, *, end: str | None = None, features: list[str] | None = None,
                  sample: bool = False, log=print) -> PanelData:
    """按**单元声明的数据窗**加载面板（`DATA = {...}`，见 `mx/unit.py:Unit.data_cfg`）。

    ★ 引擎里所有"要看面板"的地方都必须走这个函数，不能直接 `load()` ——
      否则训练用一个窗口、评价用另一个窗口，两边的行下标会对不上，
      而这类错**不会报错**，只会安静地算出一组错的指标。
    """
    kw = dict(unit.data_cfg(cfg))
    if end is not None:
        # 截断复算（audit-pit）要用更早的 end 覆盖单元声明，同时不能放宽单元的 start
        kw["end"] = end
    if features is not None:
        kw["features"] = features
    return load(cfg, sample=sample, log=log, **kw)


# ---------------------------------------------------------------- 加载
def load(cfg: Cfg, *, start: str | None = None, end: str | None = None,
         features: list[str] | None = None, sample: bool = False,
         data_root: str | None = None, log=print) -> PanelData:
    """加载面板。`start/end` 按日期切片（覆盖配置）；`features` 显式指定特征清单（消融用）；
    `sample=True` 读小样本（`trainingdata/sample/`，冒烟/调试用）。

    `data_root`：**换一份快照**（相对模块根，例如 `trainingdata_v2`）。用于
    「同一套配方、只换特征集」的 A/B —— 见 `prepared.prepare` 的 `root` 说明与
    `Unit.data_cfg` 的 `DATA = {"trainingdata": ...}`。
    """
    if data_root:
        root = (cfg.root / str(data_root)).resolve()
    else:
        root = panel_io.root_of(cfg, sample=sample)
    meta = panel_io.load_meta(root)
    if not meta:
        raise SystemExit(f"✘ 还没有初加工产物（{root}）—— 先跑 `python preparingdata.py`")
    cols = meta.get("columns") or {}
    avail_labels = list(cols.get("labels") or [])
    labels = [l for l in cfg.labels if l in set(avail_labels)]
    missing = [l for l in cfg.labels if l not in set(avail_labels)]
    if missing:
        log(f"  ⚠️ 配置里的标签在产物里不存在，已忽略：{missing}")
    if not labels:
        raise SystemExit(f"✘ 产物里没有任何配置的标签（产物有 {avail_labels}）")
    feats = _pick_features(cfg, list(cols.get("features") or []), features)
    if not feats:
        raise SystemExit("✘ 特征清单为空（检查 data.features / data.exclude_features）")

    start = start or (cfg.raw.get("data") or {}).get("start")
    end = end or (cfg.raw.get("data") or {}).get("end")
    years = _years_touching(meta, start, end)
    if not years:
        raise SystemExit(f"✘ 产物里没有落在 [{start}, {end}] 的年份分区"
                         f"（已有：{meta.get('built_years')}）")
    tag = "小样本" if sample else "全量"
    log(f"加载面板（{tag}）：{len(feats)} 个特征 + {len(labels)} 个标签 · "
        f"年份 {years[0]}~{years[-1]}"
        + (f" · 区间 {start} ~ {end}" if (start or end) else ""))

    Xdf = panel_io.read_frame(root, XK, years=years, columns=feats, start=start, end=end)
    if Xdf.empty:
        raise SystemExit(f"✘ 产物在 [{start}, {end}] 内没有任何行")
    Ydf = panel_io.read_frame(root, YK, years=years, columns=labels, start=start, end=end)
    Udf = panel_io.read_frame(root, UK, years=years, columns=["in_universe"], start=start, end=end)
    _keys_ok(Xdf, Ydf, "Y")
    _keys_ok(Xdf, Udf, UK)

    dates = sorted(set(Xdf["trade_date"].astype(str)))
    codes = np.asarray(sorted(set(Xdf["stock_code"].astype(str))), dtype=object)
    # ★ 校验代码轴与产物 meta 一致：价格层是按 **code 序号**取列的（模块② 的 Engine 顺序），
    #   一旦这里的顺序/集合漂移，价格会**静默错位**（不报错，数字全错）。宁可在这里炸。
    want_codes = [str(c) for c in ((meta.get("axis") or {}).get("codes") or [])]
    if want_codes and (len(want_codes) != len(codes) or any(a != b for a, b in zip(want_codes, codes))):
        raise SystemExit(
            f"✘ 面板的代码轴与产物 meta 不一致（产物 {len(want_codes)} 只 vs 本次 {len(codes)} 只）"
            f" —— 产物可能被改过或坏了，请 `preparingdata.py --check` 后再重跑")
    row_day, row_code = _row_index(Xdf, dates, codes)
    # ★ 完整性不变式：**完整网格**时行数必然 = 天数 × 只数。两者不等就说明产物里有重复行
    #   （或某天缺了一部分股票）—— 重复行会让下游 reshape 直接炸，或在更隐蔽的地方静默双计。
    #   这条检查只在"网格不完整"时才跑（完整时 0 成本），是增量的重复写入事故的兜底。
    if len(dates) * len(codes) != len(Xdf):
        key = row_day.astype(np.int64) * max(1, len(codes)) + row_code
        if len(np.unique(key)) != len(key):
            raise SystemExit(
                f"✘ 面板里有重复的 (交易日, 股票) 行（{len(key) - len(np.unique(key)):,} 行）"
                f" —— 产物坏了，请重跑 `preparingdata.py`（或 `--full`）")

    X = np.ascontiguousarray(Xdf[feats].to_numpy(dtype=np.float32, copy=False))
    del Xdf
    # ★ 夹取放在**载入时一次**，而不是训练时每个头各做一次。
    #   夹取是逐元素操作（不涉及任何跨样本统计量）⇒ 提前做与事后做结果逐位相同，
    #   但省掉了"每个头一份 6.6 GB 副本"。`preprocess.clip_features(already=...)` 靠这条生效。
    #   ⚠️ 不用 `out=X` 原地夹：`to_numpy(copy=False)` 返回的可能是**只读**视图
    #      （pyarrow 缓冲区的零拷贝视图），原地写会直接 ValueError。
    #      `np.clip` 不带 out 时分配一块新内存 —— 这一次分配在 15 年里只发生一次，
    #      而且**只有它**保证结果是可写的、后续所有头都能零拷贝复用。
    clip = (cfg.raw.get("preprocess") or {}).get("clip")
    if clip:
        X = np.clip(X, float(clip[0]), float(clip[1]))
    y = {l: np.ascontiguousarray(Ydf[l].to_numpy(dtype=np.float32, copy=False)) for l in labels}
    del Ydf
    universe = np.ascontiguousarray(Udf["in_universe"].to_numpy(dtype=bool, copy=False))
    del Udf

    panel = PanelData(
        dates=dates, codes=codes, feats=feats, X=X, y=y, universe=universe,
        row_day=row_day, row_code=row_code,
        _min_cov=float((cfg.raw.get("data") or {}).get("min_feature_coverage", 0.5)),
        meta={"start": dates[0], "end": dates[-1], "n_days": len(dates),
              "n_codes": len(codes), "sample": bool(sample),
              "min_feature_coverage": float((cfg.raw.get("data") or {})
                                            .get("min_feature_coverage", 0.5)),
              "panel_digest": meta.get("panel_digest"),
              "built_at": meta.get("built_at"), "source": str(root),
              "clipped": ([float(clip[0]), float(clip[1])] if clip else None)})
    panel.clipped = panel.meta["clipped"]
    panel.meta["digest"] = digest(panel)
    log(f"  网格：{len(dates)} 天 × {len(codes)} 只 = {panel.n_rows:,} 行 · "
        f"特征矩阵 {X.nbytes / 1e6:.0f} MB · 池内 {int(universe.sum()):,} 格 · "
        f"样本需特征覆盖率 ≥ {panel._min_cov:.0%}")
    for l in labels:
        m = panel.sample_mask(l)
        log(f"  标签 {l:16} 可用样本 {int(m.sum()):>9,} / {panel.n_rows:,}")
    return panel


def available_dates(cfg: Cfg, feature: str | None = None) -> list[str]:
    """初加工产物里**实际出现**的交易日（升序）。

    ★ 不用交易日历 —— 日历含未来 16 个交易日（模块② 踩过这个坑）；本函数只读产物的日期列。
    """
    root = panel_io.root_of(cfg)
    if not panel_io.load_meta(root):
        raise SystemExit(f"✘ 还没有初加工产物（{root}）—— 先跑 `python preparingdata.py`")
    days = panel_io.axis_days(root)
    if not days:
        raise SystemExit(f"✘ 产物里没有任何交易日：{root}")
    return days


def frame_for(panel: PanelData, rows: np.ndarray, values: np.ndarray,
              name: str = "value") -> pd.DataFrame:
    """把某批行号 + 数值组装成产物 DataFrame（4 列契约 + 当日截面 rank）。"""
    df = panel.sub(rows)
    df[name] = np.asarray(values, dtype=np.float32)
    df = df.rename(columns={name: "value"})
    df["rank"] = store.cs_rank_from_score(df["trade_date"], df["value"])
    df["trade_date"] = df["trade_date"].astype("string")
    df["stock_code"] = df["stock_code"].astype("string")
    df["value"] = df["value"].astype("float32")
    df["rank"] = df["rank"].astype("float32")
    return df.sort_values(["trade_date", "stock_code"], kind="stable").reset_index(drop=True)


def digest(panel: PanelData) -> str:
    """面板指纹 —— 回答"这次训练用的是哪份数据、哪些列、哪段区间"。

    ★ 不再哈希整个 X（全历史 11 GB，每次训练都算一遍是浪费）：改用
      初加工产物的面板指纹 + 本次的列清单与日期区间 —— 既便宜又更能定位。
    """
    h = hashlib.md5()
    h.update(str(panel.meta.get("panel_digest", "")).encode())
    h.update("|".join(panel.feats).encode())
    h.update(f"{panel.dates[0]}~{panel.dates[-1]}|{len(panel.dates)}|{len(panel.codes)}".encode())
    return h.hexdigest()
