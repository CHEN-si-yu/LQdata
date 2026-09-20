"""训练 —— **多标签一起训**：同一份 X，5 个期限各训一个模型。

一次 `train_fold` 的输出（全部落在**单元目录**里，见 `mx/state.py`）：
  · 每个 (头, 标签) 的验证/测试指标（逐日截面 RankIC/IC → ICIR/t、分层、单调、换手）
  · 模型权重 → `<单元>/model_train/fold<k>/<头>__<标签>/model.pkl`
  · 指标 → `<单元>/model_train/fold<k>/<头>__<标签>/eval.json`
  · 全面板打分（4 列契约）→ `<单元>/model_train/fold<k>/score__<头>__<标签>/`
  · 折汇总 → `<单元>/model_train/fold<k>_summary.json`

★ 可复现：同种子两次训练结果逐位一致（sklearn/lightgbm 都吃 `random_state`；不做任何随机切分）。
★ 设备：`resolve_device(cfg)` —— 线性/GBDT 走 CPU（LightGBM 官方 wheel 就是 CPU 版），
  NN 走 auto（有卡即 cuda）。代码里不出现硬编码设备。
★ 无 test 的单元（实战 `best`）：`split.test is None`，这里只评验证窗，不再假装有样本外。
"""
from __future__ import annotations

import time

import numpy as np

from . import dataset, evaluate, labels as L, preprocess, state
from .config import Cfg
from .data import PanelData, digest, frame_for, load
from .models import build
from .models.base import resolve_device


def _prepare_X(panel: PanelData, mode: str, clip) -> np.ndarray:
    """把一个头需要的输入矩阵准备好（clip 已在载入面板时做过，这里通常零拷贝）。"""
    X = preprocess.clip_features(panel.X, clip, already=panel.clipped)
    X, notes = preprocess.fill_for(X, mode)
    return X


def _predict(model, X, rows, panel) -> np.ndarray:
    """打分 —— 需要"行→交易日"映射的头（`predict_needs_day`）才多传一个 `day`。

    ★ 为什么要有这个分支：`nn_v8`（参考工程复刻）的打分是 `z(lin) + z(top)`，
      而 z 是**逐日**做的 —— 不告诉它日边界，它就会把整个评价块当成一天去标准化，
      排名会整体走样（且**不报错**）。默认 False ⇒ 其余模型族的调用一字不变。
    """
    if getattr(model, "predict_needs_day", False):
        return model.predict(X[rows], day=panel.row_day[rows])
    return model.predict(X[rows])


def train_one(cfg: Cfg, panel: PanelData, model_name: str, label: str,
              split: dataset.Split, X: np.ndarray, log=print,
              model=None, seed: int | None = None,
              out_model=None, out_eval=None) -> dict:
    """训一个 (头, 标签)，返回指标 dict。

    `model` 传入时用外部构造好的实例（单元就是这么做的：`{单元}/model.py` 定义头）。
    `out_model/out_eval` 是落盘位置；给了才写。

    ★ 本函数是**损失函数设计**的落点：标签变换与样本权重在这里算出来、传给模型，
      模型自己只负责"拿 X 和 y 去拟合"。这样 `ridge` / `gbdt` /  `nn` 三个族
      可以共享同一条损失口径，对比才有意义（`mx/labels.py` 有全部推导）。
    """
    t0 = time.time()
    model = model if model is not None else build(cfg, model_name)
    # ★ 摊平成一份 spec，目标与权重共用（`labels.loss_spec` 有为什么必须这么做的说明）
    spec = L.loss_spec(model.params)

    tr, va = split.train, split.valid
    # ★★ 可执行性掩码：**把"买不进"的行从训练/验证样本里切掉**（测试集不动）。
    #   为什么切行而不是压权重：LightGBM 对 weight=0 的行仍计入 min_data_in_leaf、仍参与分箱，
    #   "零权重"表达不了"这不是样本"（`labels.make_buyable` 有完整说明）。
    #   README §8.4：模型选中的 top1 有 29.4% 次日买不进（基准 0.70%），
    #   其中 22.7% 来自"T 日收盘涨停"那一类 —— 掩码直接对着它去。
    n_drop_tr = n_drop_va = 0
    buy = L.make_buyable(panel, spec)
    if buy is not None:
        k_tr, k_va = buy[tr], buy[va]
        n_drop_tr, n_drop_va = int((~k_tr).sum()), int((~k_va).sum())
        tr, va = tr[k_tr], va[k_va]
    day_tr, day_va = panel.row_day[tr], panel.row_day[va]
    y_raw_tr, y_raw_va = panel.y[label][tr], panel.y[label][va]

    y_tr = L.make_target(y_raw_tr, day_tr, spec)
    y_va = L.make_target(y_raw_va, day_va, spec)
    # ★ 尾部权重的排序依据用**原始收益率**：头部的投资含义是"未来涨得最多的那几只"
    w = L.make_weights(panel.dates_arr, day_tr, y_raw_tr, spec)

    Xtr, Xva = X[tr], X[va]
    n_te = len(split.test) if split.test is not None else 0
    log(f"   训练 {model_name}/{label}: train {len(Xtr):,} · valid {len(Xva):,} · "
        f"test {n_te:,}" + ("（本口径无 test）" if split.test is None else ""))
    # ★ 损失口径要报**两段**：数据集侧（标签变换/样本权重）在 `labels.describe`，
    #   模型侧（神经网络的 w_mse/w_ic/w_rankic/w_top/w_listnet/rdrop）在模型自己的
    #   `loss_desc()`。只报前者会把神经网络头一律谎报成"V2 口径"（§11-14）。
    ld = getattr(model, "loss_desc", None)
    ld = ld() if callable(ld) else None
    log(f"     损失口径：{L.describe(spec)}" + (f" · {ld}" if ld else ""))
    if buy is not None:
        log(f"     可执行性掩码：剔掉 train {n_drop_tr:,} 行 / valid {n_drop_va:,} 行"
            f"（占 {n_drop_tr / max(1, n_drop_tr + len(Xtr)):.2%} / "
            f"{n_drop_va / max(1, n_drop_va + len(Xva)):.2%}）· 测试集不动")
    model.fit(Xtr, y_tr, Xva, y_va, sample_weight=w, day=day_tr, day_v=day_va,
              raw=y_raw_tr, raw_v=y_raw_va)
    sec = time.time() - t0

    out: dict = {"model": model_name, "label": label, "type": model.type,
                 "device": resolve_device(cfg), "seconds": round(sec, 2),
                 "seed": int(seed if seed is not None else cfg.seed),
                 "split": split.as_dict(),
                 "n_features": int(X.shape[1]), "model_info": model.info(),
                 "loss_spec": L.describe(spec) + ((f" · {ld}") if ld else ""),
                 "n_train_dropped": n_drop_tr, "n_valid_dropped": n_drop_va}
    # ★ 无 test 的单元只评验证窗（不拿训练段冒充样本外）
    blocks = [("valid", split.valid)] + ([("test", split.test)] if split.test is not None else [])
    for block, rows in blocks:
        if rows is None or len(rows) == 0:
            continue
        sc = _predict(model, X, rows, panel)
        out[block] = evaluate.evaluate(panel.dates_arr[panel.row_day[rows]],
                                       panel.codes[panel.row_code[rows]], sc,
                                       panel.y[label][rows], cfg)
    # 特征重要性（前 20）——和模块② 的单因子强弱对照
    # ★ 线性头的输入比 `panel.feats` **多一列**（`preprocess.fill_for` 追加的"缺失占比"），
    #   所以判据要用 `X.shape[1]` 而不是 `len(panel.feats)` —— 否则 ridge 永远报不出重要性。
    imp = model.importance()
    if imp is not None and len(imp) == X.shape[1]:
        extra = X.shape[1] - len(panel.feats)
        base = panel.feats + [f"__missing_ratio_{k}__" for k in range(max(0, extra))]
        idx = np.argsort(-np.asarray(imp))[:20]
        out["top_features"] = [{"factor": base[i], "importance": float(imp[i])} for i in idx]
        out["n_importance_cols"] = len(imp)
    te = out.get("test") or {}
    log(f"     {model_name}/{label} 用时 {sec:.1f}s · "
        f"valid RankIC {evaluate.fmt((out.get('valid') or {}).get('rankic_mean'))} · "
        f"ICIR {evaluate.fmt((out.get('valid') or {}).get('rankic_icir'), 2)} · "
        f"test RankIC {evaluate.fmt(te.get('rankic_mean'))}")
    if out_model:
        model.save(out_model)
    if out_eval:
        state.save_json(out_eval, out)
    return out


# ================================================================ 单元（V{N}）训练
def train_fold(cfg: Cfg, unit, fold: int, smoke: bool = False, sample: bool = False,
               log=print) -> dict:
    """训练**一个折** —— 折 = 一次「种子 × 划分」的独立训练，产物落进单元目录。

    这是 `{单元}/run.py` 背后干的全部事情：单元只负责声明"本版有哪些头 + 什么切分"，
    训练/落盘/打分/记账的机制一律在引擎里（纪律：`mx/` 共享，版本只增不改）。
    """
    from . import store
    from . import unit as U
    seed = U.fold_seed(unit, fold)
    if getattr(unit, "archived", False):
        raise SystemExit(f"✘ {unit.name} 是**已冻结的历史单元**（ARCHIVED）—— 不要在它上面训练/重跑。"
                         f"现行口径请新建 V{{N}}。")
    if smoke and sample:
        raise SystemExit("✘ --smoke 与 --sample 不要同时用：两者的产物树不同，混着来会分不清")
    t0 = time.time()
    cfg.ensure_dirs()
    state.unit_ensure_dirs(cfg, unit.name, smoke, sample)
    sp_cfg = unit.split_cfg(cfg)
    log(f"══ 单元 {unit.name} · 折 {fold}/{len(unit.seeds)}（seed={seed}）"
        f"{' · smoke' if smoke else ''}{' · 小样本' if sample else ''} · "
        f"引擎 {U.engine_hash(cfg.root)}")
    if sp_cfg.get("mode") == "date":
        log(f"  切分口径：train_end={sp_cfg.get('train_end') or '数据末日'} · "
            f"test={sp_cfg.get('test_start') or '无'} ~ {sp_cfg.get('test_end') or '无'}"
            + ("　🚀 实战口径（无 test，全量数据训练）" if unit.is_production else ""))
    panel = load(cfg, sample=sample, log=log, **unit.data_cfg(cfg))
    heads = unit.heads(cfg, seed)
    if smoke:                              # 冒烟：把树模型压到几秒
        for m in heads:
            if m.type in ("gbdt", "gbdt_w", "gbdt_rank", "gbdt_focus", "xgb", "xgb_rank"):
                m.params.update({"n_estimators": 40, "learning_rate": 0.1})
            elif m.type in ("nn", "nn_v8", "nn_wpcc"):
                # ★ 新增的神经网络类型必须**逐个登记**，否则冒烟会按全量跑
                #   （2026-09-19 复刻 nn_v8/nn_wpcc 时实测踩到：`--smoke` 跑成了
                #   与正式训练同价，白烧半小时 GPU）
                m.params.update({"epochs": 2, "hidden": [64, 16],
                                 "top_hidden": [16, 8], "hidden1": 64, "hidden2": 16,
                                 "soft_rank_max": 200})

    min_days = 5 if smoke else dataset.MIN_DAYS_PER_BLOCK
    # 「有特征」的格子（无特征处打分不落盘）—— 用 PanelData 缓存的那份，别重算。
    # ★★ 2026-09-18 修：原先写的是 `n_feats() >= 1`，比训练样本的阈值（min_feature_coverage）
    #   松得多 —— **同一份数据两把尺子**。现在与 `sample_mask` 共用 `avail_mask()`。
    #   实测这条只影响 2 个格子（README §8.4「归因修正」）⇒ 报成"正确性收口"，别当提升。
    avail = panel.avail_mask()
    all_labels = list(unit.my_labels(cfg))
    results: list[dict] = []

    # ★ 循环顺序是「头在外、标签在内」而不是反过来，理由只有一条：**内存**。
    #   每个头都要一份自己的预处理副本（线性/神经网络要填 NaN 再拼一列，就是一次全量拷贝），
    #   全历史面板一份 ≈ 6.6 GB。头在外 ⇒ 任何时刻只持有一份；头在内 ⇒ 同时持有 N 份。
    #   `make_splits` 只是行下标运算（毫秒级），每个头重算一次完全可以接受。
    for m in heads:
        my_labels = [lb for lb in all_labels if not m.labels or lb in m.labels]
        if not my_labels:
            log(f"  ⊘ 头 {m.name} 声明要训的标签一个都不在单元的标签列表里 —— 跳过")
            continue
        mode = preprocess.fill_mode_for(cfg, m.name)
        Xm = _prepare_X(panel, mode, cfg.raw["preprocess"].get("clip"))
        log(f"  头 {m.name}（{m.type}）· 填充 {mode} · X {Xm.shape} · 标签 {my_labels}")
        for label in my_labels:
            try:
                splits = dataset.make_splits(panel, label, cfg, log=log, min_days=min_days,
                                             fold=fold, n_folds=len(unit.seeds), split=sp_cfg)
            except SystemExit as exc:
                log(f"  ⊘ {label} 跳过：{exc}")
                continue
            for sp in splits:
                try:
                    r = train_one(cfg, panel, m.name, label, sp, Xm, log=log,
                                  model=m, seed=seed,
                                  out_model=state.unit_model_path(cfg, unit.name, fold, m.name,
                                                                  label, smoke, sample),
                                  out_eval=state.unit_eval_path(cfg, unit.name, fold, m.name,
                                                                label, smoke, sample))
                except NotImplementedError as exc:
                    log(f"    ⊘ {m.name}/{label} 未实现：{str(exc)[:100]}")
                    continue
                r["fold"] = fold
                r["sample"] = bool(sample)
                results.append(r)
                # ---- 全面板打分（4 列契约）→ analysis 的输入
                # ★ 这里打的是**整块面板**，所以"行→交易日"就是 panel.row_day 本身
                sc = (m.predict(Xm, day=panel.row_day) if getattr(m, "predict_needs_day", False)
                      else m.predict(Xm)).astype(np.float32)
                sc = np.where(avail, sc, np.nan)
                df = frame_for(panel, np.arange(panel.n_rows), sc, name="value")
                outp = state.unit_fold_score_path(cfg, unit.name, fold, m.name, label, smoke, sample)
                store.write_frame(outp, df)
                log(f"     打分落盘 {outp.relative_to(cfg.unit_dir(unit.name))}"
                    f"（非空 {np.isfinite(sc).mean():.1%}）")
        del Xm                          # ★ 显式释放，下一份副本才进来

    summary = {"unit": unit.name, "fold": fold, "seed": seed, "smoke": bool(smoke),
               "sample": bool(sample), "split": sp_cfg,
               "finished_at": state.now(), "seconds": round(time.time() - t0, 1),
               "engine_hash": U.engine_hash(cfg.root),
               "panel": {"n_days": len(panel.dates), "n_codes": int(len(panel.codes)),
                         "n_feats": len(panel.feats), "digest": digest(panel),
                         "panel_digest": panel.meta.get("panel_digest"),
                         "start": panel.dates[0], "end": panel.dates[-1]},
               "n_runs": len(results), "results": results}
    state.save_json(state.unit_fold_summary_path(cfg, unit.name, fold, smoke, sample),
                    summary)
    log(f"  ✔ 折 {fold} 完成：{len(results)} 个 (头 × 标签) · 用时 {summary['seconds']}s")
    return summary
