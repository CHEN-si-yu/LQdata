"""V16 —— ★ **模板单元**：以后所有版本都从这一份复制出去改。

## 这一版为什么存在

2026-09-20 的顶层重构把项目定成三层，并定下两条规矩：

1. **顶层只允许 `trainingdata/`、`V*/`、`best/` 三个文件夹**。顶层的 `mx/`（引擎）、
   `main.py`（调度）、`tools/`、`tests/` 全部删除 —— 也就是说
   **`V16/mx/` 取代顶层 `mx/`，成为引擎的权威副本**。
2. **每个单元必须自包含**：只读外部数据源 `trainingdata/`，不依赖本目录以外的任何文件，
   自己就能完成**训练 + 回测**整条链路。

V16 就是这套规矩的第一个产物，也是往后的模板。

## 它跟 V12/V13/V14 有什么不同

那三个单元虽然也自带 `mx/`，但它们的 `run.py` / `analysis.py` 写的是

    ROOT = Path(__file__).resolve().parents[1]     # ✘ 指向**模块根**，不是单元自己

后果是：在仓内 `cd V12 && bash train.sh` 时，`import mx` 命中的是**顶层引擎**，
而顶层引擎的防分叉闸门（`mx/unit.py:load_unit`）会当场拒绝：

    ✘ 单元 `V12` 是**自包含**的（它自带引擎 …/V12/mx/），不能用模块根的 mx/ 跑

也就是说它们**在本仓内根本跑不起来**，只有 `python main.py <cmd> <unit>` 那条路（用 `.parent`）
能用。上一轮自包含化的验收之所以通过，是因为它先把顶层 `mx/` 改名藏起来了 ——
恰好绕过了这个 bug，**验收方式掩盖了缺陷**。

V16 把这行改成 `parents[0]`（= 单元自己），于是：

| 场景 | V12/V13/V14 | **V16** |
|:--|:--|:--|
| 仓内 `cd <单元> && python run.py 1` | ✘ 被闸门拒绝 | ✔ |
| 仓内 `cd <单元> && bash train.sh` | ✘ 每折都失败 | ✔ |
| 仓内 `cd <单元> && python analysis.py` | ✘ | ✔ |
| 仓内 `python main.py run <单元>` | ✔ | ✔ |
| 打包发出去（顶层 `mx/` 不存在） | ✔（靠巧合） | ✔（本来就对） |

## 配方：轻量但完整

**这一版不求效果**（用户 2026-09-20 明确："具体什么模型、效果差都无所谓"），
它要证明的是**整条链路在自包含条件下能跑通**：加载面板 → 训练 → 打分 → 评价 →
策略矩阵 → 回测 → 落盘。所以刻意选最便宜的一档：

| 维度 | 取值 | 为什么 |
|:--|:--|:--|
| 特征 | 306 个（`trainingdata` 里的全部） | 不引入额外变量 |
| 标签 | 5 个（1d/3d/5d/10d/20d） | 全标签覆盖，评价层会逐标签出 IC |
| 头 | `ridge_w` + `xgb` | 线性 + 树两条完全不同的路径，最能暴露链路问题 |
| 折 | **1 折**（`SEEDS = [42]`） | 4 折要跑 4 倍时间；模板不需要折间区间 |
| 主腿 | `label_ret_20d` | 与 V12/current 主线一致，榜单可直接摆在一起 |
| 数据窗 | 2018 起 | 与 V11/V12 同窗，且把早年覆盖率低的部分让出去 |

**要迭代出新版本**：`cp -r V16 V17`，改 `NAME`/`TITLE`，改 `heads()`（这是架构与超参的
唯一权威定义），跑 `bash train.sh` + `python analysis.py`。**只增不改** ——
改本目录会让已有结论失去对照。

## 这一版不做什么

- 不做特征筛选（306 全上）、不做多折集成、不做超参搜索 —— 那些是**后续版本**的变量，
  模板要干净。
- 不声明 `DATA={"features": ...}`：先用全量特征，特征集消融留给下一版。
  ★ 但 `trainingdata/meta.json:feature_sets` 里有一个 `sample_20`（20 个随机抽样的特征），
  想让这一版几秒钟跑完冒烟，加一行 `DATA = {"start": "2018-01-01", "features": "sample_20"}`
  即可 —— 那是**给模板用的快速通道**，不是结论。
"""
from __future__ import annotations

import os

from mx import combine as C
from mx.models import build

NAME = "V16"
TITLE = "★ 模板单元：自包含 · 轻量但完整（训练 + 回测全链路）"
DESCRIPTION = "2115 池 · 306 特征 · 2018 起 · ridge_w + xgb · 主腿 20d · 1 折"

# 全历史快照里只取 2018 起（与 V11/V12 同窗）。收窄窗口是**声明**，不会凭空造数据。
DATA = {"start": "2018-01-01"}

# 研究口径：固定 test 窗（样本外）。`best` 晋级时引擎会自动丢掉 test 窗，
# 所以这里照研究口径写即可（见 mx/unit.py:Unit.split_cfg 的生产覆盖）。
# 隔断由 purge 自动保证：T 的标签用到 open(T+1+h) ⇒ 每个标签按自己的 h 往后退。
SPLIT = {
    "mode": "date",
    "train_end": "2025-08-10",
    "test_start": "2025-09-01",
    "test_end": "2026-09-01",
    "valid_ratio": 0.20,
}

# ★ 单折。折数 = len(SEEDS)，所以这里就是 1 折（`mx/unit.py:folds`）。
SEEDS = [42]
LABELS = None                       # None = 用 conf 里的全部 5 个标签
PRIMARY_LABEL = "label_ret_20d"


def heads(cfg, seed: int):
    """本版的两个头 —— **架构与超参的唯一权威定义就在这里**（不要散回 conf.yaml）。

    `MX_THREADS` 由 `train.sh` 按并发折数下发：并发折 × 每折线程会互相抢核，
    而容器实际可用的核数远小于 `os.cpu_count()`（后者报的是宿主机）。
    """
    threads = int(os.environ.get("MX_THREADS", 8))
    return [
        build(cfg, "ridge_w", seed=seed, overrides={
            # 标签做截面秩高斯化：把重尾的收益压成有界目标，线性头对离群点敏感
            "label_transform": "rank_gauss",
            "alpha": float(cfg.raw["models"]["ridge"]["params"].get("alpha", 1.0)),
        }),
        build(cfg, "xgb", seed=seed, overrides={
            "label_transform": "rank_gauss",
            # 时间衰减 + 头部加权：越近的样本越重，且更看重"买对头部"
            "sample_weight": {"half_life": 250, "tail_frac": 0.20,
                              "tail_boost": 2.0, "tail_side": "top"},
            "n_estimators": 400,      # 比 conf 默认的 500 少一点，模板要跑得快
            "n_jobs": threads,
        }),
    ]


# 策略与模型**同步迭代**：改这一行 + 重跑 analysis 即可，不必重训。
# ★ 固定策略 `topn_hold {'n': 5, 'hold_days': 10}` 必须在列 —— 榜单拿它做同池横向对比。
STRATEGIES = [
    {"name": "top1_daily", "params": {}},                          # 基线：持股 top1、次日换手
    {"name": "top1_alt", "params": {}},                            # 对照：隔日换手
    {"name": "topn_hold", "params": {"n": 5, "hold_days": 10}},    # ★ 固定策略（跨模型可比）
    {"name": "topn_hold", "params": {"n": 5, "hold_days": 20}},
    {"name": "topn_hold", "params": {"n": 3, "hold_days": 10}},
    {"name": "top5_rank_exit", "params": {"n": 5, "m": 10}},       # 低换手：跌出前 10 才卖
    {"name": "topn_hold_margin", "params": {"n": 5, "margin": 0.20}},
    {"name": "cash", "params": {}},                                # 空仓对照（净值应恒为 1）
]


def combine(per_head: dict):
    """汇总口径：各头**逐日截面 z 后等权**（量纲无关，且只用到当日截面信息）。

    ★ 等权是**基线口径**，不是结论。已知缺陷（V10b/V11 的 A/B 抓到）：把一个 top1 崩掉的
      头等权混进来会拖死整个集成。合并规则本身是个待正面比较的版本变量 —— 留给后续版本。
    """
    return C.combine_equal(per_head)
