"""模型基类 + 注册表 —— 对标模块② 的 `fea/spec.py`：**声明式、加模型不改框架**。

★ CPU/GPU 两手准备的两个落点：
  ① `resolve_device()` —— 设备只由"配置 + 运行时可用性"决定，代码里**不出现**硬编码设备；
  ② 模型权重一律 `save/load` 到 `data/models/<模型>/<标签>/`，换机器只需换 torch 的 CUDA 轮子，
     训练/推理代码一行不改。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..config import Cfg

REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """把一个模型实现登记进注册表（用它的 `type` 作为键）。"""
    REGISTRY[cls.type] = cls        # type: ignore[attr-defined]
    return cls


def resolve_device(cfg: Cfg) -> str:
    """训练设备：`auto` → 有 CUDA 就用 cuda，否则 cpu（**CPU 上先跑通，GPU 上不改代码**）。"""
    want = str(cfg.raw["train"].get("device", "auto")).lower()
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:               # 没装 torch（线性/GBDT 不需要它）→ 纯 CPU 路径
        return "cpu"


class ModelBase:
    type = "base"
    #: 缺失值怎么处理 —— 由模型类自己声明，`preprocess.fill_mode_for` 读它。
    #: `"lgbm"` = 保留 NaN（树原生处理）；`"linear"`/`"nn"` = 填固定常数 0.5 + 一列缺失占比。
    fill_mode = "lgbm"
    #: 打分时是否需要"行→交易日"的映射。默认不需要；参考工程复刻的 `nn_v8` 需要 ——
    #: 它的打分是 `z(lin) + z(top)`，而 z 是**逐日**做的，不知道日边界就标准化错了对象。
    #: `mx/train.py` 只在为真时才把 `day` 传进 `predict()`（不改其他模型的行为）。
    predict_needs_day = False

    def __init__(self, name: str, params: dict, cfg: Cfg, seed: int | None = None):
        self.name = name
        self.params = dict(params or {})
        self.cfg = cfg
        # ★ 折 = 「种子 × 划分」的一次独立训练；模型一律读 self.seed（不要读 cfg.seed），
        #   否则同一单元的多折会全部退化成同一个模型（参考工程「折=种子」惯例）。
        self.seed = int(seed) if seed is not None else int(cfg.seed)
        # ★ 可选：本头只训这几个标签（`None` = 单元声明的全部）。
        #   神经网络一个标签要训一遍，5 个标签就是 5 倍成本 —— 靠这个收窄。
        self.labels = self.params.get("labels")

    # ---- 子类必须实现
    def fit(self, X: np.ndarray, y: np.ndarray,
            Xv: np.ndarray | None = None, yv: np.ndarray | None = None,
            *, sample_weight: np.ndarray | None = None,
            day: np.ndarray | None = None, day_v: np.ndarray | None = None,
            raw: np.ndarray | None = None, raw_v: np.ndarray | None = None) -> "ModelBase":
        """训一个 (头, 标签)。

        参数里**只有 X / y / Xv / yv 是必须的**，其余按模型需要取用：

        | 参数 | 含义 | 谁需要 |
        |:--|:--|:--|
        | `sample_weight` | 样本权重（时间衰减 × 尾部，来自 `mx/labels.py`） | 加权岭 / 加权 GBDT / NN |
        | `day` / `day_v` | 行 → 交易日下标，**按日连续升序** | 排序目标、逐日 softmax、按日分批的 NN |
        | `raw` / `raw_v` | **未变换**的原始收益率 | NN 里"直接优化组合收益"那一项 |

        ★ `y` 是**变换后**的训练目标（可能已秩高斯化），`raw` 才是真实收益。
          两者语义不同、不可互换 —— 拿秩高斯值去算组合收益会得到一个无量纲的怪数。
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ---- 可选
    def importance(self) -> np.ndarray | None:
        """特征重要性（线性= |系数|，树=增益）。用于和模块② 的因子强弱对照。"""
        return None

    def loss_desc(self) -> str | None:
        """本头**自己的**损失构成（一行文字），给日志与榜单归因用；不适用就返回 None。

        ★ 为什么需要这个钩子：`labels.describe` 只看得懂**数据集侧**的两个口径
          （`label_transform` / `sample_weight`）。神经网络还有一组**模型侧**的损失权重
          （`w_mse` / `w_ic` / `w_rankic` / `w_top` / `w_listnet` / `rdrop`）——
          不报出来的话，日志会把这些头一律报成"原值标签·等权（V2 口径）"，
          归因时就会把结论安到错误的配方上（§11-14 那条"日志说错话比不打印更危险"）。
        """
        return None

    def info(self) -> dict:
        return {"model": self.name, "type": self.type, "seed": self.seed,
                "params": self.params}

    # ---- 存取（pickle；模型都很小，不引入额外依赖）
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, cfg: Cfg) -> "ModelBase":
        with open(path, "rb") as f:
            m = pickle.load(f)
        m.cfg = cfg
        return m


def build(cfg: Cfg, model_name: str, seed: int | None = None,
          overrides: dict | None = None) -> ModelBase:
    """按 `conf/config.yaml` 的 `models.<name>.type` 实例化（可注入 seed / 覆盖超参）。

    ★ 单元（`Model/V{N}/model.py`）就是通过 `build(cfg, name, seed=...)` 声明"本版用哪些头"的 ——
      架构/超参的唯一权威定义在单元的 model.py，而不是散落在 conf 里。
    """
    spec = (cfg.model_cfgs.get(model_name) or {})
    mtype = str(spec.get("type", ""))
    if not mtype:
        raise SystemExit(f"配置里没有模型 {model_name}（可选：{sorted(cfg.model_cfgs)}）")
    cls = REGISTRY.get(mtype)
    if cls is None:
        raise SystemExit(f"模型类型 {mtype} 未实现（已注册：{sorted(REGISTRY)}）")
    params = dict(spec.get("params") or {})
    if overrides:
        params.update(overrides)
    return cls(model_name, params, cfg, seed=seed)
