"""配置与路径 —— 全模块唯一的路径真相源。

## 两种运行形态（`ROOT` 由 `mx/` 的位置决定，见下）

**研究形态**（模块根）：`mx/` 在模块根，单元平铺同级，共享一份 `trainingdata/`。

    model/
      main.py  preparingdata.py  requirements.txt  README.md
      mx/                    ← 引擎（本文件 + conf.yaml 都在这）
      trainingdata/          ← 初加工产物（X / Y / universe / P / sample / meta.json）
      V1/ V2/ …              ← 研究单元（一版一个目录，只增不改）
      best/                  ← 实战单元

**自包含形态**（单元包）：`mx/` 与 `main.py` 被**逐字复制**进单元目录，
单元因此不依赖任何外部文件夹（唯一例外是 `trainingdata/`）。

    V12/
      mx/                    ← 自带引擎（与根 mx/ 同一份代码）
      main.py  model.py  run.py  train.sh  analysis.py  README.md
      trainingdata/  或  ../trainingdata/     ← 二选一，见 `_resolve_trainingdata`

★ `ROOT = mx/..` 这条规则让**两种形态自动成立**：`mx/` 复制到哪，哪里就是根。
  这正是"单元可独立拷贝"能成立的原因 —— 不需要任何环境变量或配置改写。

与模块② 的 `fea/config.py` 同构：`Cfg.raw` 保留原始 dict，常用项做成属性。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent          # = mx/ 的上一级
BEST = "best"          # 实战单元名（有生产口径覆盖：无 test、全量数据）
CONF = ROOT / "mx" / "conf.yaml"


class Cfg:
    def __init__(self, raw: dict):
        self.raw = raw
        self.root = ROOT
        self._td: Path | None = None          # trainingdata 解析结果（构造后算一次）

    # ---- 路径
    def _p(self, key: str) -> Path | None:
        """配置里的相对路径。**缺键返回 None**（不抛 KeyError）。

        ★ 为什么容错：自包含单元的 `conf.yaml` 里**没有** `factors_*` 三条
          （它不该知道上游在哪，那是构建期的事）。而 `Cfg` 是同一份代码，
          属性被谁碰一下就 KeyError 会让整个单元起不来。
          运行期没有任何代码会去读 `factors_*`；真读了拿到 None 会在调用处炸，
          报错位置比在这里炸清楚得多。
        """
        v = (self.raw.get("paths") or {}).get(key)
        return (self.root / v).resolve() if v else None

    @property
    def factors_dir(self) -> Path | None:
        """模块② 的因子产物目录（只读，**构建期**用）。"""
        return self._p("factors")

    @property
    def factors_state(self) -> Path | None:
        return self._p("factors_state")

    @property
    def factors_root(self) -> Path | None:
        """模块② 工程根目录 —— 只为 `import fea.*` 用（见 upstream.py，构建期）。"""
        return self._p("factors_root")

    def _resolve_trainingdata(self) -> Path:
        """产物根 —— **三候选按序取第一个真有 `meta.json` 的**。

        1. `$MX_DATA`（显式指定，**权威**：设了就用它，不做回落 ——
           否则设错路径会静默去读另一份数据、照样出数）
        2. `<root>/trainingdata`      —— 单元自带（打包发送时把数据放这里）
        3. `<root>/../trainingdata`   —— 在模块根下原位运行

        ★ 判据必须是 **`meta.json` 存在**，不是"目录存在"。踩过的形状：
          任何一个 `mkdir(trainingdata)`（`ensure_dirs` 曾经就会）都会造出一个
          空目录，于是候选 2 抢先命中、单元永远读不到真数据 ——
          而报错是"没有 meta.json"，把排查方向指到完全错的地方。
        """
        env = os.environ.get("MX_DATA")
        if env:
            return Path(env).expanduser().resolve()
        for c in (self.root / "trainingdata", self.root.parent / "trainingdata"):
            if (c / "meta.json").exists():
                return c.resolve()
        # 都没有：回落到第二候选，让下游给出"没有 meta.json"的可读报错
        return (self.root / "trainingdata").resolve()

    @property
    def trainingdata(self) -> Path:
        """初加工产物根（`preparingdata.py` 的落点，模型侧只读）。"""
        if self._td is None:
            self._td = self._resolve_trainingdata()
        return self._td

    @property
    def trainingdata_sample(self) -> Path:
        return self.trainingdata / "sample"

    @property
    def units_root(self) -> Path:
        """单元目录的父目录（现在 = 模块根，单元平铺：`V1/`、`V2/`、`best/`）。"""
        return self._p("units")

    @property
    def best_dir(self) -> Path:
        """**实战区**：当前生效的单元（由 `main.py promote` 晋级而来）。"""
        return self._p("best")

    def unit_dir(self, name: str) -> Path:
        name = str(name)
        cand = self.best_dir if name == BEST else self.units_root / name
        if (cand / "model.py").exists():
            return cand
        # ★ 自包含形态的自指回退：单元包里 `mx/` 与 `model.py` 同级，
        #   于是 `units_root`（`.`) 指向单元自己，`units_root/<name>` 自然不存在。
        #   此时"这个单元"就是 `root` 本身。
        if self.root.name == name and (self.root / "model.py").exists():
            return self.root
        return cand

    @staticmethod
    def is_selfcontained(d: Path) -> bool:
        """该单元目录是否**自带引擎**（`<单元>/mx/` 存在）。

        自带引擎 = 它是自包含交付物，应当用**它自己的** mx/ 跑。
        """
        return (Path(d) / "mx").is_dir() and (Path(d) / "model.py").exists()

    def unit_names(self) -> list[str]:
        """所有单元名 = 模块根下带 `model.py` 的目录（按名字排序，`best` 永远最后）。

        ★ **跳过自带引擎的目录**：那些是自包含单元，要用它们自己的 `mx/` 跑
          （`cd <单元> && python analysis.py`）。列在这里会诱导人从模块根发命令，
          而模块根的 `mx/` 是**另一份引擎** —— 见 `load_unit` 里的硬闸门。
        """
        out = []
        for p in sorted(self.units_root.iterdir()):
            if not p.is_dir() or p.name.startswith((".", "_")):
                continue
            if not (p / "model.py").exists():        # 只认有 model.py 的目录
                continue
            if self.is_selfcontained(p):             # 自包含单元不归模块根管
                continue
            out.append(p.name)
        out = [n for n in out if n != BEST]
        if (self.best_dir / "model.py").exists() and not self.is_selfcontained(self.best_dir):
            out.append(BEST)
        return out

    # ---- 常用配置项
    @property
    def labels(self) -> list[str]:
        return list((self.raw.get("data") or {}).get("labels") or [])

    @property
    def features(self) -> list[str] | None:
        f = (self.raw.get("data") or {}).get("features")
        return list(f) if f else None

    @property
    def exclude_features(self) -> list[str]:
        return list((self.raw.get("data") or {}).get("exclude_features") or [])

    @property
    def min_cross_section(self) -> int:
        return int((self.raw.get("data") or {}).get("min_cross_section", 100))

    @property
    def split_cfg(self) -> dict:
        """切分的**全局默认**；单元可以有自己的 `SPLIT` 覆盖它（见 mx/unit.py）。"""
        return dict(self.raw.get("split") or {})

    @property
    def seed(self) -> int:
        return int((self.raw.get("train") or {}).get("seed", 42))

    @property
    def device(self) -> str:
        return str((self.raw.get("train") or {}).get("device", "auto"))

    @property
    def model_cfgs(self) -> dict[str, dict]:
        return dict(self.raw.get("models") or {})

    def label_horizon(self, label: str) -> int:
        """`label_ret_5d` → 5（用于 purge/embargo 与持仓天数）。"""
        s = str(label).rsplit("_", 1)[-1]
        return int(s[:-1]) if s.endswith("d") and s[:-1].isdigit() else 1

    def ensure_dirs(self) -> None:
        """确保产物落点存在。

        ★ **不 mkdir `trainingdata`**（2026-09-19 改）。它原先会建，
          而 `_resolve_trainingdata` 按"有没有 meta.json"选候选 —— 一旦这里造出空目录，
          单元就会命中那个空目录、永远读不到真数据，报错还指向"没有 meta.json"。
          这个 mkdir 本来就是多余的：写产物的 `panel_io.atomic_parquet` 自己会
          `mkdir(parents=True)`。所以删掉它对现有流程零影响，只消除一个陷阱。
        """
        return None


def load(path: Path | None = None) -> Cfg:
    p = Path(path) if path else CONF
    raw: dict[str, Any] = {}
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Cfg(raw)
