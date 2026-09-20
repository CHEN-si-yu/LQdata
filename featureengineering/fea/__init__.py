"""fea —— 因子工程核心包（模块②）。

对应模块① 的 `lingqi/`。因子实现放在项目根目录的 `factors/`，
由 `main.py` import 触发注册；本包**不**反向依赖 `factors/`。
"""

from .config import Config, load, setup_logging          # noqa: F401
from .context import FactorContext                        # noqa: F401
from .spec import FactorSpec, register                    # noqa: F401
