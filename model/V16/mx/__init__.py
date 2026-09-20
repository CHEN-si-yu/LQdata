"""模块③ 模型工程 —— 内部包。

分层（对标模块② 的 `fea/`）：
    config.py     配置与路径
    upstream.py   ★ 跨模块只读适配层（所有 `import fea.*` 都收敛在这一个文件里）
    store.py      产物读写（与因子侧**同契约**：4 列 parquet）
    data.py       因子面板加载（列栈 + 对齐校验 + 缓存）
    dataset.py    ★ 样本构造 + purged/embargo 时序切分（防泄漏的关键）
    preprocess.py 方向统一 / 缺失处理 / 特征初筛
    models/       模型实现（注册表 + linear / gbdt / nn 桩）
    train.py      多标签训练
    predict.py    推理 → 每日打分
    evaluate.py   评价（逐日截面 IC/RankIC → ICIR/t、分层、换手）
    backtest.py   可交易性 + 等权 top-N 组合回测
"""
