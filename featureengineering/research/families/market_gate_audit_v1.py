"""市场因子审计别名：原样复算已批准的市场因子（复审 M1 工程因果门）。

与 `legacy_gate_audit_20261002` 同一模式：builder 复核正式定义的 AST 哈希后
**返回原函数**，使引擎沿用真实依赖与配方（inspect 源文件、recipe 均指向正式实现）。
别名只用于隔离命名空间下的截断复算 + 两阶段增量重放证明（= 股票 G1 的市场版），
不改变任何正式对象、不写生产 data/ 与 state/。

参数（family allowed_variants 逐项预注册）：
- original_name：正式市场因子名（conf/production.json 中的 mkt_*）
- definition_sha256：预注册时正式定义的哈希；对不上即拒绝（防审计对象漂移）
"""


def build(parameters):
    import factors  # noqa: F401  （确保正式注册表已装载）
    from fea.config import load
    from fea.release import production_specs, definition

    cfg = load()
    original = next(s for s in production_specs(cfg) if s.name == parameters["original_name"])
    if not original.is_market:
        raise ValueError("市场审计别名要求 is_market 原对象")
    if definition(original, cfg) != parameters["definition_sha256"]:
        raise ValueError("原正式因子在审计预注册之后发生了变化")
    # 返回原可调用对象：保留真实源身份与依赖语义。
    return original.fn
