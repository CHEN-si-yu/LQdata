"""生成市场因子复审的追溯登记定义（conf/research/market_review_20261003/）。

- 8 个族（按既有自然聚类）各一条 Idea/Family；61 个候选 = 逐因子审计别名
  （implementation 指向 research.families.market_gate_audit_v1:build）。
- 候选参数 {original_name, definition_sha256} 从 conf/production.json 读取；
  spec 元数据（deps/start/warmup/…）镜像自正式注册表 —— 注册时 builder 会复核哈希，
  任何漂移即拒绝。
- units/availability/missing_policy 如实注明「与正式因子一致，本候选仅做逐值复算」。
- 本脚本只读生产注册表与配置，只写 conf/research/market_review_20261003/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "conf" / "research" / "market_review_20261003"

FAMILIES = [
    {
        "key": "breadth_daily",
        "names": ["mkt_up_ratio", "mkt_strong_up_ratio", "mkt_strong_down_ratio",
                  "mkt_dispersion", "mkt_above_ma20_ratio", "mkt_above_ma60_ratio",
                  "mkt_new_high60_ratio", "mkt_new_low60_ratio",
                  "mkt_amount_hhi", "mkt_illiquidity_median"],
        "hypothesis": "固定池收盘横截面结构量（涨跌家数、离散度、均线位置、新高新低、成交集中度、非流动性）"
                      "对未来 1/5 日等权市场收益存在可检验的（方向逐项预登记的）关系；"
                      "其中透明描述量可作为市场状态上下文使用。",
        "mechanism": "宽度类量刻画当日市场参与结构与风险结构（扩散度、极端度、位置与集中度）；"
                     "预测方向与先验强度按因子在 freeze 方向表逐项预登记，不依据事后结果调整。",
        "fields": ["stock_daily.open", "stock_daily.close", "stock_daily.vol",
                   "stock_daily.amount", "stock_adj_factor.adj_factor"],
        "baseline": ["mkt_up_ratio", "mkt_dispersion"],
    },
    {
        "key": "breadth_am",
        "names": ["mkt_am_up_ratio", "mkt_am_strong_up_ratio", "mkt_am_strong_down_ratio",
                  "mkt_am_ew_return", "mkt_am_dispersion", "mkt_am_downside_rms",
                  "mkt_am_up_amount_share", "mkt_am_amount_ratio20",
                  "mkt_am_reversal_ratio", "mkt_am_range_mean"],
        "hypothesis": "上午（11:30 截面）宽度与活动量（家数占比、等权收益、离散、下行 RMS、额份额、量比、回补、振幅）"
                      "对未来 1/5 日等权市场收益存在可检验的（方向逐项预登记的）关系。",
        "mechanism": "上午截面是 D 日先验信息的首个完整消化窗口；其结构量刻画日内情绪与承接。"
                     "上游为日终文件，生产可用时点仍为收盘后（D+1 可用性约定不变）。",
        "fields": ["stock_daily.open", "stock_daily.high", "stock_daily.low",
                   "stock_daily.close", "stock_daily.amount",
                   "stock_adj_factor.adj_factor", "stock_history_5min.*"],
        "baseline": ["mkt_up_ratio", "mkt_dispersion", "mkt_am_ew_return"],
    },
    {
        "key": "index",
        "names": ["mkt_idx_csi300_ret1", "mkt_idx_csi300_momentum20",
                  "mkt_idx_csi300_volatility20", "mkt_idx_csi300_drawdown60",
                  "mkt_idx_csi300_ma20_gap", "mkt_idx_csi300_amount_ratio20",
                  "mkt_idx_small_large_momentum20", "mkt_idx_sz_sh_momentum20",
                  "mkt_idx_csi500_csi300_vol_ratio20", "mkt_idx_sh_sz_return_corr60",
                  "mkt_idx_growth_value_spread20", "mkt_idx_csi300_amplitude20",
                  "mkt_idx_csi300_amount_share", "mkt_idx_broad_dispersion",
                  "mkt_idx_csi300_range_position250"],
        "hypothesis": "指数层行情与结构量（收益、动量、波动、回撤、偏离、量比、风格/规模差、联动、振幅、份额、位置）"
                      "对未来 1/5 日等权市场收益存在可检验的（方向逐项预登记的）关系。",
        "mechanism": "指数按官方编制范围（不受固定池限制），价格指数不含分红；"
                     "指数动量/风格差刻画风险偏好与风格延续，量比/联动刻画参与度与系统耦合。",
        "fields": ["index_daily.open", "index_daily.high", "index_daily.low",
                   "index_daily.close", "index_daily.amount",
                   "stock_daily.amount", "stock_adj_factor.adj_factor"],
        "baseline": ["mkt_idx_csi300_momentum20", "mkt_idx_csi300_volatility20"],
    },
    {
        "key": "limit",
        "names": ["mkt_limit_up_ratio", "mkt_limit_down_ratio", "mkt_limit_net_ratio",
                  "mkt_limit_streak_max", "mkt_limit_streak_share",
                  "mkt_limit_broken_ratio", "mkt_limit_seal_amount_share",
                  "mkt_full_flat_ratio", "mkt_full_up_change5"],
        "hypothesis": "涨跌停生态（涨跌停家数占比、净额、连板高度与占比、炸板率、涨停吸金占比；全市场口径平盘/宽度变化）"
                      "对未来 1/5 日等权市场收益存在可检验的（方向逐项预登记的）关系。",
        "mechanism": "涨跌停是 A 股情绪与赚钱效应的直接度量：扩散度（净额）偏延续，极端度（连板高度/炸板/吸金集中）偏退潮风险；"
                     "方向与先验强度逐项预登记。limit 事件表 2020 年起可用。",
        "fields": ["stock_limit_list.limit", "stock_limit_list.limit_times",
                   "stock_daily.amount", "stock_daily.close", "stock_adj_factor.adj_factor",
                   "stock_market_distribution_history.*"],
        "baseline": ["mkt_up_ratio", "mkt_strong_up_ratio"],
    },
    {
        "key": "flow",
        "names": ["mkt_mf_large_net_ratio", "mkt_mf_large_net_ratio_ma5",
                  "mkt_mf_large_breadth", "mkt_mf_elg_turnover_share"],
        "hypothesis": "主力资金结构（大单净流入占比、其 5 日均、大单净流入家数占比、超大单成交占比）"
                      "对未来 1/5 日等权市场收益存在可检验的（方向逐项预登记的）关系。",
        "mechanism": "大单资金净流入代理知情交易与资金推动；金额单位万元，仅同表内比率，不跨表相除。",
        "fields": ["stock_main_fund_flow.lg_buy", "stock_main_fund_flow.lg_sell",
                   "stock_main_fund_flow.elg_buy", "stock_main_fund_flow.elg_sell",
                   "stock_daily.amount", "stock_adj_factor.adj_factor"],
        "baseline": ["mkt_up_ratio"],
    },
    {
        "key": "margin",
        "names": ["mkt_margin_buy_ratio", "mkt_margin_net_buy_ratio",
                  "mkt_margin_balance_chg5", "mkt_margin_balance_chg20",
                  "mkt_margin_short_chg20", "mkt_margin_breadth"],
        "hypothesis": "两融结构（融资买入占比、净买入占比、融资余额 5/20 日变化、融券余额变化、净买入家数占比）"
                      "对未来 1/5 日等权市场收益存在可检验的（方向逐项预登记的）关系。",
        "mechanism": "融资盘是杠杆增量资金与散户情绪合成器：余额与净买入趋势偏资金推动，融券余额变化偏空头压力；"
                     "源表实测滞后 1 个交易日，网格整体下移一行后使用（lag_grid 已声明）。",
        "fields": ["stock_margin_detail.rzye", "stock_margin_detail.rzmre",
                   "stock_margin_detail.rzche", "stock_margin_detail.rqye",
                   "stock_daily.amount", "stock_adj_factor.adj_factor"],
        "baseline": ["mkt_up_ratio"],
    },
    {
        "key": "sector",
        "names": ["mkt_sector_dispersion", "mkt_sector_mom20_dispersion",
                  "mkt_sector_leader_spread", "mkt_sector_rotation_speed",
                  "mkt_sector_amount_hhi", "mkt_sector_amount_ratio20"],
        "hypothesis": "行业板块结构（日收益离散、20 日动量离散、领涨强度、轮动速度、成交集中度、成交额量比）"
                      "对未来 1/5 日等权市场收益存在可检验的（方向逐项预登记的）关系。",
        "mechanism": "板块指数层口径：行业候选经 dc_blocks「行业板块」名称交叉核验；"
                     "分化/主线强度偏延续，轮动过快与集中度过高偏结构脆弱。",
        "fields": ["tdx_daily.close", "tdx_daily.amount", "tdx_blocks.*", "dc_blocks.*"],
        "baseline": ["mkt_dispersion", "mkt_amount_hhi"],
    },
    {
        "key": "liquidity",
        "names": ["mkt_no_trade_ratio"],
        "hypothesis": "池内无真实成交股票占比对未来 1/5 日等权市场收益存在可检验的关系（方向预登记）。",
        "mechanism": "无成交占比度量流动性冻结的程度（停牌/极端无量），是市场可交易性的直接状态量。",
        "fields": ["stock_daily.vol", "stock_daily.close", "stock_adj_factor.adj_factor"],
        "baseline": ["mkt_illiquidity_median"],
    },
]

IDEA_TAIL = ("本文件为 2026-10-03 市场因子复审的**追溯登记**（既有正式因子的逐值审计别名），"
             "非事前研究登记；方向与先验强度的预登记见 freeze 方向表，检验为回顾性筛选，"
             "不构成未触碰留出样本上的结论（retrospective_screening;historical_selection_is_not_untouched_holdout）。")


def candidate_payload(name, family_id, spec, definition_sha256):
    spec_meta = {
        "group": spec.group,
        "desc": spec.desc,
        "formula": spec.formula,
        "deps": list(spec.deps),
        "start": spec.start,
        "warmup_days": int(spec.warmup_days),
        "higher_is_better": bool(spec.higher_is_better),
        "version": 1,
        "note": f"市场审计别名（mka26 复审）：逐值复算正式因子 {name}@v{spec.version}；" + spec.note,
        "is_market": True,
    }
    if spec.lagged_ok:
        spec_meta["lagged_ok"] = list(spec.lagged_ok)
    if spec.align_fill is not None and not (isinstance(spec.align_fill, float) and spec.align_fill != spec.align_fill):
        spec_meta["align_fill"] = float(spec.align_fill)
    return {
        "name": f"mka26_{name}_v1",
        "family_id": family_id,
        "implementation": "research.families.market_gate_audit_v1:build",
        "parameters": {"original_name": name, "definition_sha256": definition_sha256},
        "units": f"与正式因子 {name} 完全一致（本候选仅做逐值复算，不改变任何口径）；单位与可用性详见其正式 note。",
        "availability": f"与正式因子 {name} 一致：T 日值于 T 日收盘后可用；"
                        + ("源表滞后 1 个交易日、网格已下移（lag_grid）。" if spec.lagged_ok else
                           "无额外滞后；上午族截面时点 11:30 但生产可用时点为收盘后。"),
        "missing_policy": "与正式因子一致：缺失保持 NaN，不用未来值回填；对齐前缀按正式声明（align_fill 或 NaN）。",
        "spec": spec_meta,
    }


def main():
    import factors  # noqa: F401  （装载正式注册表）
    from fea.config import load
    from fea.release import production_specs

    cfg = load()
    production = json.loads((ROOT / "conf" / "production.json").read_text(encoding="utf-8"))
    hashes = {o["name"]: o["definition_sha256"] for o in production["objects"]}
    specs = {s.name: s for s in production_specs(cfg) if s.is_market}
    assert len(specs) == 61, f"预期 61 个市场因子，实得 {len(specs)}"

    all_names = [n for f in FAMILIES for n in f["names"]]
    assert sorted(all_names) == sorted(specs), "族划分必须恰好覆盖全部市场因子"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ideas").mkdir(exist_ok=True)
    (OUT / "families").mkdir(exist_ok=True)
    (OUT / "candidates").mkdir(exist_ok=True)

    idea_files, family_files, candidate_files = [], [], []
    for fam in FAMILIES:
        idea_id = f"mkta26_{fam['key']}_idea_v1"
        family_id = f"mkta26_{fam['key']}_family_v1"
        idea = {
            "idea_id": idea_id,
            "hypothesis": fam["hypothesis"],
            "mechanism": fam["mechanism"],
            "expected_direction": "逐因子预登记（freeze 方向表：+1/-1/0），不依据事后结果翻转",
            "target_horizons": [1, 5],
            "falsifiable": "若主期限 HAC 95% 区间含 0、分年同号不足、或对既有市场因子增量不显著（或符号相反），"
                           "则该因子的信号主张被证伪；绿卡描述量不声称预测性。",
            "fields": fam["fields"],
            "registration_note": IDEA_TAIL,
        }
        variants = [{"original_name": n, "definition_sha256": hashes[n]} for n in fam["names"]]
        family = {
            "family_id": family_id,
            "idea_id": idea_id,
            "trial_budget": len(fam["names"]),
            "allowed_variants": variants,
            "baseline": fam["baseline"],
            "hypothesis_boundary": fam["mechanism"],
            "search_policy": "只登记既有正式因子的逐值审计别名，不扩窗口、不改参数、不新增变体",
            "registration_note": IDEA_TAIL,
        }
        ip = OUT / "ideas" / f"{idea_id}.yaml"
        fp = OUT / "families" / f"{family_id}.yaml"
        ip.write_text(json.dumps(idea, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fp.write_text(json.dumps(family, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        idea_files.append(f"ideas/{idea_id}.yaml")
        family_files.append(f"families/{family_id}.yaml")
        for n in fam["names"]:
            payload = candidate_payload(n, family_id, specs[n], hashes[n])
            cp = OUT / "candidates" / f"{payload['name']}.yaml"
            cp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            candidate_files.append(f"candidates/{payload['name']}.yaml")

    manifest = {
        "batch_id": "market_review_20261003",
        "definition_files": {"ideas": idea_files, "families": family_files,
                             "candidates": candidate_files},
        "idea_count": len(idea_files),
        "family_count": len(family_files),
        "candidate_count": len(candidate_files),
        "protocol": {"stocks": 2115, "output_start": "2018-01-01",
                     "primary_horizons": [1, 5], "auxiliary_horizons": [3, 10, 20]},
        "note": "市场因子复审追溯登记（mka26）：61 个正式市场因子的逐值审计别名；"
                "登记≠批准，逐值复算证明由 research batch 的 truncation/incremental 校验给出。",
    }
    (OUT / "manifest.yaml").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"生成完成：{len(idea_files)} ideas / {len(family_files)} families / "
          f"{len(candidate_files)} candidates -> {OUT}")


if __name__ == "__main__":
    main()
