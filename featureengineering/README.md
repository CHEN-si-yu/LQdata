# 日频因子工程

读取只读上游 `../datadownload/data`，生成股票因子、市场因子和未来收益标签。直接执行即开始增量生产：

```bash
python /root/autodl-fs/featureengineering/main.py
```

当前入口围绕五件事：全量重算、增量补齐、智能跳过、并行计算和生产进度。固定股票轴、公式、PIT、预热及年度 Parquet 落盘合同沿用原实现。生产不执行指纹、审核、评估、监控、字典生成或发布；结果直接写入 `data/`，断点保存在 `state/`。

## 常用操作

在项目根目录执行：

```bash
python main.py                              # 日更
python main.py --jobs 1                     # 首次补历史（含 GPU 候选）
python main.py --jobs 12                     # 增量，指定并行度
python main.py rebuild --jobs 1             # 强制重算全部启用因子
python main.py rebuild --resume --jobs 1    # 全量中断后补剩余部分
python main.py run momentum_20 roe_ttm       # 指定因子，自动包含父依赖
python main.py run --group market            # 只选市场因子
python main.py rebuild --from-year 2025 --to-year 2026 --jobs 1
python main.py run momentum_20 --start 2026-09-01 --end 2026-09-30 --refresh
python main.py run --dry-run                 # 查看待算计划，不生成因子
python main.py list                          # 注册因子、类别、台账覆盖
python main.py status                        # 与 list 相同
```

`run/incremental` 表示增量，`rebuild/full` 表示强制重算请求范围。默认截止为上游 `stock_daily` 最新交易日；显式截止不会超过源截止。限定日期的重算保留范围外结果。

已有结果按覆盖和物理分区跳过；新注册因子补历史；标签随新交易日回算未成熟尾部，覆盖跨年。每个成功年分区立即保存断点。中断后再次执行 `python main.py` 即可补齐；再次执行不带 `--resume` 的 `rebuild` 会重新重算。公式修改须显式重建；上游历史修订须显式指定日期并使用 `--refresh`。

有 CUDA 时，含 R16ML1 候选的首次补历史、跨年指定运行及全量重算须使用 `--jobs 1`：候选使用全局 nonblocking GPU 锁，多个年度 worker 同时执行会因设备占用拒绝运行。无 CUDA 自动退回 CPU；自 2026-10-13 起，CPU 按派生缓存目录、任务年份和审计截止日加锁，不争抢 GPU 锁，不同年度可按自动资源预算调度。同一年度缓存的重复写入仍拒绝。数值设置（种子 / 线程 / 确定性 / float64）属于冻结合同；**硬件与解释器版本漂移自 2026-10-09 起只告警不阻断**，跨环境末位差异属允许范围，详见 [R16 交付记录](doc/04_研究记录_R16交付_20261008.md) 末节。

## 进度与输出

终端默认显示总进度和名称对齐的逐因子 `tqdm` 进度条。文本模式采用时间戳、①—④阶段摘要、`[序号/总数]`、已有覆盖、写入/新增行数与耗时：

```bash
python main.py --progress plain              # 对齐文本
python main.py --progress bars               # 强制多行 tqdm
python main.py --progress off                # 只显示阶段摘要
python main.py > logs/console.log 2>&1        # logs 目录须已存在；自动使用文本
```

自动模式检测进度输出所在的标准错误是否为终端。不传 `--jobs` 或指定 `--jobs 0` 时，按配置默认 12 个计算进程选择，并受配置上限 20、CPU 配额、项目内存预算及单进程地址空间限制。显式指定正整数（如 `--jobs 40`）时直接使用该值，豁免这些项目资源限制，包括每任务内存预留造成的调度限流；进度条仍显示实际内存。操作系统和容器自身的硬限制仍然有效。持久进程按因子×年份领取任务，复用输入与按需派生缓存。生产写入受 `state/daily.lock` 独占锁保护。

| 对象 | 落盘位置 | 列顺序与类型 |
|---|---|---|
| 股票因子、标签 | `data/factors/<name>/year=YYYY/data.parquet` | `trade_date:string, stock_code:string, value:float32, rank:float32` |
| 市场因子 | `data/market_factors/<name>/year=YYYY/data.parquet` | `trade_date:string, value:float32` |

日志位于 `logs/run_factors_*.log`；断点位于 `state/<name>.json`。`list/status` 展示注册表和台账，实际计算时再检查分区；旧发布指针与旧字典不能代表当前运行结果。

## 文档与验证

- [文档导航](doc/README.md)：当前说明与历史材料的分类入口。
- [项目手册](doc/01_生产_项目手册.md)：计算口径、运行、恢复、资源和维护。
- [入口命令说明](doc/01_生产_命令接口.md)：完整参数与恢复语义。
- [因子注册与落盘](doc/01_生产_因子注册与存储.md)：注册来源、增加因子与输出合同。
- [生产重构验证记录](doc/01_生产_重构验证_20261007.md)：20 个场景测试及 5 个真实因子的原引擎对拍证据。

隔离测试命令与结果边界见验证记录。历史研究、裁决和冻结政策按[历史分类索引](doc/README.md)查阅，其旧命令和旧流程不用于当前生产。

2026-10-08 R16 已完成收尾，新增两项生产交付，当时为 260 股票/47 市场/5 标签（合计 312）。2026-10-18 用户重新授权 R17 继续挖掘，目标股票因子 >500；2026-10-19 用户要求停止新方向并收尾。R17 最终为 **310 股票/47 市场/5 标签（合计 362）**，新增 50 个正式股票因子，未达到 500 目标。最终交付、证据保证与未准入事项见 [R17 交付记录](doc/04_研究记录_R17交付_20261019.md)，过程与缺陷记录见 [R17 轮次记录](doc/04_研究记录_R17_20261018.md)。

**2026-10-20 模型侧复核**：R17 的 50 个因子整体未通过下游模型的增量价值复核（整批/子集不涨 IC、复合信号对现有 260 列的残差 IC≈0），建议触发因子侧重新审批/复审；30 个财务构成因子为优先回档候选，回档需另行批准。详见 [R17模型侧复核](doc/04_研究记录_R17模型侧复核_20261020.md)。
