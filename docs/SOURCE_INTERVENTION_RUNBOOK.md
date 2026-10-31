# ExDTI 来源同步干预：运行手册

2026-09-22。新入口位于 `exdti_evidence`。旧训练脚本、旧模型、旧数据保持原样，继续用于历史原型对照。新论文编译入口为 `O:\Ex\ExDTI_Doc\paper\main_purple.tex`。

2026-10-28 路径更新：资料目录从 `O:\Ex\ExDTI` 更名为 `O:\Ex\ExDTI_Doc`，实验代码仍在 `E:\VSCode\ExDTI`。本手册的历史备份记录保留当时路径，查找现存资料时请使用新资料根目录；详见 [路径同步记录](O:/Ex/ExDTI_Doc/docs/maintenance/PATH_SYNC_20261028.md)。

## 已完成的验证

- 30 项测试通过，覆盖数据身份归并、缺失标签保护、来源同步撤除、OR/AND 支持、条件匹配、派生缓存、四状态、引用、特征泄漏、梯度及训练接口。
- 在 RTX 4060 Laptop、PyTorch 2.8.0+cu126、Transformers 4.40.0 上，实际加载缓存中的 MolT5-base，完成动作头训练一步、证据模型训练一步、保存/加载及两个真实保留测试配对的生成。证据生成器有 9,457,924 个可训练参数；本次显存配置冻结 T5 主干，通过图编码、状态和引用规划的连续表示适配解码器。
- 实际报告位于 `output/source_intervention_molt5_20260922_final/{action,evidence,eval}/`。两个样本和一步训练只用于工程验证；其中的 F1、损失及文本不能用作论文性能证据。
- `output/source_intervention_molt5_20260922/` 和 `output/source_intervention_tiny_20260922/` 是数据最终修正前的验证记录，保留溯源，不与 final 目录混报。

## 1. 数据与任务

冻结 DrugBank 三分类数据为 5,583 个配对，训练/验证/测试为 4,471/556/556。正式配置读取 `data/source_evidence/drugbank_channels/`：每条来源带有文本、关系图和缓存三个通道，缓存在撤除后重写。任务是在已知 DTI 中预测 inhibitor、antagonist、agonist，不是 DTI 存在性二分类。缺失动作、其他动作、多个核心动作不会被强行归入这三个标签。该划分上的 5,623 次单来源撤除里，冲突为 0，同时包含图示三类局部修订的干预为 0；主要监督是唯一依据丢失后的状态改变，以及未受支持查询的保持。

完整审计、冷实体划分、新 ChEMBL 数据及 GtoPdb 接口说明见 [DATASET_EVIDENCE_20260922.md](DATASET_EVIDENCE_20260922.md)。ChEMBL 试点仍是 51 个配对。后续按每类 300 条机制记录得到 754 个配对，划分位于 `data/source_evidence/chembl_actions_300_split/`（606/74/74）。其中 684 个靶标序列已出现在本地 DrugBank，5 个配对精确重合，因此不能当作未见靶标成绩。该划分尚未训练。

## 2. 最小验证

在 `E:\VSCode\ExDTI` 运行：

```powershell
.\.exddi-conda\python.exe -m unittest discover -s tests -v
.\.exddi-conda\python.exe -m exdti_evidence.demo
```

`demo` 对应用户图中的 s1/s2/s3：撤除 s1 后 c1 变为依据不足，c2 保留并改引 s2，c3 保留。文件明确标记为合成示例，不参与科学性能统计。

`configs/source_intervention/*_smoke.json` 是短步配置；`tiny_*` 使用随机初始化小型 T5，仅供调试。完整 MolT5 配置以 `molt5_*` 开头。已有检查点的输出目录会拒绝覆盖；重跑前在复制的配置中指定新的 `output_dir`，并同步下游检查点路径。旧文件不会自动删除。

## 3. 正式实验入口

```powershell
.\.exddi-conda\python.exe -m exdti_evidence.run train-action configs/source_intervention/action_train.json
.\.exddi-conda\python.exe -m exdti_evidence.run train-evidence configs/source_intervention/evidence_train.json
.\.exddi-conda\python.exe -m exdti_evidence.run evaluate-action configs/source_intervention/evaluate.json
.\.exddi-conda\python.exe -m exdti_evidence.run evaluate configs/source_intervention/evaluate.json
.\.exddi-conda\python.exe -m exdti_evidence.score output/source_intervention_full/eval/predictions.jsonl --output output/source_intervention_full/eval/metrics.json
```

这些正式训练配置已经准备，**尚未跑完正式训练、多个种子或消融**。`max_steps=2000` 是初始实验预算，不是已选定的最佳训练长度。应在验证集选择训练预算、损失权重和置信度阈值，再固定测试设置。

动作预测器读取 SMILES 和去头蛋白序列，不读取动作标签、名称、来源或机制文本。两种输入分别分配词元预算，避免长 SMILES 挤掉全部蛋白输入；默认药物 128、蛋白 240 个词元。该截断仍可能损失远端序列信息，正式实验应记录长度敏感性。训练说明模型时预测器冻结，并在每一对证据视图中只计算一次 p。

正式动作训练默认更新整个 T5 编码器；短步动作验证只训练动作头。正式证据配置默认 `freeze_backbone=true`，沿用已在 8 GB 显卡跑通的主干冻结方案；设置为 false 可训练整个编码器和解码器，但完整模型的该显存模式尚未在本轮实测。`local_files_only=true` 使用已缓存 MolT5；其他机器需要先取得同一模型或显式改为 false。

评估会核对分词长度、模型路径、图开关、干预方式和状态规划开关与训练一致，并校验动作预测检查点。新检查点记录训练配对摘要，评估拒绝精确结构/序列配对重叠。不同化学等价表示仍依赖数据层的身份清理，不能因此声称已消除全部污染。

## 4. 已准备的消融

每项均有 `NAME_train.json` 与 `NAME_eval.json`，共享正式动作预测器。

| NAME | 改变内容 |
|---|---|
| no_change | 关闭局部状态变化损失 |
| no_keep | 关闭不变状态保持损失 |
| no_graph | 移除图节点通道，保留来源文本 |
| no_plan | 解码器不接收断言规划记忆 |
| single_view | 仅训练原始证据视图，关闭变化/保持项 |
| text_only_removal | 只遮蔽被撤除来源文本，故意保留旧图通道 |
| graph_only_removal | 只更新图，故意保留旧来源文本 |

例如：

```powershell
.\.exddi-conda\python.exe -m exdti_evidence.run train-evidence configs/source_intervention/no_change_train.json
.\.exddi-conda\python.exe -m exdti_evidence.run evaluate configs/source_intervention/no_change_eval.json
```

完全结构化、准确抽取的断言已有精确符号解法，`provenance.py` 的规则加模板是必须保留的强对照。它同时定义当前数据库状态监督，其满分属于构造检查，不能用作独立效果证据。研究价值需要在真实文本、不完备抽取、条件区分和局部生成修订中进一步验证。

## 5. 模型与论文的对应

| 图中模块 | 实现 |
|---|---|
| 来源与依赖 | `schema.py`、`provenance.py` |
| 冻结 p=f(D,T) | `model.py:ActionPredictor`、`run.py` |
| 文本/图/缓存同步重建 | `build_evidence`，不信任旧缓存 payload |
| 共享图文编码 | T5 编码器和两层来源—断言二部图消息传递 |
| 四状态规划与引用 | `EvidenceGenerator.plan` |
| 状态条件化解码 | 预测软状态、软引用汇聚、固定 p 的连续记忆 |
| 五项配对损失 | `paired_loss` |
| 来源撤除后的评价 | `metrics.py`、`score.py` |

神经特征只读取明确白名单。支持集合的 OR/AND 逻辑用于重建与监督；完整支持 DNF、目标状态和真实动作标签不拼进模型输入。图输入仅含可观测断言及展开后的来源关联。引用监督学习有效支持/反驳来源的并集，规则文本选最短完整见证集合，两者不是同一标签目标。

## 6. 解释结果怎样读

`predictions.jsonl` 分别保存固定预测概率、前后视图的规划状态和引用分数、原始生成文本、仅供评分的符号参考及规则基线。原始解码文本没有自动药理真实性保证，也不因规划器正确而被视为正确。

指标区分：

- `coverage`：所有动作查询中，模型作出支持/反驳/冲突陈述的比例。
- `unsupported_claim_rate`：上述陈述中，包含参考证据未支持极性的比例。
- `effective_supported_coverage`：真实明确支持的查询中，被正确报告为明确支持的比例，对应论文有效覆盖率。
- `supported_state_error_rate`：模型明确报告支持的查询中，参考并非“支持且无反证”的比例，对应论文主要 UCR。
- `state_macro_f1_all_four`：固定四状态平均；另保留仅对出现状态平均的指标并明确命名。
- `supported_retention`：撤除前后都真实支持的查询中，两次均被正确保留的比例。无适用样本时返回 null，不伪装为 100%。

`score` 提供阈值—覆盖率曲线；正式对比需在验证集确定阈值并对齐有效覆盖率。测试曲线用于描述，不能用于回选最佳阈值。当前真实库注释主要包含支持/依据不足；冲突、反驳和共同支持的生物学标注集仍待建立。

## 7. 备份与恢复

备份位于 `O:\Ex\ExDTI\archive\backups\20260922_2136_source_intervention`：论文及资产 ZIP、实验源文件 ZIP、逐文件 SHA256 清单、原 Git HEAD/status 和用户原有未提交修改 patch。大型原始数据库、环境、缓存与旧模型没有复制，因为本轮没有改动它们；排除项在 manifest 中明确记录。

需要恢复时，先解压到另一个目录核对，再按需取回文件。不要用重置 Git 工作区代替恢复：本轮开始前已有用户未提交修改。

论文修改实际以紫色排版；代码保持有效 Python 语法，紫色改动清单和新增源码预览位于 `docs/SOURCE_INTERVENTION_CHANGES.html`。

## 2026-10-31 专利对齐

- 评价配置新增 `release_verification`（权利要求9发布前核验），开启后预测中另存 `released_claims`，原始神经输出保留。详见 `docs/PATENT_ALIGNMENT_20261031.md`。
- `build_source_evidence.py` 现按行内物种/亚型/实验条件字段生成条件；数据导出含这些字段时需重建 `data/source_evidence`（写入新目录，勿覆盖）。

