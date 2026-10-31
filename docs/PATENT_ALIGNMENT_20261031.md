# 专利对齐记录（2026-10-31）

依据：`发明专利申请稿-来源同步干预_提交老师版_20261031.docx`（以专利为准）。分支 `patent-align-20261031`；注意 `exdti_evidence/`、`tests/`、`configs/source_intervention/`、`prepare_data/build_source_evidence.py` 原为 git 未跟踪文件，改动前完整备份于 `_backup_patent_align_20261031/`。

| 专利要素 | 代码位置 | 本次改动 |
|---|---|---|
| 权利要求9、[0023]、[0102]–[0103] 发布前来源见证核验 | `exdti_evidence/provenance.py` `release_check`；`run.py` evaluate 中 `release_verification` | 新增。核验器仅依据当前活动来源与明示规则（`assess_claims`）重定状态：仅正→支持、仅反→反驳、两者→冲突、皆无→依据不足；先查引用ID属于活动来源，再分别查正、反完整见证；未通过时不放行神经输出，改用重定状态+有效见证+规则模板，冲突列示正反来源；不把反驳/冲突改写为依据不足；原始神经输出 `claims` 原样保留供审计，放行结果写入 `released_claims`。配置开关 `release_verification`：代码默认 false（旧行为可复现），`configs/source_intervention/*eval*.json` 与 `evaluate.json` 设为 true。 |
| [0087]/[0048] CACHE 节点 | `exdti_evidence/features.py` 缓存节点段 | 仅改注释：缓存文本以 CACHE 前缀形成附加图节点，与关系节点并列而非关系节点，不新增独立证据。行为原已一致。 |
| 权利要求10、[0118] 撤除来源选择 | `exdti_evidence/run.py` `choose_removal` | 补 docstring；行为原已一致（明示干预列表优先取 step mod 长度，否则 step mod 来源数选一个来源，再按 source_unit_id 展开副本）。 |
| 权利要求3、[0071] 条件键值 | `prepare_data/build_source_evidence.py` `normalized_condition` | 由行内 organism/species、target_subtype/isoform、assay_context 等字段生成规范化条件，缺失保持 unspecified。当前 `data/action_v1/all_eligible.jsonl` 无这些字段，重建结果与现有数据一致，故未重建数据；若导出加入物种等字段，需重建到新目录。 |
| [0085]、[0095]、[0098]、[0121] 超参数 | `configs/source_intervention/action_train.json`、`evidence_train.json`、`evidence_eval_*.json` | 核对一致，无需改动：2000步、累积8/4、Adafactor 1e-4、种子421、长度384/128/240/160/128、max_sources 32、阈值0.5、五项损失权重均为1。 |
| ProtBERT（[0131]，仅现有技术） | — | 不加入模型；代码中无 ProtBERT。 |

测试：新增 `tests/test_patent_alignment.py`（11项），全部 43 项通过。

重要：`output/source_intervention_full/` 下全部结果产生于本次改动之前，不含发布前核验。训练代码路径（撤除规则、损失、模型）未改，因此以原检查点仅重跑评价即可得到核验效果，输出在 `output/source_intervention_patent_20261031/`。
