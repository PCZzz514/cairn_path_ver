# 实施审查第 1 轮

仅 Claude 修改实现。先修复以下问题，再启动完整实验。

1. crt.py: train_head 中 log(phase='head_epoch', arm=arm, lr=lr, seed=seed, **record) 重复传入 lr（record 也含 lr），会在第一轮直接 TypeError。应区分 base_lr 与 epoch_lr，避免重复键。同步核对其他日志调用。
2. cRT 的 cosine 使用现代 PyTorch 标准顺序：每轮 optimizer 完毕后 scheduler.step()，使第1轮使用配置的完整 base_lr，10轮均有实际更新。移除静默压制 scheduler 顺序警告。原仓库旧调用顺序受 PyTorch 版本影响；在 README 记清此适配。自然与cRT保持完全相同调度。
3. parent_test_check 除报告 argmax_equal 外应要求全部原始 test top1 一致；行序/labels/allclose 保持。不可把小概率误差下的分类差异静默当作通过。
4. K-Paths 实测显示为每pair构建 search view 是瓶颈：pair27885 view1.255秒/full图官方0.0087秒；23/24 pair节点路径一致，但pair179372关系序列因equal-length遍历发生变化。优先改成一次构建完整 nx.DiGraph（全局ID有序、复合关系signature保持），fork workers共享，仅每pair调用官方同样的 shortest_simple_paths 加候选预算。完整图构建约11.4秒，服务器439GiB可用，资源允许。components/距离可继续用于fallback/真实shortest metadata。
5. 将同样2000个真实pairs性能抽样在完整图实现上复测，核对原子展开、方向、重复关系、fallback和长路径，并要求24connected pair与官方完整图一致。1/2000候选截断不是普遍不足K，仍明确保存此预算适配与截断标志。若选择完整图，应删去primary与search-view tie差异相关的成功宣称；保留旧实现证据供比较即可。
6. cRT/Step2 的新结果应导出85行class_counts表（train/val/test支持、预测次数、P/R/F1），保留fixed85主分数及训练和测试都正支持集合的额外诊断。无需新增防御框架。
7. 最终检查后台任务依赖：GPU4两个Step1并发，GPU5/6各Step2，GPU7串行Step3，只消费自身Step2 best和对应Step1固定lr；Codex low每2400秒，不改配置。
8. monitor_loop.sh 使用已授权且此前成功的 --dangerously-bypass-approvals-and-sandbox（替换 -s workspace-write 与 sandbox_workspace_write.network_access）。不增加审批层；prompt约束只能读状态、请求本次任务恢复，训练由独立host任务启动。首次监控应核对 codex_exit=0、after.json 与 next_check 均真实存在。
9. Step2 在原 export_test 后，也应将 Top5_accuracy 和 diagnostic_train_and_test_positive_classes 补入 test_metrics.json；可直接使用已导出的概率和 crt.evaluation 的同一计算口径，绝不用于选择checkpoint。原 loss_terms/selected_epoch 等字段保留。现在分类层有这两项，而 Step2 仅增加了 class_counts。
10. 文档需区分官方预处理和 DiGraph：官方 augmented_network_utils.merge_duplicate_relations 先把 parallel relations 合成复合关系，再送 DiGraph；不要把我们的tuple实现描述成纠正官方完整pipeline“丢弃多关系”。我们是保留复合语义并按现有DRKG原子relationIDs展开的适配。

以上是具体实现问题和性能数据，不授权改变模型、划分或损失。仍暂不启动完整训练；修复并给出真实验证证据后交桌面审核。
