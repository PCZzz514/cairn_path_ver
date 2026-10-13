# DDI cRT / K-Paths 实施计划
日期：2026-10-13。Claude 编写实现，桌面 Codex 规划与审核，服务器 Codex CLI 仅监控。
用户已授权非沙箱执行。所有新增文件、缓存、日志、训练输出只能在 /data/interns/zhangpc/cairn_new 内。cairn_dev 只读。不得修改原训练或已有模型文件。不得写 smoke test、hash/sha256、泛化防御框架、重复审计审批层。仅必要方法一致性和真实数据验证。

## 目的与分组
model = 当前加速 DSN-DDI + EmerGNN + RareDDIE，with_path=true。
原模型：accelerated_four/outputs/path_{single_drugout,double_drugout}/best.pt。
Step1：冻结原模型最后 Linear(256,85) 之前全部表示，重置最后一层，cRT 均衡采样；另配相同预算自然采样重训控制。
Step2：K-Paths 替换原路径检索，端到端从相同 seed42 初始化来源重新训练，暂不 cRT。
Step3：在 Step2 选定 best 上提取它自己的固定表示，再独立初始化最后层做 cRT 和自然采样控制。不得移植 Step1 头。
GPU4 同时跑 Step1 单冷药与双冷药（特征抽取及头训练都只见 GPU4）。
GPU5 Step2 单冷药，GPU6 Step2 双冷药。GPU7 待各 Step2 完成后串行 Step3，不占其他 GPU。
当前 primary 两种冷药完整训练，最大100epochs、patience15。150epochs 是之前 PairOut/CellOut，不能误用到本次。
不得改 loss、FIN、DSN、NAI、PVR、mu 预测、MLP 前缀，不加 gating/logit adjustment/contrastive 等。

## 已阅读的依据（实现者仍需先读）
cRT 原文 references/cRT.pdf / cRT.txt（1910.09217v2，ICLR2020，第4节）。
官方 source references/cRT-official：
data/ClassAwareSampler.py，models/DotProductClassifier.py，config/ImageNet_LT/cls_crt.yaml，run_networks.py。
官方版本 f162adef63fdf592d0b8a0f6e52bc481fc08785f。
K-Paths 原文 references/K-Paths.pdf / K-Paths.txt（2502.13344，第3节与Algorithm1）。
官方 source references/K-Paths：
k-paths/utils/Kpaths_utils.py，augmented_network_utils.py，subgraph_utils.py，k-paths/get-subgraph.py。
官方版本 bf25d24eac6939574542d8994957b340ea1f5ab8。
来源 README 链接：https://github.com/facebookresearch/classifier-balancing 与 https://github.com/rsinghlab/K-Paths。
官方 Kpaths_utils 注释声称不许重复关系，但实际 two_nodes_in_graph 只去重整条关系序列；以实际代码与Algorithm1为准。
官方 cRT epoch_steps=floor(Ntrain/batch)，不要用过采样长度决定epoch更新次数。

## 当前生产实现与数据
复用 accelerated_four/dsn/model.py 的 DSNRareDDI、data.py 的 DDIDataset/Collator/to_device、metrics.py。
原 final classifier: LayerNorm128 -> Linear128,256 -> GELU -> Dropout.2 -> Linear256,85。
原训练 accelerated_four/dsn/train.py，配置 configs/path_single_drugout.yaml 与 path_double_drugout.yaml。
冷药冻结数据 data/prepared/{protocol}/ddi.parquet 与 pair_indices.npy，assets.pt，pairs.npy。
训练149048各；单val20785/test16978；双val760/test534。严格有序drug_a/drug_b和原 frozen_row_index，每行一次预测。
FIN entities96766×128，relations105×128，可训练。NAI 六类cap30固定。
路径层4跳，每层相同ordered-pair gate共享双向、乘法消息sum、inverse105+self210，不更改。
原全模型损失 CE + .1 rec + beta KL，beta max.001 warmup20%（100*1165updates，23300warm），batch128，AdamW1e-4/weight_decay1e-4，CPUthreads2/workers2。
严格 fixed85 val MacroF1 improvement 保存 best；patience15；test 仅完成val选择后一次正式导出。
资产与冻结划分只读复用，无复制大表必要。路径单独缓存并替换 collator.paths，不得依赖原路径子图生成 K-Paths。
全背景 drkg = ../cairn_dev/data/processed/drkg_triples.pt，键 triples/relations/entities。triples顺序(h,r,t)。现已有4488504条非DDI三元组、105关系。无Compound-Compound，禁止加入训练DDI、任何val/test答案边。源不允许随机cap30截断。
参考 cairn_dsn/paths.py 与 prepare.py，旧 meetingcap32/armcap2，实际max节点130边695。

## cRT 实现规范
1. 父模型 strict load 原 best；eval + requires_grad False。使用最后Linear的 forward_pre_hook 抽取真实 g_ab(256) 和原 logits，不复制/重写整个 forward，不使用随机z作为分类表示。前缀dropout关闭。保存 train/val/test features、原logits、frozen rows、有序药物IDs、labels、父checkpoint/config/epoch和来源路径，禁哈希。
2. 抽取时验证 old_last_linear(g) 与同批原model logits allclose(atol1e-6,rtol1e-5)，exact argmax；核对行序与父test概率一致。保存实际误差。父全部params/buffers保持固定。真正 head 训练过程中父模型不参与优化，缓存固定。
3. sampler 忠实 RandomCycleIter：随机循环train实际存在classIDs，再循环该类样本，每次取同类连续4个；支持稀疏实际classID字典，不能对label重编号；仍输出85 logits，不mask训练缺失类。sampler nominal len=max_class_count*n_train_classes，实际每epoch只291步 floor149048/512，512每batch。
4. CE唯一loss；只训练新的 nn.Linear(256,85) 权重和bias。SGD momentum.9 wd.0005 cosine10epochs；batch512。
5. development head seed42 比较共同lr候选 [.01,.05,.2]。每个lr的自然/cRT头用完全相同初始权重，梯度预算相同。按两arm各自val最佳F1均值选一个共同lr（仅val，tie按候选顺序），不为两arm分别挑lr。之后固定该lr，head seeds42/43/44（42可复用选中结果），seed43/44也逐seed相同初始化。说明三次是head随机性重复，非端到端多seed。
6. 每个头val固定85F1严格择优、自然和cRT10epochs同预算，不做提前终止头；最终test选择完后导出。Step3固定用各protocol Step1所选lr，不重新用test或额外lr选Step3。
7. export metrics Acc/Top5/fixed85F1/MacroAUROC supported计数、85 perclass、confusion、full probabilities+rowIDs。补train&test均正样本F1诊断，明确类别IDs/count，主指标永远fixed85。test从不用来选lr/epoch/headseed。

## K-Paths 实现规范
官方默认 K10,L3；本次主实验K10,L4，为保持现有4跳明确标注参数适配。无需这轮额外L3训练。
- 完整无DDI背景，有向边+每条显式inverse，按全局nodeID稳定插入。节点图使用 DiGraph，不能用MultiDiGraph调用shortest_simple_paths。
- 同有向nodepair的所有原子关系排序后组成可逆完整 tuple/signature（重复fact保留multiplicity，但关系序列签名去重relationID）。选择路径后恢复原子关系，不增加composite learned embeddings。不得只取最后一个原子关系。
- shortest_simple_paths/Yen loopless，shortest first。完整有序relation signature sequence去重，保留第一次遇到的node路径；同path里允许重复r，r1/r2顺序重要。
- 若完整NetworkX full graph性能不够，可以使用所有<=4hop路线的完备、标签无关图视图（如两个端点full twohop balls的union含完整相关边）缩小 nx 搜索；须证明没有去掉<=4路线，且稳定equal-length tie顺序有依据。不能先抽meetingcap32/armcap2旧子图，不能随机局部截断。
- production候选nodepath计数上限1000，属明确计算预算适配；保存is_exact/candidate_count/budget_truncated/status，不能把预算截断叫无路径。记录实际运行秒数。若1000导致普遍不足K/截断，先报告桌面审核，不能静默改指标或伪装完整检索。
- official fallback：双端皆不在graph空；只有一端存在该端out-neighbor按单边复合关系去重最多K；双端存在但全graph不连通分别neighbors（可能2K）；双端连通但最短>L空，没有neighborfallback。
- 用全graph连通性判定disconnected/overlength；nonDDI+inverse graph可用connected_components支持，不可把截断混入。
- selected 全路径原子edge union、保留sourcefactmultiplicity、加inverse仅一次+self-loop一次，完全不补诱导边。连接路径path_mask根据真实selected连接；fallback另记录，不能使药物间mask true。
- 资源上限对齐原130nodes/695edges。K10L4通常<=32nodes；若原子展开超预算只按完整path整体纳入，不截断edge造成断路，记录预算原因；节点/边预算不引入类别偏好。
- 覆盖全部ordered pairs，不drop no-path样本。元数据每pair：drugIDs、FIN endpoints、old shortestdistance（若不可得计算真实距离而非oldcachelength推断）、selected path nodes及完整relation signatures、length counts、候选/选择数量、graph nodes/edges、exact/truncated、fallback、seconds。2hop共同Gene/SideEffect/Disease只作诊断，不加排序偏好。
- 核对少量真实DRKG子图与官方方法实际调用结果（可直接AST加载纯函数避datasets依赖）。必要案例覆盖multi-rel expansion、重复关系、方向、one-end-absent、disconnected、overlength。只做方法一致性验证，不搞toy forward smoke test或大测试框架。
- 稳定CPU并行缓存准备，实际样本速度先测，再全量；不要让full graph重复build每pair。使用已有环境，只新增依赖到cairn_new，如确需。
- 使用同原训练代码 DSNRareDDI 模型参数/初始化调用和seed42；记录 shared initial state_dict 或replayed original seed42初始化来源，K模型不得load原best；不能宣称旧运行有完整initial state证据。原Bbaseline作为既有固定参照，注明一份端到端seed。
- step2trainer复用模型/数据/指标，尽量与原train区别仅topology路径、namespace、GPU5/6、metadata，不改优化顺序/损失/随机数消耗/epoch计划。

## 新目录与后台交接
全部代码置 crt_kpaths/，配置、日志、输出、cache均在它或logs/crt_kpaths内。不修改原 accelerated_four、cairn_dsn、旧monitor。
建议 sampler.py, crt.py, kpaths.py, prepare.py, train.py, jobs.py, monitor_loop.sh, configs/, README.md。名称可合理调整，避免复制模型主体。
Claude先完成code/必要核对，写implementation_summary.md、准确改动清单。暂时不要launch完整训练，交桌面审核。可执行真实数据方法检查与抽取小段性能测量，但不得toy模型/smoke训练，不占GPU0–3。
Launcher host detached/session独立于Claude/Codex生命周期，任务可按自身last断点resume，严禁静默重头/改batch/停他人训练。
Step1单/双GPU4并发。K准备可CPU先完成，两Step2 GPU5/6独立并发。每Step2有正式test_complete后Step3在GPU7串行并使用自己的best。
后台monitor每2400秒调用现有codex exec -m gpt-6.1-sol -c model_reasoning_effort=low，认证CODEX_HOME=$PWD/.codex_runtime；CPU only CUDA_VISIBLE_DEVICES=''。prompt仅状态/本次任务恢复与依赖调度，禁止改模型/损失/划分/预算或自行启动其他实验。不得调用hash/sha256/smoke。
保持status.json、各phasePID、进度日志、每epoch记录、test结果。monitor直到全部已授权Step1/Step2/Step3和控制头完成。后台启动后需确认真实first训练更新，以及Codex首次监控成功与next_check每40min。不用巨量checklist或另造审批层。

