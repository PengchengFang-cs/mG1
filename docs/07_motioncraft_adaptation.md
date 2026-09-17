# MotionCraft → 物理闭环策略：适配方案（步骤 5，待审，未实现）

目标：用 MotionCraft 的架构与训练形式（root 先、body 后的双流 DiT；rectified flow；网络预测 x0、loss 在速度空间；logit-normal t；AdaLN-Zero 注入；Euler 采样 + CFG）训练一个文本条件的闭环物理策略，在 docs/06 的协议上评测。本文只定方案，不写代码；第 8 节列出需要你拍板的项。

## 1. 数据与 token（30 fps，一帧一个 token）

来源：`data/humanml3d_phys/hml_phys_train.pkl`（8,718 段，17.6 h）；val 530 段做损失监控；test 只用于最终评测。

每帧 token = **root 流 15 维 + body 流 420 维 = 435 维**，全部按 UniPhys `motion_repr_utils.get_repr` 的定义从 root_state / dof_state / body_pos 计算（代码现成、经过验证），并做窗口级规范化（`cano_seq_smpl_or_smplx`：以历史第一帧的根 xy 为原点、朝向转到 +Y）：

| 流 | 分量 | 维度 | 来源 |
|---|---|---|---|
| root | root_trans | 3 | root_state[0:3]，规范化坐标系 |
| root | root_rot_6d | 6 | 根四元数 → 旋转矩阵前两列 |
| root | root_trans_vel | 3 | root_state[7:10] |
| root | root_rot_vel | 3 | root_state[10:13] |
| body | local_positions | 72 | 24 刚体位置，减根 xy、按根朝向旋转 |
| body | local_vel | 72 | 相邻帧位置差在根系下 |
| body | dof_pose_6d | 138 | 23 关节轴角 → 6D |
| body | dof_vel | 69 | dof_state[...,1] |
| body | **action** | 69 | PULSE/PHC 策略动作（PD 目标 = offset + scale·action，offset/scale 来自环境） |

说明：
- 动作放在 body 流：动作与关节一一对应，root 流保持纯"计划"语义。
- 帧 t 的 token 里的 action 是 PHC 记录器与该帧配对的动作，即**产生第 t 行状态的那个动作**（状态在物理步之后记录）；闭环缓冲区同样配对，未来第一行的动作就是当前状态下要执行的动作（见 §13）。
- 归一化：训练集逐维均值/方差（root、body 状态、动作分别统计），MotionCraft 的 `variance_eps` 保留。
- 第二版换 32 维 PULSE 隐动作只需把 action 字段换成 pulse_z（body 流 383 维），闭环时加 PULSE 解码器一次前向。

文本：每段 3–4 句 HumanML3D 描述；带时间标签的句子只分配给与其时间段重叠的窗口（UniPhys 的重叠规则）；无描述的窗口用空文本（无条件）。编码器 CLIP ViT-B/32：token 特征（≤ 50 个）走 MotionCraft 的联合注意力，pooled 特征进 AdaLN。全部离线预计算并缓存（44,970 句）。

镜像：第一版不用。

## 1.5 文本条件的两项补充（用户 2026-09-15 确认：进度条件进第一版，BABEL 留第二版对照）

问题：HumanML3D 是整段一句话（只有约 15% 带子片段时间标签），而闭环策略按 32 帧滑窗训练，同一句话在不同时刻对应不同的动作阶段，模型不知道"进行到哪了"。UniPhys 用的 BABEL 是帧级标注，没有这个问题。

1. **进度条件**：每个滑窗的 (已过时长 / 片段总长, 片段总长) 两个标量做嵌入加进 AdaLN（与时间步嵌入相加）。推理时二者合法可得：评测协议给每条条目真值长度（CLoSD/MDM 同样以长度为条件），已过时长即执行步数。
2. **BABEL 辅助文本（第二版对照，不进第一版）**：理由：测试时必须置空，收益只能间接传递且不确定；缺标注的片段集中在少数子集，训练分布有偏。我们的物理片段就是 AMASS 序列，UniPhys 数据里的 `frame_labels`（proc_label + start_t/end_t）按 AMASS 文件名可直接对上，覆盖我们 66%（train 5,759/8,734，val 363/528，test 1,097/1,640）的片段，每个序列中位数 5 个动作段。训练时文本条件 = HumanML3D 整句 + 当前滑窗重叠的 BABEL 短语（第二路 token，独立 dropout 0.5，无标注的窗口置空）；测试时只给整句，第二路置空，协议不变。BABEL 的时间标签需按 UniPhys 的 t_scale 规则对到跟踪帧（KIT/EKUT/MPI_mosh 用 0.9）。

## 2. 任务结构：MotionCraft 的 control 任务即闭环形式

- 窗口 T = H + F 帧。历史 H 帧作为 **observed 帧**（observed_mask = 1，token 用真实值 imputation：z_imp = z·(1−m) + x_obs·m），未来 F 帧生成。这正是 MotionCraft 现成的 control 机制（`hy273_constraints` / `unified_kimodo_datasets` 的 observed_mask），只是把"稀疏关键帧"换成"稠密历史前缀"。
- 建议 **H = 16，F = 16**（MIND 历史 16；UniPhys T=32 里上下文 4）；闭环每次执行前 **K = 4** 个动作再重规划（MIND 动作 horizon 4）。这三个数是超参，见第 8 节。
- 训练样本：从训练段以 stride 1 滑窗（约 1.9 M 个窗口）；不足 T 的段丢弃或前端重复第一帧（历史全为静止）。
- **闭环鲁棒性数据增强**（必须做，之前 G1 线的教训）：历史 token 在训练时以概率 0.5 加小高斯噪声（UniPhys 的 stabilization 思路）；未来的动作维度不加。不做 DAgger（先看纯 BC 的数字，和 UniPhys/PDP/MIND 同类）。

## 3. root 先、body 后

- root 流：输入 [z_imp_root(15), mask, 文本]，输出未来 15 维 root token（相当于"计划"：未来 F 帧根轨迹、朝向、速度）。
- body 流：输入 [预测的 root token（detach，MotionCraft 的 `detach_root_bridge`）, z_imp_body(420), mask, 文本]，输出未来 body token（状态 + 动作）。
- 关键约束：body 流**训练时也吃预测的 root**（而不是真值 root），否则闭环时 root 的误差会暴露给 body。MotionCraft 里本来就是这样接的，保留；self-conditioning 关闭。
- 桥接维度：MotionCraft 的 LOCAL_ROOT_DIM = 4（局部根）在这里直接换成 15 维 root token；如果想保持 4 维可取 (Δx, Δy, Δyaw, z)，默认用 15 维。

## 4. 生成形式与损失（MotionCraft 原样）

- rectified flow：z_t = t·x0 + (1−t)·ε，t ~ logit-normal；网络输出 x̂0；v̂ = (x̂0 − z_t)/(1−t)，1−t 下限 0.05（`velocity_loss_t_eps`）；loss = ‖v̂ − v‖²，只在未来帧上算；root 流与 body 流损失相加，动作维度权重 w_a（默认 1，备选 2）。
- 文本 dropout 0.1（CFG）；EMA 按 MotionCraft 默认。
- 采样：Euler N 步（先用 MotionCraft 默认 32 步，闭环评测算力够；之后再减到 8–10 步或蒸馏），CFG 尺度先 2.5（UniPhys 值），扫 {1.5, 2.5, 3.5}。

## 5. 闭环执行（评测与部署同一套）

每 K 步：从 Isaac 读最近 H 帧的 root_state / dof_state / body_pos 和已执行动作 → 规范化坐标 + 计算 token + 归一化 → 采样 F 帧 → 反归一化 → 取前 K 帧的 69 维动作逐步执行（PD 目标 = offset + scale·a）。起始：随机站立姿态（与 UniPhys 一致），历史用第一帧重复填充、动作为零。摔倒判定与 docs/06 相同。这个循环复用 `hml_phys/uniphys_rollout.py` 的环境部分，只换 `policy()`。

## 6. 模型规模与训练预算（建议）

- 数据 17.6 h，比 MotionCraft 的语料小得多，模型缩小：hidden 512、8 头、root 流 double 2 / single 4、body 流 double 3 / single 6，约 40–60 M 参数（MotionCraft 默认 1024 / 3+6 / 3+6）。
- batch 256 窗口，AdamW lr 2e-4，warmup 2k，100 k 步；单张 A100 约 1 天。验证：val 损失 + 每 10 k 步 64 条短闭环（站立/行走成功率）作为早期预警，正式数字只在 test 上按协议算一次。

## 7. 评测与对照表

按 docs/06：R-Precision / FID / MM-Dist / Diversity（/ MModality）+ Floating / Penetration / Foot-sliding / Skating / Jerk + Duration。对照行：运动学真值、物理真值（我们的 PULSE 数据）、UniPhys 官方权重（我们跑）、UniPhys / CLoSD / PDP（论文，Guo 评估器）、MIND / SCRIPT（论文，评估器不同只作趋势参考）。

## 8. 需要你定的项（默认值）

| 项 | 默认 | 备选 |
|---|---|---|
| 历史 H / 未来 F / 执行 K | 16 / 16 / 4 | 4/16/8（UniPhys 式），16/4/1（MIND 式） |
| 动作 | 69 维 action | 32 维 pulse_z（第二版） |
| 历史噪声增强 | 开，σ=0.05（归一化单位），p=0.5 | 关 |
| 文本编码 | CLIP ViT-B/32 token + pooled | HyText（Qwen3 + CLIP-L）后续 |
| 模型 | hidden 512，2+4 / 3+6 | 768，3+6 / 3+6 |
| Euler 步数 / CFG | 32 / 2.5 | 10 / {1.5, 3.5} |
| 镜像数据 | 不用 | 物理镜像（左右刚体和关节互换、y 取反、动作对应互换）第二版 |
| 进度条件 | 开（已过/总长两标量进 AdaLN） | 关 |
| BABEL 辅助文本 | 关（第二版对照） | 开（第二路 token，dropout 0.5，测试置空） |

## 9. 实现清单（步骤 6，需另行批准）

- `hml_phys/tokens.py`：调用 UniPhys `motion_repr_utils` 做规范化与 token 计算（训练与闭环共用）；统计量文件。
- `hml_phys/dataset.py`：滑窗数据集、文本分配、CLIP 缓存、历史噪声增强。
- `hml_phys/mc_model.py`：从 `vendor_motioncraft/models/raw_motion/unified_kimodo_flow_dit.py` 和 `models/codeflow/dit_blocks.py` 派生，改输入维度（15 / 420）、桥接维度、observed_mask 语义；flow 调度直接用 `models/raw_motion/flow_schedule.py`。
- `scripts/hml_phys/train_mc.py`：训练脚本（单卡），日志与 ckpt。
- `hml_phys/mc_policy.py` + `UniPhys/main_hml_rollout.py --policy mc`：闭环。
- 评测直接用 `scripts/hml_phys/07_eval_rollouts.py`。

## 10. 定稿（用户 2026-09-15 确认；参考对象改为 MIND，当前最强且结构最接近）

MIND（2605.26006）的相关设定：历史 16 帧；只生成动作，4 帧 horizon；动作 DiT 以"意图"为条件——意图 = 未来状态经 1D 因果卷积 VAE 压成 32 维隐变量（时间 1/4 下采样），分近期（16 帧）与整段（全序列均匀下采样到 16 帧）两个尺度，由文本预测；6 块 DiT、768 维、8 头；文本 CLIP ViT-L/14，两层 Transformer 适配，交叉注意力注入（消融中优于 AdaLN）；文本 dropout 0.1，CFG 3.5；x0 预测；batch 448、80k 步、8×4090 约 12 h；损失三项等权；无历史噪声、无 DAgger；评测起始为统一中性站姿；代码未发布。

| 项 | 取值 | 依据 |
|---|---|---|
| 历史 / 未来 / 执行 K | 16 / 32 / 4（T=48，滑窗 stride 1） | MIND 历史 16、动作 4；我们的 32 帧未来状态预测充当 MIND 的近期意图（16 帧）角色 |
| 模型 | hidden 768，8 头，root 2+4 / body 3+6（≈100M）；备选 512 | MIND 6×768 |
| 文本 | CLIP ViT-L/14 token + pooled，dropout 0.1；注入沿用 MotionCraft 的联合注意力 + AdaLN | MIND 用 L/14 与交叉注意力 |
| CFG | 3.5（扫 2.5 / 5） | MIND 3.5 |
| 进度条件 | 开：(已过时长/总长, 总长) 两标量嵌入 | MIND 的整段意图起同样作用，我们用最简形式 |
| 历史噪声增强 | 默认关，保留开关（σ 0.05，p 0.5） | MIND 不用；若闭环不稳再开 |
| EMA / t 分布 | EMA 0.995 每 10 步；logit-normal(−0.8, 0.8) | MotionCraft 生产配置 unified_kimodo.yaml（其 CLI 默认为 0.9999/1 与 (0,1)） |
| 起步静止增强 | p 0.2：历史替换为首帧重复、速度与动作置零 | MIND 从中性站姿起步，训练数据需覆盖此情形 |
| 损失 | root / body 等权，动作维度权重 1 | MIND 三项等权 |
| 采样 | Euler 32 步（MotionCraft），之后减步 | MIND 未说明步数 |
| 优化 | batch 256，lr 1e-4，100k 步，单 A100 | MIND batch 448、80k 步 |
| 评测起始 | 统一中性站姿（所有方法相同） | MIND |
| 数据 | is_succ 全用；覆盖率与 time_stretch 只记标记；镜像不用 | — |
| BABEL 辅助文本 | 关（第二版对照） | — |

与 MIND 的结构差异（有意保留）：MIND 把"未来状态"压成 VAE 隐变量再做条件，我们让 root/body 双流直接生成未来状态与动作（UniPhys 式联合生成 + MotionCraft 的双流），这是本工作要验证的点；MIND 的整段意图对应我们的进度条件，第二版可升级为"整段计划"隐变量。

说明：文中"KIT"指 AMASS 的 KIT 子集（HumanML3D 的最大来源，100 fps），不是 KIT-ML 数据集。

## 11. 起步问题的处理（用户 2026-09-15 确认；评测统一中性站姿，不用真值摆位）
1. 数据自身覆盖：48.8% 的训练片段开头 0.5 s 接近静止且全部直立（关节均速 <0.15 m/s）。
2. 静止起步增强 p=0.1：历史 16 帧替换为窗口首帧姿态重复、速度置零、动作 = 保持该姿态的 PD 目标（首帧关节角反算：(dof_pos − offset)/scale）。
3. 中性站姿增强 p=0.05：历史直接用评测所用的固定中性站姿（PHC 默认站姿）。
4. 推理预热：rollout 开始先执行数步"保持站姿"动作，使历史缓冲区为真实物理帧。
5. 进度条件在起步时为 0。

## 12. 其余对齐项
- 损失：flow（速度空间）+ 位置速度一致性（0.01），两项。
- 训练：单 A100，batch 256，100k 步（约 1 天），不做训练中闭环抽查；取固定步数权重；测试集只跑一次。
- UniPhys 对照行改为中性站姿起步重跑（25 min），全表同口径。
- 文本分配：带时间标签的句子分给重叠窗口；否则用整句；都无则空文本。
- 文本编码器 CLIP ViT-L/14（需下载）。

## 13. 实现细节与有意偏离（审查 A/B 后记录，2026-09-15）
- 帧约定：PHC 记录器在物理步之后写状态，与同一步施加的动作配对；token 第 t 行 = (该步之后的状态, 该步的动作)。闭环历史缓冲区按同样配对压入，未来第一行的动作就是当前状态下要执行的动作。训练与闭环一致。
- 一致性损失只覆盖 root 平移与 root 线速度（关节位置/速度的关系依赖逐帧朝向与根系变换，不做）。
- CFG 为两分支（空文本 / 文本，历史两侧都给），不是 MotionCraft control 协议的四分支（我们不训练"无历史"分支）。
- body 流的桥接在历史帧上用观测到的 root 而非预测 root；输出投影零初始化；标量条件的最后一层零初始化。
- 文本分配按"窗口未来段与时间标签重叠"，增强窗口的未来段为片段 [0:32)。无描述窗口、dropout、CFG 无条件分支三者统一用 CLIP("") 特征。
- 中性站姿增强：把片段未来段绕竖直轴旋转到与中性姿态的髋部朝向一致再拼接；只对开头静止且直立（根高 >0.8 m）的片段做。
- 进度条件：训练 = (窗口起点 + 16) / 片段帧数，测试 = 已执行帧 / (真值长度/20 × 30)。子片段条目的"总长"是子片段长度，训练侧是整段长度，属已知近似。
- 闭环评测必须关闭模仿环境的参考跟踪终止（termination_distances = 1e6），否则不跟踪隐藏参考动作的策略会在 1 s 内被判摔倒。
- 模型 181M 参数（双流块含文本流），大于文档估计的 100M。
- root 流输入为完整 token（root + body + 掩码），只输出 root（审查 B 的 M2；MotionCraft 的 root 阶段同样看完整状态）。
- 归一化统计：局部骨盆 xy 恒为 0、若干 6D 旋转分量由关节结构固定，这些维度 std 落到 sqrt(1e-5)；逐帧位移 local_vel 的 std 在 0.006–0.05 量级，是真实尺度不是异常。统一按 z-score，不加下限（UniPhys 同样不加）。
- 闭环预热的 2 步不记录、不计入长度，与 UniPhys 对照行一致。

## 14. v2（2026-09-16，用户定）
- 整段预测：窗口 = 16 帧历史 + 未来到片段末尾（上限 304 帧），可变长度，`valid` 掩码；测试时未来长度 = 目标总长 − 已执行帧数（下限 4，上限 304），每 4 步重规划。进度标量保留。
- 模型 hidden 512（81.6M）；bs256 × 5 万步；ckpt 每 1 万步 + best_val；ckpt 选择只在测试集完整协议下做（项目 CLAUDE.md §1）。
- 归一化统计按整段窗口重算（token_stats_v2.npz）。其余与 §10–§13 相同。
