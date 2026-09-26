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

## 15. v3 方案（2026-09-18 用户确认）

v1/v2 的结论：文本匹配 0.254（v1 25k），物理指标已达真值水平，摔倒率与文本理解是短板；逐帧生成整段（v2）又贵又坏，弃用。v3 只做四项改动，全部是修正与补齐，不引入新模块。

### 改动 1（必改）根到身体的局部根桥接
现状：`mc_model.forward` 把预测出的 15 维**绝对**根 token 直接送给 body 流。KiMoDo、ARDY、MotionCraft 旧版、moge_UMO_ST 四个实现全都先转成局部速度根，`hy273_root_conditioning.py` 在两个 MotionCraft 仓库里逐字节相同。这是实现退化，必须补回。

做法（对应我们 z-up、30 Hz、15 维根 token = root_trans 3 + root_rot_6d 6 + root_trans_vel 3 + root_rot_vel 3）：
- 由预测的 root token 算 4 维局部根：`[0]` 偏航角速度（由相邻帧 root_rot_6d 的水平朝向对算 atan2(cross, dot) × fps；朝向取 6D 的第 0、2 个元素，因为 6D 是前两列按行展平 = [M00,M01,M10,M11,M20,M21]，机体 x 轴是第 0 列）、`[1][2]` 相邻帧 root_trans 的 x/y 差 × fps、`[3]` root_trans 的 z（根高）。最后一有效帧复制前一帧。
- FP32 计算；用**单独的局部根均值方差**归一化（与 token 统计一起存在 `token_stats_v3.npz` 的 `local_root_mean/std`，按训练时的同一采样律统计）。
- 训练时 `detach`（现有逻辑保留），测试时保留梯度。
- body 流输入从 `[bridge_root 15 | z_body 420 | mask 420]` 改为 `[local_root 4 | z_body 420 | mask 420]`。
- 历史帧的桥接仍用观测值（现有 `bridge = bridge*(1-m) + z_root*m` 的语义保留，在转局部根之前施加），差分跨历史与未来边界时用真实的最后一帧历史，保证连续性精确。

### 改动 2（必改）回到短未来
`--whole_sequence` 关闭；F 回到 32（可配 16），K=4。v2 的整段逐帧生成（F=304）证实：摔倒率 29%→75%，rollout 从 90 min 涨到 3.5 h，损失被永不执行的远期帧主导。

### 改动 3（确认）SCRIPT 采样 + ARDY 带符号位置编码的长历史
依据：SCRIPT 消融，去掉历史 R@1 0.117、均匀采样 0.419、非均匀采样 0.435（+0.318，是所有论文中单项收益最大的）；且"更长历史"反而 R@1 0.395、FID 0.166，存在甜点。

做法：
- **取帧**：近期稠密 `N_s = 16` 帧原样保留；远期从其余 `L_max − N_s = 138` 帧中按 SCRIPT 式(6) 的指数偏置抽 `N_l = 16` 帧：`I_i = ⌊L_distant·(1 + ln(1 − u_i(1 − e^(−α)))/α)⌋`，`u_i ~ U[0,1]`（每次随机，本身是增强）。`L_max = 154`（30 Hz 约 5.1 s），`α` 可配。
- **接法**：不新增交叉注意力模块（SCRIPT 用两级交叉注意力，ARDY 放同一条序列，我们取后者）。历史 token 仍在同一窗口、同一双向自注意力、仍是观测帧不算损失。
- **位置编码**：改为 ARDY 式带符号索引——`token_index = 真实帧偏移`，第一个生成帧为 0，历史为负（稠密 −16..−1，稀疏按抽到的真实偏移落在 −154..−17），未来 0..F−1。需要支持负索引的位置编码（RoPE 位置可直接传负值，或按 ARDY 写一张正负拼接的正弦表）。这是非均匀间隔能被正确理解的前提。
- **窗口规模**：历史 32 + 未来 32 = 64 token（现为 48），算力约 1.8×，远低于 v2。
- **训练时随机化** `N_l`、`α`，偶尔 `N_l = 0`，使历史长度成为测试时可扫的旋钮，并用于定位 SCRIPT 所述的甜点。

### 改动 3b（与 3 绑定）坐标原点移到最新历史帧
历史拉到 5 s 后，若仍以窗口最旧帧为原点，"现在"可能已在七八米外，落在归一化统计覆盖最差的区域（v2 的教训：root_trans std 从 0.18/0.34 涨到 0.49/0.77）。
- `canonicalize` 改用第 `N_s − 1` 帧（最新的历史帧）建坐标系：该帧根的水平位置移到原点，按该帧髋部朝向旋转到 +y。
- 重算 token 统计（`token_stats_v3.npz`）与局部根统计。
- 不需要朝向 prefix token：ARDY 需要它是因为它只平移不旋转，我们做了旋转规范化，一切相对。
- 不影响执行：执行的是 69 维关节目标，与世界坐标系无关。

### 不做
- FSQ / 任何对身体或动作的量化（姿态错是瑕疵，动作错是摔倒）
- 逐帧生成整段（v2 已证伪）
- MIND 的意图 VAE（留待改动 3 的数字出来后再评估）
- 关闭 pooled 文本进 AdaLN（用户：暂无必要）
- 换文本编码器（MIND 用 CLIP-L 即达 0.468，非瓶颈）
- 数据量与镜像（追平 MIND 之前不碰）
- 帧级语义通道（需新设计，非移植；BABEL 仅训练时可用）

### 其余保持不变
模型 hidden 512（81.6M）/ 768；rectified flow、x0 预测、速度空间损失、1−t 下限 0.05、logit-normal(−0.8, 0.8)；CLIP ViT-L/14 token + pooled 双通路；进度条件；一致性损失 0.01；AdamW 1e-4/0.01、bf16、EMA 0.995/10、文本 dropout 0.1；静止起步增强 p=0.1 与中性站姿增强 p=0.05；Euler 32 步、CFG 3.5；评测统一中性站姿、随机选句、逐条目目标长度、20 次重复；**ckpt 选择只在测试集完整协议下做**（项目 CLAUDE.md §1）。

## 16. 审查后的修正（2026-09-18，两份独立审查 logs/hml_phys/review_v3_A.md、review_v3_B.md）

两份审查一致指出两个严重缺陷，B 另发现第三个，均已修复并补了回归测试：

1. **局部根「最后一有效行复制前一行」索引错**（两份都列为 critical）。参考实现 KimodoRootConditioner 假设补零在**末尾**，而 v3 的训练窗口把未用的稀疏槽补在**开头**，于是 `n_valid-1` 落在未来段中间：约 95% 的训练样本里有一行未来帧的桥接被前一行覆盖，而真正的最后一行从未被写、保持为 0。闭环侧没有前补零，所以这同时是训练与测试的不一致。已改为**按 valid 掩码定位最后一个有效行**，两端补零都正确；numpy 参考同步修正。
2. **`heading_quat` 把原点帧的朝向强加到跨度第 0 行**。UniPhys 把第 0 帧的朝向四元数硬编码为 (0,0,0,1)（这是绕 z 的 180 度，不是单位四元数），因为规范化后该帧恰好是退化情形：forward 与目标恰好反向，四元数的轴与标量部分同时为零。v3 把原点移到最新历史帧后，这个硬编码仍打在第 0 行，使该行以及受其影响的第 1 行 local_vel 带上任意偏航误差。凡是 `l_distant=0` 的窗口（每段开头、闭环缓冲区未满时的每次重规划）必然命中。已把 origin 贯穿到 get_repr / heading_quat，硬编码打在 origin 行。
3. **6D 旋转取错了两个分量**。token 里的 6D 是 `as_matrix()[..., :-1].reshape(6)`，按行展平后是 [M00,M01,M10,M11,M20,M21]；机体 x 轴是第 0 列 (M00,M10,M20)，水平分量应取第 0、2 个元素。原实现取第 0、1 个，即世界 x 轴在机体系下的表示，其角度是**偏航角的相反数**——左转会给出向右的角速度，与平移通道自相矛盾。已改为 `rot6d[..., [0,2]]` 并重算统计。纯偏航构造验证：原实现 −0.700，正确 +0.700。

其余修正：稀疏采样去重后改为**按同一分布重抽**而非用最近帧补齐（原做法使 alpha=0 的均值为 73 而非均匀的 68.5）；统计改为按训练时的随机采样律拟合（平均稀疏帧 7.3）；alpha 训练区间下探到 0，使 SCRIPT 消融里的「均匀采样」在测试时可达；统计缓冲区改为非持久化，v1/v2 的 ckpt 仍可加载；闭环对 v3 之前的 ckpt 自动回退到绝对位置编码以保持其原始协议；闭环新增 `+hml.h_sparse / +hml.alpha / +hml.l_max`，历史长度成为测试时可扫的旋钮；`08_paper_metrics` 的 Duration 分母改为真值长度（去掉执行余量）；`collate` 带出修正后的 `n_hist`；CLI 默认值对齐 §14 与 §15。

**规则冲突的处理**：训练脚本原本会存一个 val 损失最低的 ckpt，而项目 CLAUDE.md §1 明令「不筛选 ckpt」、val 损失只能作健康监控。已**停止写 best_val.pt**，只按固定步数存档，val 损失继续记曲线。docs/06 §2.1b 中现存的 mc_v1 best_val 行产生于该规则确立之前，保留并注明来历。

**回归测试**（scripts/hml_phys/check_tokens.py 已扩充）：非零 origin 下所有非原点行的朝向误差 0.000 度、原点行确为 180 度偏航；局部根在前补零、后补零、无补零三种布局下一致（3e-8），与 numpy 参考一致（1.7e-7）；批量与逐窗口分词一致。数据加载吞吐（8 worker、bs256）：v3 窗口 63 ms/batch，v1 式窗口 21 ms/batch，约 3 倍代价但远低于 GPU 步时。

## 17. v4：按部位结构化（MoGeFlow 样式，2026-09-19 用户定）

**动机（v3 的诊断结论）**：v3 的躯干流里文本注意力占比只有 7.2%（根流 20%），语义要经过 4 维 local-root 这个瓶颈才能到达真正被执行的动作通道；CFG 扫描也证实引导救不回来（R@1 饱和在 0.236，摔倒率从 27% 升到 79%）。用户的判断是直接换掉本体：**不再做 root→body 两段**，改成自己 MoGeFlow 的按部位结构化写法。**VQ/码本部分不要**（我们的 token 是连续物理特征）。

**照搬 MoGeFlow 的部分**（`vendor_mogeflow/models/codeflow/part_structured_motion_code_flow.py`）：
- 每个部位一套 `LayerNorm(affine) + Linear` 入口投影，拼成一条整宽 per-frame token，过**同一个** DiT 主干（double-stream + single-stream、文本-运动联合注意力、AdaLN-Zero、q/k RMSNorm、SwiGLU、1 轴 RoPE），出口每部位一个零初始化 `FinalLayer`。
- 硬约束 `hidden = 部位数 × part_dim`。（「部位边界落在注意力头边界上」是**我们自己加的**约束：MoGeFlow 只断言 `hidden % num_heads == 0`，它发布的 768 = 6×128、12 头×64 碰巧也是每部位 2 头。这条约束等价于 `heads % 6 == 0`。）
- DiT 模块实际从 `vendor_motioncraft/.../dit_blocks.py` 加载（经 `mc_model._blocks`），与 `vendor_mogeflow` 那份逐行对比过：MotionCraft 版只多一条 `control_input_dim=0` 时不生效的控制分支，所用各块（含 `FinalLayer`）完全一致。
- `cond = timestep + pooled text`。

**丢掉的**：root/body 两段、4 维 local-root 桥接、码本/VQ 的一切。
**保留我们自己的**：稀疏长历史（SCRIPT eq.6）+ 带符号帧位置（ARDY）+ observed-mask 硬填充 + 进度/总长标量条件 —— MoGeFlow 是纯文生动作、没有任何历史机制，这几项只能我们自己留着，mask 通道与部位输入拼在一起（`Linear(2*d_p → part_dim)`）。

**部位划分**（`hml_phys/tokens.py: part_channels()`，24 body / 23 joint，435 通道无重叠全覆盖）：

| 部位 | bodies | 通道数 |
|---|---|---|
| root | 0（+ 15 维根 token） | 21 |
| spine | 9–13 | 90 |
| left_arm | 14–18 | 90 |
| right_arm | 19–23 | 90 |
| left_leg | 1–4 | 72 |
| right_leg | 5–8 | 72 |

每个部位带走**自己关节的 69 维 PD 动作通道**（对应 MoGeFlow 把足部接触挂在腿上的做法），动作通道 100% 被覆盖。该划分即 PHC 自己的 `limb_weight_group`（`humanoid.py:390-395`）再把骨盆单列。**与 MoGeFlow 的差别**：它发布的是**有重叠**的六部位划分（`part_vq_hml3d_overlap_best_top3.pth`），我们的无重叠；手臂部位的入口投影看不到躯干，交给共享主干去混合。

**规模（用户定「512 3 6」）**：512 不能被 6 整除，向下取到 `lcm(6 部位, 12 头) = 12` 的倍数 → **hidden 504 = 6 × 84，12 头 × 42，每部位正好 2 个头**；3 double + 6 single，mlp_ratio 4.0 → **69.1M**（< 100M 上限，v3 是 81.6M）。

**用户定的其余四项**：① 仍预测 x0、损失换到速度空间（不变）；② t 分布（见下，审查后改为 logit-normal）；③ 采样 Euler **32 步**、**CFG 3.5**（采样时间网格均匀）；④ 规模控制在 100M 内。

### 17.1 审查后的三项设计修改（2026-09-19，用户同意；审查 logs/hml_phys/review_v4_{A,B,C}.md）

1. **训练 t 分布：均匀 → logit-normal(−0.8, 0.8)**（与 v3 / MotionCraft 相同，`--t_dist logit_normal`）。我们预测 x0 再换算速度 `v̂ = (x̂0 − z_t)/(1−t)`，网络输出上的梯度被乘以 `1/(1−t)`（上限 20，`v_eps=0.05`）。实测：`E[1/clamp(1−t,0.05)²]` 均匀 38.9、logit-normal 2.96；均匀下 t>0.95 占损失权重 51%、t<0.6 只占 4%（logit-normal 为 ≈0% / 75%）；同数据 100 步 A/B 梯度范数均匀 20–49、logit-normal 0.85–2.0，`grad_clip 1.0` 下均匀会每步被裁 20–50 倍（v3 收敛时 gn≈0.41，从不被裁）。Euler 从 t=0 积分，大体轨迹由低 t 决定，而 v3 的探针结论正是低 t 时条件信号弱于噪声 3–6 倍。
   **更正**：MoGeFlow 配置默认 `--time_schedule logit_normal`（p_mean −1.5），但它实际的 HumanML3D 训练脚本 `scripts/launch/train_humanml3d_pscf_standard.sh:58` 传的是 `--time_schedule uniform`。二者不矛盾：MoGeFlow **直接预测速度**、无 `1/(1−t)` 因子，所以均匀 t 对它是安全的；MotionCraft 预测 x0，所以用偏向低 t 的 logit-normal。两个仓库各自的 t 分布都与自己的参数化匹配，「x0 预测 + 均匀 t」两边都没用过。
2. **损失按部位平均**（MoGeFlow `part_structured_motion_code_flow.py:289-290`）：先在部位内取通道均值，再对 6 个部位等权平均。之前写成 435 通道统一均值，根 token 只占 3.45%（v3 为 50%）、动作通道 15.9%；现在每部位 1/6，根 token ≈11.9%。`w_root / w_body / w_action` 在部位内部生效。已用「误差只放在某一个部位」验证每部位贡献恰为 1/6。
3. **LayerNorm 丢掉的统计量拼回去**：部位入口改为 `Linear(2·d_p + 2 → part_dim)`，输入 `[LN(z_p) | mask | mean(z_p), std(z_p)]`。MoGeFlow 归一化的是同质的 VQ 码向量；我们的部位向量是异质物理量，审查 B 在 3,520 帧测试 token 上实测 LayerNorm 去掉的方差占比：root 31.6%、腿 16.5–18.7%、脊柱/手臂 ≈12%（根高度 42%、根角速度 x 44%）。拼回后 `LN(z)·std + mean` 精确还原 `z_p`（已验证），参数量不变（69.14M）。

### 17.2 审查发现的实现问题（已修复）

- **CRITICAL** `mc_rollout.py` 启动横幅读 `model.local_root`，v4 每次闭环评测都会在第一步前崩溃 → `getattr`，横幅加 `arch`。
- 头数约束实为 `heads % 6 == 0`（改 hidden 修不了）→ 不满足时直接报错；hidden 取整到 `2·heads` 的倍数（RoPE 要偶数 head_dim）。
- 两个探针 `probe_text_attention.py` / `probe_history_use.py` 增加 part 分支（v4 的动机「躯干流文本注意力 7.2%」训后要能复测）。
- `check_tokens.py` 增加部位划分永久回归测试（覆盖、无重叠、每部位的 bodies/joints 通道集合、往返）。
- `mlp_ratio / dropout` 写入 ckpt args 并由 `load_policy` 读取；`--resume` 从 ckpt 恢复结构参数；args 日志移到取整之后；预测按 `valid` 掩码（MoGeFlow 同）；`max_text_tokens` 生效；`vendor_mogeflow/` 加入 .gitignore；过期注释。
- 两段结构（v1–v3）路径回归不变（81.6M，损失与日志格式一致）。

**代码**：`hml_phys/part_model.py: PartPhysPolicyDiT`；`flow.py: euler_sample_single` + `sample_t(dist=)`；`train_mc.py --arch part --hidden 512 --heads 12 --depth 3,6 --t_dist logit_normal`（启动脚本 `scripts/hml_phys/train_v4.sh`）；`mc_rollout.py` 按 ckpt 的 `args["arch"]` 自动选分支，v1–v3 的 ckpt 照旧走两段路径。

**冒烟测试（修改后重跑）**：60 步训练 flow 3.01→2.06，梯度范数 1.8–4.3（v3 第 100 步 3.76，同量级）；ckpt 经 `load_policy` 复原（arch=part、t_dist=logit_normal、69.14M），32 步 CFG 3.5 采样有限、历史帧逐位相同。注意这只证明链路通，零初始化输出头使「采样有限」在未训练时平凡成立；`run_rollouts_mc` 的闭环本身要等第一次真实评测才会走到。

## 18. v5：v4 + 文本交叉注意力（2026-09-19 用户定）

**动机（MIND 的发现 + 我们自己的测量）**：MIND 消融里，在只有 AdaLN 的基线上加一条**文本交叉注意力**是单项最大的收益（R@1 +0.142）。交叉注意力的 softmax 只在文本 token 上归一化，运动 token 必须读文本；我们用的是 MotionCraft 的联合注意力，运动和文本在同一个 softmax 里，运动 token 可以几乎不看文本——v3 实测 body 流运动 query 分给文本的质量只有 7.2%（均匀参考 20.8%）。v4 去掉了 4 维桥接，但仍是联合注意力，所以这条要单独补。

**实现：照搬用户 moge_UMO_ST 的 `local_text_cross_attention`**（commit `0844a1f`；`vendor_moge_umo/models/codeflow/dit_blocks.py: FrameMotionTextDiT._inject_local_text`，L1160–1221；`models/raw_motion/llm2vec_cache.py: local_proj`，L86–89）：
- **位置**：每个双流块之后（我们是 3 个），单流块不加。
- **更新**：`motion ← motion + tanh(g_i) · joint_attn_i(LN(motion), M, key_valid=文本有效位, query_valid=运动有效位)`。
- **权重复用**：`joint_attn_i` 就是该双流块自己的注意力模块（同一套 q / kv / out 投影、q/k RMSNorm），交叉注意力不加 RoPE。与 moge_UMO_ST 一致。
- **query 归一化**：每块一个无仿射 `LayerNorm(eps 1e-6)`。
- **文本记忆 M**：`LayerNorm(768) + Linear(768→504)` 作用在原始 CLIP 词级特征上，与联合注意力里的文本流（`token_proj`）分开；padding 位置零、并被 key 掩码屏蔽。
- **门控**：每块一个标量 `g_i`，零初始化，所以**初始化时 v5 与 v4 逐位相同**。新模块在 `torch.random.fork_rng` 里构造（moge_UMO_ST 同），其余所有权重的初始化与 v4 完全一致。
- 新增参数 389.1k → **69.53M**。

**与 moge_UMO_ST 的差别（有意）**：
- 文本编码器：它用 LLM2Vec 的上下文 token 缓存，我们只有 CLIP ViT-L/14 词级特征（与联合注意力同源），用单独的投影区分两条通路。
- 无条件分支：它在文本 dropout 时把 local 记忆整段置为 padding、更新清零；我们的 CFG 无条件分支一律用 CLIP("") 的特征（训练 dropout 与采样一致），交叉注意力也读这组空句 token。两种写法在各自的训练/采样之间都是自洽的。
- **联合注意力里看到的文本不同（审查 v5-A 指出，已核实 `llm2vec_cache.py:229-246`）**：moge_UMO_ST 只在 `sentence` 模式下产生这条局部记忆，此时联合注意力里只有**一个句子 token**，交叉注意力是**唯一**逐词读文本的通路；我们的联合注意力照旧看全部 50 个 CLIP token，交叉注意力把同一组词再读一遍（还包括 CLIP 的起止 token，moge 会跳过指令 token）。所以 v5 检验的是「在联合注意力之上**再加**一条只在文本上归一化的 softmax」，不是 moge 的原配置；与 MIND 也不同（MIND 的 R2 是用交叉注意力**替换** AdaLN，文本只从交叉注意力进）。另：vendor_moge_umo 里签入的配置都没有打开这条通路（全部为 false），没有现成的训练配方可比门控行为。

**代码**：`part_model.py: PartPhysPolicyDiT(text_cross_attention=True)` + `_trunk_with_text_xattn`（逐行复刻 vendor_motioncraft `FrameMotionTextDiT.forward`，只在双流块后插入；关闭时仍直接调用原 backbone，v4 路径不变）；`train_mc.py --text_xattn 1`（写入 ckpt args，`--resume` 恢复）；`mc_rollout.load_policy` 读 `text_xattn`；`probe_text_attention.py` 只统计联合注意力（key 数 = 运动 + 文本槽），另报每块的门控 `tanh(g)` 与注入量 `|gate·update|/|motion|`。启动脚本 `scripts/hml_phys/train_v5.sh`（除 `--text_xattn 1` 外与 v4 完全相同）。

**冒烟**：同种子构造，共享权重逐位相同；门控为 0 时 v5 输出与 v4 **逐位相等**；门控 0.5 时改变有效文本 token 使输出变化 0.057，只改 padding 槽输出逐位不变；60 步训练损失与 v4 冒烟一致到小数点后 4 位（门控从 0 起步，符合预期），门控离开 0（≈5e-4）；ckpt 复原、32 步 CFG 3.5 采样有限且历史帧逐位相同；v4 的 ckpt 仍可严格加载；探针在 v5 ckpt 上正常输出联合注意力与交叉注意力两部分。

## 19. MIND 意图机制精读（2026-09-19，arXiv 2605.26006 全文 §4.2–4.4、表 2、图 5、图 7、附录 A/B）

**消融表 2 的行号以论文图 5 的标注为准**：R1 = AdaLN；R2 = 交叉注意力；R3 = R2 + IIP；R4 = R3 + VAE；R5 = 「Ours w/o IIP」（交叉注意力 + HIP + VAE）；R6 = R5 + IIP（完整）。

| 行 | 配置 | R@1 | 相对上一档 |
|---|---|---|---|
| R1 | AdaLN（单个全局 token） | 0.174 | — |
| R2 | 词级文本交叉注意力 | 0.316 | +0.142 |
| R3 | R2 + 近期意图 IIP（未来 16 帧，原始状态空间） | 0.323 | +0.008 |
| R4 | R3 + VAE 压缩 | 0.360 | +0.037 |
| R5 | R2 + 整段意图 HIP + VAE，**无近期意图** | 0.292 | 比 R2 **低 0.024** |
| R6 | 全部 | 0.468 | 比 R4 +0.108 |

**更正（此前对话里的错误读法）**：我曾把 R4 读成「加整段意图」、R5 读成「近期 + VAE 无整段」，据此说「近期未来几乎不值钱、关键是整段」。正确读法相反：**近期意图是地基**（压缩后共 +0.045），**整段意图单独用有害**（R5 < R2），只有叠在近期意图上才带来最大的一跳（+0.108）。论文原文：「relying solely on holistic intent … instead degrades performance … lacking sufficient temporal specificity」。

**具体做法**
- 状态 358 维：根高 1、非根关节在根坐标系下的位置 23×3、24 关节 6D 局部旋转、24 关节局部线速度与角速度；**不含动作**。
- **意图 VAE**：1D 因果 ResNet 卷积编码器/解码器，时间 4 倍下采样，隐变量 32 维；σ-VAE 损失 `L_rec + λ_KL·KL`，λ_KL = 1e-5。**先单独训练，训完冻结，只保留编码器**。图 7：32 维优于 16/64/128/256；1/4 下采样明显优于 1/2 和不下采样。
- **HIP（整段意图）**：一个扩散模型，文本 → 整段意图。训练目标是整段状态**均匀重采样到 16 帧**后的编码（→ 4 个 32 维隐向量），与片段时长无关；测试时每条指令生成一次。
- **IIP（近期意图）**：自回归扩散模型，条件为文本、历史 16 帧的编码 `E(S_h)`、HIP；目标是未来 16 帧的编码 `E(S_{t+1..t+16})`。HIP 与 IIP 共用同一个隐空间。
- **ADiT（动作）**：只生成 4 帧动作；历史用**原始状态**（不压缩）；文本 token、HIP、IIP 都经**交叉注意力**进入。喂给 ADiT 的是 HIP / IIP 的**最后一层隐状态**，而不是它们去噪出来的隐变量。
- 三个模型都是 6 块 DiT、768 维、8 头；文本是 CLIP ViT-L/14 加两层 Transformer 适配器；**联合端到端训练** `L = L_HIP + L_IIP + L_ADiT`（均为 x0 预测的 MSE）；CFG 以 10% 概率遮文本 token，scale 3.5；batch 448，80k 步。
- **附录 B**：UniPhys 用一个共享解码器联合建模状态与（隐）动作，作者认为状态和动作之间的模态差异会互相干扰，削弱文本对动作的直接作用。

**论文没写清的（代码未发布）**
1. 「最后一层隐状态」在训练时取自哪个噪声水平：训练时 HIP/IIP 的输入是随机 k 加噪后的隐变量，测试时大概是去噪最后一步的；ADiT 的梯度是否回传进 HIP/IIP 也没说（「端到端联合优化」暗示回传）。
2. 整段意图重采样到 16 帧后丢掉了时长；ADiT 怎么知道自己处在整段的哪个位置没说（我们的 progress / total_len 标量可以补上）。
3. HIP / IIP 在测试时的采样步数没说。

**与我们的对照**：v4/v5 是 UniPhys 式的联合生成——同一个 token 里一起去噪 32 帧**原始**未来状态和动作，相当于 MIND 的 R3 档（近期意图、未压缩），且正是附录 B 批评的联合建模；我们没有整段意图，只有 progress / total_len 两个标量。

**审查（logs/hml_phys/review_v5_{A,B,C}.md）后修正**：`--resume` 时 EMA 留在 CPU 导致第一次 EMA 更新崩溃（v1–v4 同样存在，已修）；resume 不再继承旧 ckpt 的 val 损失作为最佳阈值；`--text_xattn` 只允许用于 part 结构；训练日志每行打印三个门控 `tanh(g)`（对正在跑的 v5 不生效，v5 的门控从各存档读）；探针门控改科学计数、删死代码。

## 20. 路线 C：MIND 意图 VAE（2026-09-19 用户定：先只训 VAE，看测试集重建，再做路线 A）

**实现**
- `hml_phys/intent_vae.py`：MIND 引用的 1D 因果卷积即 MotionStreamer（ICCV 2025，Xiao et al.）的 causal TAE，逐行移植自 `vendor_motionstreamer`（commit `8aace3f`：`models/causal_cnn.py`、`models/resnet.py`、`models/tae.py`、`utils/losses.py`）。结构：CausalConv1d → 2 级〔stride-2 因果卷积 + 3 层因果 ResNet（膨胀 1/3/9）〕→ Linear 到 (μ, logvar)，logvar 截断 [−30, 20]；解码器对称（最近邻上采样）。宽 1024。只改两处：输入维 272 → 366，隐变量 16 → 32（MIND）。76.8M 参数。16 帧 → 4 个 32 维隐向量。已验证编码器和解码器都严格因果（第 i 个隐向量只依赖前 4i+3 帧）。
- **损失**：MotionStreamer 的 σ-VAE 实现原样（共享最优 σ 的高斯 NLL，按全部元素求和；KL 对隐维与时间求和、对 batch 取平均），KL 系数取 MIND 的 λ_KL = 1e-5。**注意**：MIND 没说重建项是求和还是平均；按 MotionStreamer 的求和写法，1e-5 的 KL 几乎不起约束作用（冒烟中 |μ| 很快涨到 2–3），相当于 Stable Diffusion 那种 1e-6 量级的 KL（SD 的 VAE 也如此，下游再乘一个缩放因子）。下游扩散前按训练集隐变量的经验方差归一化。
- **状态**：我们 token 的非动作部分 366 维 = 根 15 + body 状态 351（局部位置 72、局部速度 72、关节 6D 138、关节角速度 69），已按 TokenStats 归一化。包含 MIND 358 维状态的全部信息，另多出窗口规范系下的根 xy 与偏航。
- **数据**（`hml_phys/intent_data.py`）：每个训练样本给三条 16 帧序列——历史 t−15..t、近期未来 t+1..t+16（都以第 t 帧、即最新历史帧为规范原点，与策略窗口完全一致，并复用策略的起步静止 p 0.1 / 中性站姿 p 0.05 增强，使闭环起步时的站立历史在分布内），以及该片段的整段序列（全片段先算 token 再均匀取 16 帧，原点为片段首帧，速度仍是瞬时速度）。
- **训练**（`scripts/hml_phys/train_intent_vae.py`，启动脚本 `train_vae.sh`）：MotionStreamer 的优化设置——AdamW lr 5e-5、betas (0.9, 0.99)、无 weight decay、线性 warm-up 1000 步、无梯度裁剪；batch 128 个窗口 × 3 条序列；10 万步（MotionStreamer 原配 200 万步，这里先看 10 万步的测试曲线）。每 5000 步在测试集固定 4096 个窗口上做确定性重建（解码 μ），按三类序列的平均 MSE 存 best_test（项目规定只用测试集）；另报物理单位误差：关节局部位置 mm、根位置 mm、根高 mm、局部速度 m/s（local_vel 存的是每帧位移，×30）、各通道组 MSE、活跃隐维数。

### 18.1 v5 改为 moge_UMO_ST 的 sentence 模式并重训（2026-09-19，用户令「把问题修改一下然后重新开始训练」）

**第一次 v5 的问题（审查 v5-B/C，logs/hml_phys/review_v5_{B,C}.md）**：
1. 门控打不开：第 5000 步 tanh(g) = −0.012 / −0.010 / +0.019，交叉注意力只给运动流加了 0.1–0.3%，损失曲线与 v4 差不到 1e-3；门控符号早期来回游走，文本投影拿不到一致的梯度方向。
2. CLIP 起始 token（slot 0）对所有句子是同一个向量，交叉注意力可以全压在它上面，得到与句子无关的偏置（v3 联合注意力里已有 32–55% 的文本注意力落在它上面）。
3. 联合注意力本来就看得到全部词，交叉注意力只是重复读一遍，没有压力打开门控。
第一次 v5 的产物归档为 `outputs/mc_v5_jointxattn_aborted`（第 5000 步）、`logs/hml_phys/train_mc_v5_jointxattn_aborted.log`，不再使用。

**修改（`--text_mode sentence_xattn`，完全照 moge_UMO_ST 的 sentence 模式）**：
- 联合注意力里只有**一个句子 token**：`sentence_proj = Linear(768→504)` 作用在池化 CLIP 特征（EOT @ text_projection）上（moge：`llm2vec_cache.py:175-196`，单句 token 过 `token_proj`）。
- **AdaLN 条件里不放文本**：`cond = timestep + 标量条件`（moge：sentence 模式下 `pooled` 为全零，`kimodo_like_flow_dit.py:246-263`）。
- 逐词文本**只**经门控交叉注意力进入，key 为 CLIP 词级特征**去掉 slot 0 的起始 token**（moge 同样跳过指令 token）；CLIP("") 去掉起始 token 后剩 1 个结束 token，CFG 无条件分支读它。
- 不再有 `token_proj` / `pooled_proj`；68.9M 参数。
- 已验证（随机化全部 AdaLN-Zero 调制层后）：门控为 0 时改词**完全不影响**输出，只有句子 token 起作用；门控 0.5 时改词使输出变化 0.057；起始 token 与 padding 位置对输出逐位无影响；长度 < 2 的文本被拒绝。v4 与第一次 v5 的 ckpt 仍可严格加载（默认 `joint_tokens`）。

**门控学习率（审查 v5-B 建议的补救，实测后不采用）**：同数据同种子 1500 步短训（batch 128，2000 段）：
| 设置 | 第 100 / 500 / 1500 步门控 tanh(g) | 第 1500 步测试损失 |
|---|---|---|
| sentence 模式，门控 lr ×1 | ±0.0005–0.001 / ±0.002–0.004 / −0.010, +0.007, +0.011（**单调、方向一致**） | 2.0621 |
| sentence 模式，门控 lr ×30 | ±0.02–0.03 / ±0.008–0.024 / −0.019, +0.015, +0.019（先冲高再回落） | 2.0632 |
门控开多大由这条通路当前的用处决定，加大学习率只让它先冲过头再回落，测试损失无差别。保留与 moge 一致的 ×1（`--xgate_lr_mult` 留作开关，默认 1）。对比第一次 v5：同等门控幅度要 5000 步且符号游走。

**重训**：2026-09-19 12:29，1398709 blossom03，`scripts/hml_phys/train_v5.sh`（v4 设置 + `--text_xattn 1 --text_mode sentence_xattn`），setsid 方式启动；训完自动评第 5 万步 ckpt（测试集完整闭环）并跑两个探针。

### 18.2 v5b：只走交叉注意力（2026-09-19 用户令「抛弃掉那个单 token 的路线，只走交叉注意力那条 line」）

`--text_mode xattn_only`：在 v5（sentence 模式）基础上去掉句子 token——联合注意力里**没有任何文本**（双流块的文本流为空，其 `text_mod` / `text_ffn` 参数永远只见到零长度输入，已冻结），AdaLN 里也没有文本；句子级和词级文本**全部只经**门控交叉注意力进入（与 MIND 一致：文本只从交叉注意力进）。起始 token 仍排除在 key 之外。68.5M 参数，其中可训练 54.8M。已验证（随机化 AdaLN-Zero 调制层后）：门控为 0 时输出与任何文本（词和句向量）都无关；门控 0.5 时改词使输出变化 0.057，池化句向量对输出无影响，起始 token 不可见。其余设置与 v4/v5 完全相同。2026-09-19 13:55 在 1476691（pink7024 H200）启动，`scripts/hml_phys/train_v5b.sh`，103 ms/步；训完自动评第 5 万步 ckpt 并跑探针。

### 20.1 VAE v2（2026-09-19 用户批准）

v1（`outputs/intent_vae_v1`）第 10 万步测试集：历史 / 近期未来 16 帧关节位置误差 14.5 / 14.6 mm、根位置 61 / 56 mm、根高 2.5 mm、速度 0.09 / 0.10 m/s，测试与训练 MSE 接近（0.091 vs 0.072）；**整段序列严重过拟合**（测试 MSE 0.50 vs 训练 0.04，关节 49 mm、根 187 mm，且随训练变差）——每个片段只有一条确定的整段采样，全部 8.7k 条被 76.8M 的模型背下；`best_test.pt` 因此停在第 2 万步。后验方差 ≈1e-7（确定性自编码器）。

v2 两处修改（`scripts/hml_phys/train_vae_v2.sh`）：
1. **整段序列扩增**（`--holi_aug 1`，只作用于训练）：概率 0.25 用与测试完全相同的整段均匀采样；否则在片段两端各随机裁掉至多 25%（覆盖 ≥ 50%，实测平均 74%），在该子区间上均匀取 16 帧并加 ±半个间隔的随机相位；token 在子区间上重算，原点为第一个取样帧（与整段定义一致）。已验证 p_exact = 1 时与 v1 的整段序列逐位相同。
2. **根通道额外损失**（`--root_loss 7`）：MotionStreamer 原版做法（`utils/losses.py: forward_root`，对根特征单独拟合 σ 的高斯 NLL，权重 7）；我们作用于 15 维根 token。MIND 未说明是否使用。
其余不变。2026-09-19 13:55 在 1476690（pink7024 H200 GPU0）启动，25 ms/步，10 万步约 45 分钟。

## 21. 路线 A 方案（2026-09-19，用户已批准：§21.4 四项全部取第一个选项；训练 20 万步、每 5 万步存档）

目标：照 MIND 的多尺度意图机制（§19），让语义先落到「压缩后的未来状态」上，再由动作策略执行。VAE 用路线 C 训好的 `outputs/intent_vae_v2/best_test.pt`（冻结，只用编码器）。

### 21.1 三个模型
| 模型 | 输入 | 输出 | 何时运行 |
|---|---|---|---|
| **HIP 整段意图** | 文本 | 整段意图：该句描述的整个片段均匀取 16 帧 → VAE 编码 → 4×32 | 每个 episode 开始时一次 |
| **IIP 近期意图** | 文本 + 历史意图（最近 16 帧 → VAE → 4×32）+ HIP 的末层隐状态 | 近期意图：未来 16 帧 → VAE → 4×32 | 每次重规划 |
| **ADiT 动作策略**（v5 底座） | 历史原始状态 + 文本 + HIP、IIP 的末层隐状态 | **只出动作**：未来 4 帧 × 69 维 | 每次重规划，执行 4 帧 |

- HIP / IIP：各自是一个小 DiT，输入是加噪的意图隐变量（4 个 token；IIP 前面再拼 4 个历史意图 token 作为已知前缀——MIND 图 3 未说明历史意图的注入方式，这是我们的选择），文本经交叉注意力进入（CLIP 词级特征 + 两层 Transformer 适配器，**HIP 与 IIP 共用**；MIND 是三个分支共用，我们的动作策略按 §21.5 保留 v5 自己的文本通路，不经适配器）。
- 意图隐变量按训练集上的经验均值/方差归一化（VAE v2 后验方差 2e-10 至 3e-9，取 μ）；历史、近期、整段三类共用一套统计（MIND：共享隐空间），归一化后整段隐变量的离散度约为另两类的 1.8 倍。
- 整段意图的「整段」：无时间标签的句子取整个片段；带时间标签的句子取它描述的那一段（与测试协议把带标签句子切成独立子片段一致）。
- 进度 / 总长两个标量保留，**且同时给了动作策略和 IIP**（MIND 的整段意图丢掉了时长，两者都需要知道自己走到了哪；MIND 式 (3) 的 IIP 没有这两个标量，属于我们的偏离）。

### 21.2 训练
- 三个模型**联合端到端训练**：`L = L_HIP + L_IIP + L_ADiT`，等权（MIND 式 (5)）。三者都用我们现有的 flow 约定：x0 预测、速度空间损失、logit-normal t。
- ADiT 的损失只算在 4 帧未来动作上；历史行是观测、不算损失。
- 文本 dropout 10%（三个模型同一个 dropout 掩码）。
- 数据：沿用策略窗口（规范原点 = 最新历史帧，起步静止 / 中性站姿增强），每个样本另给三条 16 帧状态序列供冻结的 VAE 在 GPU 上编码；窗口要求未来至少 16 帧（IIP 的目标）。
- **20 万步**（用户定），batch 256，其余优化设置同 v4/v5。**每 5 万步存一次**（5/10/15/20 万）；每 5000 步算一次测试集去噪损失（三项分开记），另存损失最低的 best_test。预计约 5.5–7 小时。

### 21.3 闭环推理
episode 开始：HIP 采样一次（Euler 32，CFG 3.5）。每 4 帧：VAE 编码最近 16 帧 → IIP 采样（Euler 32，CFG 3.5）→ ADiT 采样 4 帧动作（Euler 32，CFG 3.5）→ 执行。ADiT 的序列从 64 行缩到 36 行（16 稀疏 + 16 稠密 + 4 未来），IIP 只有 8 个 token，预计单次 rollout 耗时与 v5 相当。

### 21.4 待定项（需用户拍板）
1. **ADiT 输出**：只出 4 帧动作（MIND；附录 B 认为状态与动作联合建模会互相干扰）vs 保留 v5 的 32 帧「状态 + 动作」联合生成、只额外加意图条件。
2. **意图怎么进 ADiT**：作为额外 token 放进联合注意力的文本流（与句子 token 并列，共 1 + 4 + 4 个）——v5 对 v5b 的结果显示，在我们的主干里联合注意力中的 token 比门控交叉注意力有效得多；vs MIND 原文的交叉注意力。
3. **规模**：HIP、IIP 各 4 块 × 384 宽 vs MIND 原配各 6 块 × 768 宽（MIND 未报参数量）。〔实测更正：4 × 384 在标准 SwiGLU 倍率 4 下总计 108.7M，超过 100M；实施时保留 4 × 384、把意图 DiT 的 SwiGLU 倍率降到 1.5，可训练参数 99.84M。推理时另有冻结的 VAE 编码器 37.9M（整个 VAE 76.8M，推理只用编码器）。〕
4. **末层隐状态怎么取**（MIND 未写清）：训练时用「干净意图输入」下的一次前向取隐状态喂给 ADiT（与测试时「采样到最后一步」的输入状态一致，梯度照常从 ADiT 回传进 HIP/IIP）vs 直接用训练损失那次加噪前向的隐状态（噪声水平训练与测试不一致）。

### 21.5 其余默认（照做，除非用户另有意见）
- ADiT 保留 v5 的稀疏长历史与 sentence 模式文本（v5 是目前最好的底座）。
- CFG 在三个模型上都用 3.5（MIND 只给一个 3.5）；ADiT 的无条件分支同时去掉文本和意图。
- 评测：用户指定的那一个 ckpt（默认第 20 万步），单次 rollout、单次计算（项目 CLAUDE.md §4），完整对照表汇报；新增一个意图探针：把一个样本的意图换成另一个样本的，量动作变化，和「换空句」「换噪声」并列。

### 21.6 实施顺序
① 意图数据与隐变量统计；② HIP / IIP 模块；③ ADiT 改动；④ 联合训练脚本；⑤ 闭环推理接入；⑥ 冒烟测试；⑦ 三个独立审查（项目惯例）；⑧ 经批准后训练（约 2 小时）；⑨ 单次评测。

### 21.7 实施记录（2026-09-19）
- **意图 DiT 的 SwiGLU 倍率 4 → 1.5**：保持 4 层 × 384 宽与 100M 上限（见 §21.4-3 的更正）。可训练参数 99.84M：适配器 3.85 + HIP 13.28 + IIP 13.43 + 动作策略 68.89 + 两个投影 0.38。
- **VAE 历史输入的第 0 帧速度**：VAE 训练时历史序列所在片段从第 0 帧开始，分词器把该帧的局部速度取为第 1 帧的值；策略窗口与闭环缓冲区在此之前还有更早的帧，得到的是真实速度。`hml_phys/intent_data.py: vae_history_input` 在编码前把第 0 帧局部速度替换为第 1 帧的值，训练与闭环共用；修正后与 VAE 训练输入逐位一致（本人 431/431，审查 A 800/800）。
- **IIP 的无条件分支**：训练中文本被丢弃的样本，IIP 收到的是由空句算出的 HIP 隐状态；闭环 CFG 的无条件分支与之一致（HIP 在空句上的隐状态）。动作策略的无条件分支不带任何意图 token。
- **best_test 的挑选**：按三项测试损失之和；HIP 损失占主导（见审查 A），因此它基本反映 HIP 的状态，不代表动作策略最好。默认评测用第 20 万步 ckpt。
- 训练于 16:44 在 1476690 启动，16:47 因用户另一个项目进入该卡而停止（未产生 ckpt），在 1476691 GPU0 同种子重启。

### 21.8 审查后的修改方案（2026-09-19，三份审查 logs/hml_phys/review_A_{A,B,C}.md；**待用户批准**，代码已就绪、默认关闭）
**审查结论**：代码无严重错误；审查 B 用回放真实测试片段的替身环境驱动真实的闭环代码，43 次重规划的全部输入与训练逐位一致。两个实质问题：
1. **HIP 背训练集**：测试集 L_HIP 5k 1.53 → 10k 2.41 → 15k 3.37，训练集 1.03 → 0.54 → 0.39；IIP 与动作策略的测试损失正常下降。每个（片段, 句子）只有一条固定的整段目标。
2. **动作策略只见过真实未来算出的意图**（审查 C 第 5k 步实测，动作误差）：真实未来意图 0.459 / 不给意图 0.881 / 测试时链路 HIP→IIP 采样意图 **1.024**。§21.4-4 选的「干净意图读隐状态」让训练条件过于理想，与测试时想象出的意图对不上。
梯度每步被裁剪不构成问题（审查 C：Adam 按参数归一化，300 步对照中裁剪与否差约 1%）。

**修改（开关）**：
- a `--hip_aug 1`：HIP 的整段目标按 VAE v2 的方式扩增——以 0.25 概率用测试时的精确采样，否则在句子所描述的区间两端各裁掉至多 25%、随机相位取 16 帧。
- b `--cond_aug 0.5 --cond_aug_test 0.75`：条件扰动（级联扩散的 conditioning augmentation）。训练时 IIP 与动作策略收到的意图隐状态，改为在 `z = s·I + (1−s)·ε`、`t = s`、`s ~ U(0.5, 1)` 上读出；测试时对采样出的意图固定取 `s = 0.75` 读出。修改 §21.4-4 的决定。
- c `--select chain`：每 5000 步用测试时链路（HIP、IIP 各 Euler 32、CFG 3.5 采样出的意图）算一次动作损失 act_chain，按它存 best_test。
- d `--span_scalars 1`（审查 B MINOR-3）：带时间标签的句子，进度 / 总长改为相对句子所描述的区间，与评测把带标签句子当作独立片段一致（影响约 10% 的测试条目）。
- 另修：闭环中 IIP 读隐状态包进 `no_grad`；探针的 VAE 编码移出 bf16、新增「测试时链路意图」对照、所有条件共用一次意图读出。
冒烟：a 开启后同一窗口 8 次得到 7 条不同目标；d 只改变带标签句子；c 的 act_chain 正常输出；带 b 的 ckpt 在 16 条上闭环跑通。启动脚本 `scripts/hml_phys/train_A2.sh`（输出 outputs/mc_A_v2）。

### 21.9 ckpt 选择：教师强制损失不可用（2026-09-20）
用同一次训练的两个 ckpt 各做一次闭环评测（单次 rollout、单次计算）：

| ckpt | 训练期 act_chain（测试集） | 闭环 R@1 | 摔倒率 | Duration | Floating | Jerk |
|---|---|---|---|---|---|---|
| 第 3.5 万步（act_chain 全程最低，best_test.pt） | **0.2196** | 0.365 | 48.3% | 0.767 | 19.7 | 4.09 |
| 第 20 万步（act_chain 已升到 0.2733） | 0.2733 | **0.410** | **24.0%** | **0.904** | **16.0** | **2.74** |

即使把选择指标换成「测试时链路意图下的动作损失」（§21.8 c），它依然与闭环表现反向。原因大概率是：教师强制只看单步去噪误差，而闭环成绩主要取决于长时间自回归下的稳定性，后者随训练继续改善（摔倒率减半）。**结论：路线 A 的 ckpt 取训练步数最后一个；best_test.pt 仅作监控，不用于评测。**
