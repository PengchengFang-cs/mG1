# 仿真角色 HumanML3D 物理基准：数据、评测协议与实现（2026-09-15）

本文档记录线 C 的数据集构建与评测流水线，是后续所有数字的"尺子"。代码在 `hml_phys/` 和 `scripts/hml_phys/`。

## 1. 数据

### 1.1 来源
- **物理轨迹**：UniPhys 公开的 PULSE 跟踪 AMASS 数据（HF `yan0116/SMPL_Humanoid_offline_dataset/amass_state-action-pairs`），逐 AMASS 序列一个 joblib pkl，共 11,712 个。字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| body_pos | [T,24,3] | 24 个刚体的全局位置，MuJoCo 顺序，z 向上，米 |
| dof_state | [T,69,2] | 23 个关节 × 3 轴：[...,0] 关节角（轴角），[...,1] 关节角速度 |
| root_state | [T,13] | 根位置 3、四元数 xyzw 4、线速度 3、角速度 3 |
| action | [T,69] | PULSE 解码器输出的策略动作（PD 目标 = offset + scale·action） |
| pulse_z | [T,32] | PULSE 隐动作 |
| is_succ | bool | 跟踪是否成功（无提前终止） |
| fps | 30 | 标称帧率 |

刚体顺序：Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, R_Knee, R_Ankle, R_Toe, Torso, Spine, Chest, Neck, Head, L_Thorax, L_Shoulder, L_Elbow, L_Wrist, L_Hand, R_Thorax, R_Shoulder, R_Elbow, R_Wrist, R_Hand。
- **文本、划分、索引**：HumanML3D 官方 texts/（每句带词性标注 token）、train/val/test.txt（本地）、index.csv（官方仓库，14,616 行，指明每段的 AMASS 源文件与 20 fps 起止帧）。
- **本地副本的编号陷阱**：`/iridisfs/scratch/pf2m24/data/HumanML3D/HumanML3D` 是重新编号过的副本（其 zhuanhhh.py）：官方 007975、009707、011059 三段缺失，之后的编号依次前移；镜像片段 `M<id>` 变为 `id + 14613`（无 M 前缀）。index.csv、HF 物理数据用**官方** id；本地 texts/、new_joint_vecs/、划分文件用**本地** id。映射在 `hml_phys/hml_ids.py`（已用 texts.zip 中的原始文件核对）。本地 new_joints/ 目录另有一套编号，任何地方都不用。数据集 pkl 里同时存 `name`（本地 id）和 `official_id`。

### 1.2 匹配与覆盖
- index.csv 14,616 段中 12,150 段能在 HF 数据里找到源文件（9,408 个文件）。缺的 2,466 段：humanact12 1,191 段（不是 AMASS，无物理数据）；1,275 段的 AMASS 文件不在 HF（UniPhys 去掉了不可跟踪的动作，如推倒恢复、扶栏行走、地面动作）。
- 镜像片段（M 前缀）第一版不用。

### 1.3 时间对齐（`scripts/hml_phys/03_build_dataset.py`）
HumanML3D 片段第 k 帧对应源序列时间 `(trim + start_frame + k) / 20` 秒。`trim` 是官方 raw_pose_processing 对部分数据集做的头部裁剪：Eyes_Japan、MPI_HDM05 3 s；TotalCapture、MPI_Limits 1 s；Transitions 0.5 s。

PHC 把 AMASS 按 `skip = int(src_fps / 30)` 抽帧，所以 100 fps 的源（KIT、EKUT、部分 MPI_mosh）实际是 33.33 fps 但标 30（250 fps 的 SSM_synced 为 31.25）。处理办法：逐文件在 {30, 33.33, 31.25} 三个假设间判定——把该文件最长片段的物理轨迹按假设帧率转成 HumanML3D 263 维特征，取"根相对、朝向归一"的局部关节块（4:67）与官方 new_joint_vecs 比较，误差小者胜；再加长度可行性约束（假设所需帧数不能超过物理序列长度）。判定不了的（两者误差差距 <20%）用该数据集多数票。结果：KIT/EKUT 几乎全部 33.33，其余数据集 30，MPI_mosh 混合。每段记录 `fps_eff` 和 `time_stretch = fps_eff/30`（KIT 数据在仿真里比真实慢 11%，这是公开数据的固有缺陷，与用 PHC 数据的论文相同）。

对齐质量（修正编号后）：8,386 个成功文件全部对齐，局部关节误差中位数 0.042 m（PULSE 跟踪误差量级），没有 >0.25 m 的文件；1,524 个文件两假设差距 <20%（多为静态或短片段）用数据集多数票；382 个成功文件的物理序列比索引范围短（PHC 的 process_amass_db 会截断"坐下/腾空"结尾的序列），其片段按实际可用帧切，覆盖率 <90% 的片段带 `coverage` 标记（训练前可据此过滤）。

### 1.4 输出
`data/humanml3d_phys/hml_phys_{train,val,test}.pkl`：dict of lists，每段一项：name（本地 id）、official_id、split、source_file、hf_pkl、start/end_frame、head_trim_frames、fps_eff、time_stretch、phys_i0/i1、n_frames、coverage、file_align_err_m、texts（caption/tokens/f_tag/to_tag）、body_pos、dof_state、root_state、action、pulse_z。统计在 `build_stats.json`。

## 2. 评测协议（`hml_phys/evaluator.py`）
与 CLoSD 代码、Guo et al. 2022 一致：
1. 测试集条目按官方 `Text2MotionDatasetEval` 构造：长度 40 ≤ L < 200；带时间标签的句子切成独立子片段；每次重复随机选一句、长度按 4 取整并随机裁剪。测试集 4,646 条。
2. 生成侧：每条条目用其一句文本闭环生成一段（CLoSD 的 do_unique 做法），长度取 GT 长度。
3. 评估器：KV-Control 的 `text_mot_match`（Comp_v6_KLD005 的 meta 均值方差归一化）；R-Precision 按 32 一批；FID 用整个测试集真值嵌入统计；Diversity 300 对；MModality 可选（每句多条生成）；重复 20 次报均值 ±95% 置信区间。
4. 物理指标（`hml_phys/phys_metrics.py`，在 20 fps 的 22 关节上算）：Floating / Penetration / Foot-sliding 按 PhysDiff（CLoSD 实现，5 mm 容差，毫米）；Skating ratio 按 GMD（脚低于 5 cm 且速度 >0.5 m/s 的帧比例；分母是有效帧数，CLoSD 用补零后的 196 帧，故我们的值系统性偏大）；Jerk 为关节位置三阶差分均值（自定义，各论文定义不一），同时报 m/s³ 与 mm/frame³。两套口径：(a) 从 263 维特征恢复的关节——`process_file` 已把每段贴地，Penetration 恒为 0、Floating 是相对该段最低点，这是 CLoSD 的口径；(b) 仿真原始高度（`physics_raw`，y-up、真实地面 y=0），对仿真轨迹和物理真值都报这一套。
5. Duration = 到达目标长度前未摔倒的片段比例（摔倒 = 环境判定 done 的帧 ≤ 目标长度）。摔倒片段默认排除（CLoSD 做法），可选截断；两种口径都要连同 n_evaluated 一起报。
6. 与 CLoSD/UniPhys 的差异（明示）：每条条目 1 次 rollout（UniPhys 每句 5 次；MModality 需要 `06a --n_per_item 30` 另跑）；rollout 长度两种设置都跑：(i) 逐条目目标长度 ceil(1.5·L)+6；(ii) CLoSD 式固定 320 步；文本每条目随机选一句；起始不裁前缀（CLoSD 裁 16 步是因为它从数据前缀初始化，我们从站立姿态开始）。20 次重复只重抽评估批次，不重新生成，置信区间不含策略随机性。

### 2.1 验证结果
| 检查 | 结果 | 参考 |
|---|---|---|
| 真值 vs 真值（官方 263 维，3 次重复） | R@1 0.515 / R@2 0.706 / R@3 0.797 / MM-Dist 2.972 / Div 9.468 | Guo 2022：0.511 / 0.703 / 0.797 / 2.974 / 9.503 |
| 官方关节位置 → 我们的转换器 → 评估器 | R@1 0.515 / FID 0.003 / MM-Dist 2.970 | 应与上一行一致 ✓ |
| 真值运动学物理指标 | Floating 22.7 mm，Skating ratio 0.073，Jerk 5.37 mm/frame³ | — |

注意：本地 HumanML3D 目录下的 Mean.npy 和 new_joints/ 与官方不一致，不能用；转换器的参考骨架从 new_joint_vecs/000021 恢复。

### 2.1b 结果汇总（2026-09-15，测试集，Guo 评估器，20 次重复，±95% CI）
| 行 | 条目数 | R@1 | R@2 | R@3 | FID | MM-Dist | Div | Floating mm（仿真地面） | Skating ratio | Jerk mm/frame³ | Duration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 运动学真值（全集） | 4,646 | 0.515 | 0.706 | 0.799 | 0.002 | 2.97 | 9.47 | — | 0.073* | 5.37* | — |
| 运动学真值（有物理片段的子集，对全集） | 1,760 | 0.499 | 0.691 | 0.787 | 1.60 | 2.87 | 9.40 | — | — | — | — |
| **物理真值**（PULSE 跟踪，同子集） | 1,760 | **0.458** | 0.655 | 0.760 | 2.56 | 3.11 | 8.99 | 15.5 | 0.023 | 5.00 | — |
| UniPhys 官方权重，逐条目长度，排除摔倒 | 3,213 | 0.091 | 0.160 | 0.218 | 14.11 | 7.45 | 6.75 | 17.6 | 0.051 | 4.96 | 0.692 |
| UniPhys 官方权重，逐条目长度，截断摔倒 | 4,597 | 0.088 | 0.153 | 0.210 | 14.05 | 7.48 | 6.60 | 19.4 | 0.052 | 5.86 | 0.692 |
| UniPhys 官方权重，固定 318 步（10.6 s），随机选句，排除摔倒 | 2,447 | 0.094 | 0.164 | 0.222 | 14.12 | 7.40 | 6.82 | 17.3 | 0.052 | 4.75 | 0.527 |
| UniPhys 官方权重，固定 318 步，截断摔倒 | 4,577 | 0.088 | 0.156 | 0.213 | 13.94 | 7.49 | 6.63 | 19.5 | 0.054 | 5.86 | 0.527 |
| **UniPhys 官方权重，中性站姿起步，随机选句，逐条目长度，排除摔倒（最终口径）** | 3,303 | 0.093 | 0.166 | 0.225 | 14.71 | 7.42 | 6.75 | 17.5 | 0.051 | 4.86 | 0.711 |
| **我们 mc_v1 250k 步**（中性站姿，随机选句，Euler 32，CFG 3.5，排除摔倒） | 3,269 | 0.183 | 0.293 | 0.376 | 9.66 | 5.71 | 8.33 | 13.9 | 0.019 | 3.21 | 0.704 |
| 我们 mc_v1 250k 步，截断摔倒 | 4,642 | 0.184 | — | — | 7.85 | — | — | — | — | — | 0.704 |
| **我们 mc_v1 best_val（25k 步）**，排除摔倒 | 2,841 | **0.254** | 0.394 | 0.494 | 8.59 | 4.88 | 8.00 | 17.0 | 0.038 | 5.85 | 0.612 |
| 我们 mc_v1 best_val，截断摔倒 | 4,644 | 0.240 | — | — | 6.75 | — | — | — | — | — | 0.612 |
| 我们 mc_v1 50k 步，排除摔倒 | 3,232 | 0.230 | 0.357 | 0.445 | 10.68 | 5.28 | 7.98 | 16.6 | 0.031 | 5.57 | 0.696 |
| 我们 mc_v1 50k 步，截断摔倒 | 4,645 | 0.223 | — | — | 8.59 | — | — | — | — | — | 0.696 |
| 我们 mc_v1 100k 步，排除摔倒 | 3,282 | 0.221 | 0.342 | 0.428 | 8.98 | 5.33 | 8.23 | 15.4 | 0.025 | 4.52 | 0.708 |
| 我们 mc_v1 100k 步，截断摔倒 | 4,640 | 0.216 | — | — | 7.85 | — | — | — | — | — | 0.708 |
| **我们 mc_v2 50k 步（整段预测，81.6M）**，排除摔倒 | 1,139 | 0.230 | 0.367 | 0.462 | 9.11 | 5.17 | 7.50 | 15.9 | 0.032 | 5.45 | **0.246** |
| 我们 mc_v2 50k 步，截断摔倒 | 4,644 | 0.213 | — | — | 7.90 | — | — | — | — | — | 0.246 |
| UniPhys 论文数字（MIND 报 / SCRIPT 报，各自评估器） | — | 0.087 / 0.143 | 0.149 / — | 0.205 / — | 0.60 / 0.49 | 2.28 / — | 1.15 / — | 20.5 / — | — | — | — |
| MIND 报 Phys-GT（自家评估器） | — | 0.559 | 0.744 | 0.826 | 0.0001 | 1.28 | 1.24 | 15.6 | — | — | — |

*运动学真值的 Skating/Jerk 是 263 维恢复关节口径（无仿真地面）。物理指标的仿真地面口径下 Foot-sliding 恒为 0（刚体原点离地 ≥1.5 cm，PhysDiff 的 5 mm 接触判据不触发），故不列。
读法：R@1 与 MIND 报的 UniPhys 数字一致（0.091 vs 0.087），Floating 也一致（17.6 vs 20.5 mm，Phys-GT 15.5 vs 15.6 mm），说明流水线与他们的物理口径对上了；FID/MM-Dist/Diversity 的绝对值在不同评估器间不可比，只能在本表内部比较。

### 2.2 与论文数字的可比性
MIND 报的 Phys-GT（R@1 0.559，Diversity 1.24）和 SCRIPT 报的（R@1 0.651）不在 Guo 评估器的尺度上（Guo 评估器真值 Diversity ≈ 9.5），说明它们用了自己的评估器。CLoSD 与 UniPhys 的数字来自 Guo 评估器。因此：我们用 Guo 评估器，自报物理真值上界，并把 CLoSD/UniPhys 作为同尺度对照；MIND/SCRIPT 的数字只能作趋势参考。

## 3. 仿真转特征（`hml_phys/sim2hml.py`）
Isaac 24 刚体位置（MuJoCo 顺序、z 上、30 fps）→ 按 `MUJOCO_2_SMPL` 重排取前 22 关节 → 旋转到 y 上（正交、保手性）→ 线性插值到 20 fps → 官方 `process_file`（统一骨架到 000021、贴地、原点、首帧朝 Z+、脚触地阈值 0.002）→ 263 维。

## 4. 闭环评测（`hml_phys/uniphys_rollout.py`，`UniPhys/main_hml_rollout.py`）
- 环境：UniPhys 的 Isaac Gym + SMPL 人偶 + PULSE，控制 30 Hz（仿真 60 Hz），episode 320 步，起始为随机站立姿态，2 步均值隐动作预热（官方做法）。
- 摔倒判定：非脚部刚体接触地面且低于 0.15 m（PHC 默认）。
- 每条测试条目一次 rollout，目标长度 = ceil(GT 长度 × 1.5) + 6 帧；记录 body_pos、是否摔倒、摔倒步。
- `scripts/hml_phys/06a_prep_rollout_items.py` 生成条目；`07_eval_rollouts.py` 转换并评测。
