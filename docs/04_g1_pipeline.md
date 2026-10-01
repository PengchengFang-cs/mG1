# 线 A：G1 数据管线（TextOp tracker in Isaac Lab 2.1.0）2026-09-13

## 环境
- 容器 sandbox：/iridisfs/scratch/pf2m24/containers/isaaclab_2.1.0.sandbox（NGC isaac-lab:2.1.0，Isaac Sim 4.5，Python 3.10）
- 启动：`bash scripts/isaaclab_exec.sh [--writable] [--offline] <cmd>`；默认开 SSH 隧道并把代理传进容器（Kit 需要下载 Nucleus 资产和注册表扩展）。HOME 映射到 /iridisfs/scratch/pf2m24/isaaclab_home（扩展缓存、Kit 数据都在那）。
- 容器 python：/workspace/isaaclab/_isaac_sim/python.sh。已装 textop_tracker、rsl_rl 2.3.3 分支（editable，指向 TextOp/），joblib、tqdm、pysocks。
- 踩坑：pip 会把 numpy/torch 换成 PyPI 版本 → 必须保持 numpy==1.26.0、torch==2.5.1+cu118、torchvision==0.20.1+cu118（Isaac Sim 自带的 ml_archive 是 cu118，两处不一致会报 torchvision::nms）。写模式 (--writable) 下绑定挂载点必须先在 sandbox 里 mkdir。SIF 在节点上挂不了（无 fusermount），只能用 sandbox。

## 数据
- 参考动作：TextOp/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps/{train,val}.pkl（list of dict：feat_p, frame_ann, motion{dof(T,23), root_trans_offset, root_rot(xyzw), fps 50}）
- scripts/make_motion_subset.py → name→motion dict + meta（frame_ann）
- TextOpTracker/scripts/pklpack_to_npz.py（容器内，需 Kit 做 FK）→ artifacts/<set>/<name>/motion.npz（fps, joint_pos/vel (T,29), body_*_w (T,30,...)）
- 23 DoF 数据在转换时腕关节补零到 29。

## 跟踪器评测（scripts/track_eval.py）
- 任务 Tracking-Flat-G1-ProjGravObs-MNMLP-v0，ckpt model_75000.pt，须传 anchor_body_name=pelvis, future_steps=5, actor/critic [2048,1024,512]
- 观测 431 = 参考未来 5 帧关节位置速度 290 + 锚点未来位置 15 + 朝向 30 + 投影重力 3 + 根线速度 3 + 角速度 3 + 关节位置 29 + 关节速度 29 + 上一步动作 29
- 动作 29 维，目标 = default_joint_pos + scale × action，scale = 0.25 × effort_limit / stiffness
- 50 Hz（dt 0.005 × decimation 4）；失败终止：锚点 z 偏差 > 0.25 m、任一关键刚体 z 偏差 > 0.25 m、锚点朝向偏差 > 0.8
- val 子集 20 段、从第 0 帧开始、无随机化：成功 42 / 失败 35（≈55%）。转身、走、推撑成功率高；举物、坐下失败多。
- 速度：20 env 约 150 env-steps/s（GPU 与他人训练共享）

## 录制（scripts/record_tracker_rollouts.py）
- 输出 pkl：rollouts[i] = {proprio (T,67), prev_action (T,29), action (T,29), joint_target (T,29), root_state (T,13), ref_t (T,), motion, feat_p, frame_ann, success}
- proprio 布局 = ADAPT 的 67 维本体感知；+ prev_action = 96 维

## 线 A 策略（2026-09-25 起，用户定：G1 真机落地，底座用我们自己的 DiT、不做部位化）

### 数据与窗口（hml_phys/g1_data.py）
- token 96 = proprio 67 + action 29，50 fps。**不做规范化**：`root_lin_vel_b / root_ang_vel_b / projected_gravity_b`
  本来就在体坐标系，投影重力已编码姿态，SMPL 侧那套"以哪一帧为原点"的问题在这里不存在。
- 视窗按**时长**换算，不是按帧数：SMPL 侧 30 fps 的 16 历史 / 4 生成 = 0.53 s / 0.133 s，G1 50 fps 对应 **H=27 / F=7**。
  意图窗 `L_INTENT=28`（0.56 s，且能被 VAE 的 4× 下采样整除）。
- 文本 = 与**生成段**重叠最多的 BABEL 帧标注，不是整段 caption。
- 划分：该数据集只有 train/val（CLAUDE.md §1），所以 **train 训练、val 评测**。

### 模型（hml_phys/g1_model.py）
- `G1FlowPolicy`：扁平 `Linear(96+1 -> 512)` + `FrameMotionTextDiT` + 零初始化 AdaLN 头，照 MotionCraft
  `RawFlowDiT` 的写法。**不做部位化**：那是 MoGeFlow 为每部位一套 VQ 码本才需要的，我们没有 VQ，
  而且 v3(扁平) 0.52 / v4(部位化) 0.54 本来就在噪声里；G1 是 29 关节，`_PART_BODIES` 也没法照搬。
- `G1IntentPolicy` = 上面 + MIND 意图机制（冻结意图 VAE + HIP + IIP，末层隐状态进联合注意力）。
  G1 的 `L_INTENT/4 = 7` 个隐帧，所以 `IntentDiT` 的 `n_lat` 已参数化（SMPL 的 4 帧 ckpt 仍能加载）。
- 保留下来的只有量过有用的三样：意图机制、**不要稀疏长历史**、`sentence_xattn` 文本通路。

### 未来状态的可见性（2026-09-26，第一对训练因此作废重跑）
token 里既有状态又有动作，所以未来行的 proprio 通道必须遮掉，否则直接泄漏答案——tracker 的动作
几乎就是它产生的那个状态的函数，看得见 s_{t+1} 的策略只要把动力学反过来解就行。第一版
`g1_train_policy.py` 写的是 `z = x.clone(); z[..., ACT] = z_act`，未来行的 proprio 是真值。

在 `g1_noint/step_100000.pt` 上量（val、2048 窗、t=0.5）：

| 策略能看到的未来 proprio | val action loss |
|---|---|
| 照训练时（全部未来行）   | 0.04429 |
| 只有第一帧               | 3.46114 |
| 一帧都没有               | 3.52485 |

**80 倍**的差距，等于闭环里能用的东西基本没学到。两个训练（g1_intent 66k/200k、g1_noint 118k/200k）
当场停掉，存到 `outputs/_archive_leak/`，加 `--obs_future_state` 后重跑。

默认取 `first` 而不是 `none`：一行 = (状态 s_t, 在这个状态下施加的动作 a_t)，H 行历史给出 s_0..s_H 和
a_0..a_{H-1}，要生成的是 a_H..a_{H+F-1}——所以**第一个生成行的状态在闭环里是拿得到的**（仿真刚返回的
那一帧），后面的拿不到。`none` 连这一帧也遮（和 SMPL 侧 `intent_flow.policy_input` 完全一致），
`all` 恢复泄漏行为，只留着复量这个差距用。

### 闭环评测（scripts/g1_eval_rollout.py + scripts/hml_phys/g1_eval_metrics.py）
物理这一半照搬 ADAPT 表 1，和 `scripts/g1_physical_protocol.py` 逐项一致：2048 次 rollout × 20 s
（1000 步 @50 Hz），提示词每 5–10 s 换一次，摔倒 = 除踝/腕外任一刚体低于 `--contact_z` 或根部倾角超 60°，
外加动作平滑度、切换平滑度、脚滑。

语义这一半 ADAPT 用 TMR，我们没有，改成：G1 连杆位置 → 22 个 SMPL 关节（`hml_phys/g1_to_smpl.py`，
14 个直接对应、8 个按机器人自身 pelvis→torso 长度合成）→ y-up → 20 fps → 官方 263 维 → 我们的 Guo 评估器。
因为一段 episode 中间会换提示词，录下来的轨迹按**片段**切开（一个提示词一段），每段算一个 item；
5–10 s 一段 = 20 fps 下 100–200 帧，正好落在 HumanML3D 自己的范围里，整段 20 s 则不是。

**这套数字能走多远**：关节映射是几何对应、不是 retarget，评估器又是在人体动作上训的，所以绝对值
**不能**和论文里的 HumanML3D 数字比（同 docs/06 §2.2b），也不能和我们 SMPL 侧的行比。能比的只有
走同一条路出来的 G1 行：我们两个策略互相比，以及
- **天花板** `--source tracker`：预训练 TextOp tracker 跟参考动作，提示词取它所跟那段动作的 BABEL 标注。
  这是"物理上正确、语义上确实对得上的动作，走这条路能拿几分"。
- **地板** `--source hold`：热身后一直保持姿态不动。

摔倒按 CLAUDE.md §2 一律**截断**、不剔除；整件事只算一次（§4）。

**热身**：策略第一次规划前要有 28 帧真实物理历史。保持复位姿态不行——复位姿态是参考动作的某一帧、
不是平衡站姿，实测 16/16 在 0.3 s 内就倒了。改成让 **tracker 跑 0.56 s** 再交接，这也更符合分布：
策略训练时看到的每一行历史都是 tracker 的动作。热身帧不记录、不计入指标。

文本特征：容器里没有 clip，所以 `scripts/hml_phys/g1_build_text_dict.py` 在容器外把 75 个提示词
（加空串，CFG 的无条件分支）编成 CLIP ViT-L/14 的 词 token + pooled + 长度，存 npz 给容器读。
评估器要 `word/POS`，提示词没有词性标注，就用 HumanML3D 语料里该词最常见的词性
（`data/g1_rollouts/hml_word_pos.json`），没见过的词退回 `unk/OTHER`——评估器本来就这么处理未登录词。

### 这套测量本身有多少信号（2026-09-26，在评任何策略之前先标定）
天花板不是一条线，是一把梯子。同一条路（G1 连杆 → 22 SMPL 关节 → 20 fps → 263 维 → Guo 评估器），
HumanML3D 测试集做 FID 参照，单次 rollout、单次计算：

| 行 | R@1 | R@2 | R@3 | FID | Duration |
|---|---|---|---|---|---|
| HumanML3D 真值（人体动作 + 人写的 caption，评估器的原生输入） | 0.524 | 0.711 | 0.799 | — | — |
| **G1 参考动作（完全没有物理）** 2048 段 / 2643 片段 | 0.249 | 0.382 | 0.474 | 4.11 | 1.000 |
| **G1 tracker（仿真里跟参考动作）** 2048 次 / 2168 片段 | 0.103 | 0.173 | 0.228 | 4.63 | 0.761 |
| *对照：tracker 的动作配打乱的标签* | 0.033 | 0.058 | 0.087 | 4.63 | — |
| *地板：热身后保持姿态* 2048 次 | — | — | — | — | 0.000 |

读法：0.524 → 0.249 是**这套测量自己的损耗**（G1 骨架不是人体骨架、8 个关节是合成的、BABEL 标签不是
HumanML3D 那种描述句），策略再好也翻不过去；0.249 → 0.103 是 **tracker 的物理误差**。我们的策略是
在模仿 tracker，所以现实的上界是 0.103 这一行。

**随机水平是量出来的，不是算出来的**：把 tracker 那 2168 个片段的标签随机打乱重评（`--shuffle_captions`，
动作一帧没动，只是配错了词），R@1 掉到 0.033 —— 和 32 选 1 的理论值 0.031 对得上。所以 tracker 那行是
随机的 3.1 倍，参考动作那行是 7.5 倍。信号不强但确实存在，报策略的时候这两行和这个对照必须摆在旁边。

**地板行（保持姿态）语义上打不出分**：热身交接后 2048 次全部摔倒，平均 0.89 s，1925 个片段里只有 8 个
够得上 2 s 的下限，连一个 R-precision 批次（32 个）都凑不满。这本身就是结论——它的 Duration 是 0.000，
语义一栏如实写"不适用"，而不是填个数。语义的地板由上面那个打乱标签的对照来定。

**caption 格式在评策略之前就定死了**，标定只在"参考动作"这一行上做过一次：Guo 的文本编码器吃的是
HumanML3D 的句子，不是裸标签。三个模板在参考动作行上的 R@1 分别是 裸标签 0.211 / `a person {}` 0.249 /
`a person is {}` 0.244 / `a man {}s` 0.130，取 **`a person {}`** 作为默认，之后不再动。FID 不受影响
（只看动作）。顺带把 18 个一个词都不在 GloVe 词表里的标签（abandon、apose、backpedal…）也救回来了。

天花板行的物理数字：no-fall 0.761、动作平滑度 0.663、脚滑 0.0736 m/s、平均摔倒时刻 6.49 s。
注意 tracker 也会摔：BABEL 里有坐、躺、爬起这类动作，高度判据会把它们判成摔倒，这对天花板和策略是
同一把尺子。切换平滑度在天花板行没有意义（tracker 不换提示词），记为 0。

### 评测脚手架的两个对齐 bug，以及它们各值多少（2026-09-26）
第一轮闭环评测（noint 4 万步：success 0.0068、R@1 0.0976、FID 21.76）**作废**，产物在
`outputs/g1_eval/_invalid_misaligned/`。两个 bug 都在「历史窗口怎么交给策略」上：

1. **差一帧**：数据集的一行是 (状态 s_t, 在这个状态下施加的 a_t)——`record_tracker_rollouts.py` 是在自己的
   `env.step` **之前**拍状态。rollout 写成了先 step 再拍，推进去的是 (s_{t+1}, a_t)。
2. **当前状态被抹零**：训练是 `--obs_future_state first`，第一个生成行的状态要给策略看；rollout 把整个
   未来段填零，包括那一行，等于在模型最依赖的位置喂了个数据集均值。

判据用 **shadow policy**（`--shadow_policy`）：让策略跟在 tracker 驱动的 rollout 后面，在**同一个状态**上
预测动作但不执行，和 tracker 真正做的比。状态始终在训练分布里，所以这个数字只反映管线，不掺累积误差。
指标是 NMSE = E‖pred − a‖² / E‖a − ā‖²，0 表示完全复现，1 表示不如直接输出平均动作。
`--shadow_bug` 可以只对 shadow 这一路把 bug 放回去，好给每个 bug 定价（128 env × 300 步，约 3400 个样本）：

| shadow_bug | NMSE |
|---|---|
| none（修好之后） | **0.0552** |
| pairing（只放回差一帧） | 0.0704 |
| rowh（只放回抹零当前状态） | 0.0851 |
| both（第一轮用的那个版本） | 0.1034 |

**结论要说准**：bug 确实存在，把单步误差翻了将近一倍（0.055 → 0.103），但 0.103 仍然意味着策略解释了
tracker 动作约 90% 的方差——**不足以解释 99.3% 的摔倒率**。所以闭环失败的主因是累积误差 / 分布漂移，
不是管线。修完重跑的意义在于拿到干净的数字，不是指望它翻盘。

顺带：NMSE 0.055 也说明这个策略的**单步模仿学得很好**，问题出在把它接成闭环之后。

### 第一组闭环结果（2026-09-26，修好脚手架之后；单次 rollout、单次计算）
2048 次 × 20 s，摔倒**截断**不剔除，HumanML3D 测试集做 FID 参照，caption 模板 `a person {}`。

| 行 | R@1 | R@2 | R@3 | FID | Duration |
|---|---|---|---|---|---|
| HumanML3D 真值（评估器原生输入） | 0.524 | 0.711 | 0.799 | — | — |
| G1 参考动作（无物理） | 0.249 | 0.382 | 0.474 | 4.11 | 1.000 |
| G1 tracker（仿真跟参考） | 0.103 | 0.173 | 0.228 | 4.63 | 0.761 |
| **我们·带意图（5 万步）** | **0.176** | **0.296** | **0.387** | **13.83** | 0.052 |
| **我们·不带意图（4 万步）** | 0.126 | 0.227 | 0.304 | 16.19 | **0.242** |
| 对照：tracker 动作配打乱的标签 | 0.033 | 0.058 | 0.087 | — | — |
| 地板：热身后保持姿态 | 不适用 | | | | 0.000 |

**先把话说清楚：这两行都不是能用的控制器。**

> **2026-10-01 作废声明**：本节原有一张以 **ADAPT 论文 0.984** 为锚点的对照表，已删除。
> ADAPT 按 CLAUDE.md §5 被永久否决（未开源、无人独立验证、我们严格按规格复现 stage 1 只有 0.044），
> 它的数字不得再作为对照、锚点或目标。本节下列结论凡依赖该锚点的，一并作废。
> 本页其余内容（数据管线、G1 token 布局、评测实现）仍然有效。
> 这些 G1 实验的结果文件与权重已删除（STATUS.md §4），数字无法复核，仅作当时做过什么的记录。

同一判据下的内部参照（非 ADAPT）：

| | Duration（20 s 不摔） |
|---|---|
| tracker（我们的教师，同一判据） | 0.761 |
| 我们·不带意图 | 0.242 |
| 我们·带意图 | 0.052 |

差一个数量级，**没有哪个别人的方法比我们差**。判据不背这个锅：拿**完全没有物理误差**的参考动作过同一个
摔倒判据，只有 48/2048 = **2.3%** 误判，所以 tracker 那 23.9% 是它真的跟丢了（和 val 子集 55% 成功率
一致），我们的 0.242 是在 0.761 这个真实上界之下，按教师归一化也只有 32%。

在这个前提下，两个消融的相对关系才有意义（是两个都站不住的东西之间的差别，不是"赢"）：意图版语义
全面更好（R@1 +0.050、FID −2.4），Duration 却只有 1/5。SMPL 侧意图机制是纯收益，G1 上不是。

**R@1 高过 tracker 那行不等于"超过天花板"**。三个数要一起读：我们两行的 FID 是 13.8/16.2，tracker 是
4.63；Diversity 是 6.08/5.58，tracker 是 8.23。合起来是**塌缩到"标签的典型动作"**——策略直接吃提示词，
产出刻板的"walk 的样子"，检索反而比真人动作容易，但离真实分布远得多，也站不住。tracker 那行是
"物理正确且确实对得上标签的动作能拿几分"，不是 R@1 的上界。

**摔倒率混淆的残留，以及它的控制**。截断口径仍有一个残留偏差：活得久的 episode 会贡献切换之后的片段，
那些片段比"从静止开始跟第一个提示词"难。两行的片段数正好相反（不带意图 3250 / 2048 = 1.59 每段，
带意图 2306 / 2048 = 1.13），所以偏差是**压低**不带意图那行。只取每个 episode 的**第一个片段**重算：

| 只算第一个片段 | R@1 | R@2 | R@3 | FID |
|---|---|---|---|---|
| 我们·带意图 | 0.183 | 0.307 | 0.393 | 13.68 |
| 我们·不带意图 | 0.148 | 0.263 | 0.344 | 15.01 |

差距从 0.050 收窄到 0.034，但方向不变——**意图确实带来了语义收益，不是摔倒率混淆造出来的**。
（tracker 那行不能这样控制：它的片段来自 BABEL 标注、不是提示词切换，"第一个片段"的含义不同，
只剩 855 段且系统性偏短，所以它仍按全部片段汇报。）

**根因是纯行为克隆（BC），不是超参**。shadow 检查说单步模仿已经很好（NMSE 0.055，解释了 tracker 动作
93% 的方差），一接成闭环就崩——策略只在 tracker 走过的状态上训练过，自己一旦偏出去就没有任何东西教它
怎么回来，误差每 20 ms 复利一次。ADAPT 那 0.984 是**带 DAgger**（拿策略自己跑出来的状态、用 tracker 的
动作当标签、回灌重训）得到的（**该对照已作废，见本节开头**），而这个 flow 策略**从来没见过自己的错误**：`data/g1_rollouts/dagger1~4`
是当初用 ADAPT 式**扩散策略**采的，不是它采的。所以下一步该补的是 on-policy 那一环，不是继续扫
CFG / 采样步数 / K —— 那些不会把 0.24 变成 0.9。

**意图版为什么更差**（待验证，不作为结论）：HIP 过拟合。它的 val 损失从 1.52 一路涨到 3.37，闭环里整体意图
是**采样**出来的（训练时读的是真值隐变量加噪），采出来的东西越不准，对策略的干扰越大——语义上还能
把方向带对，平衡上就被带偏。下一步应当先修 HIP，而不是继续加东西。
