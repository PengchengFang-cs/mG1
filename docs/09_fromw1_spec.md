# FRoM-W1 复现规格（2026-09-30）

论文 arXiv 2601.12799（OpenMOSS / 复旦 NLP），代码 github.com/OpenMOSS/FRoM-W1，权重与数据在
HuggingFace `OpenMOSS-Team/FRoM-W1` 与 `OpenMOSS-Team/FRoM-W1-Datasets`。

**为什么做它**：这是目前**唯一**在「语言指令驱动人形全身控制」上完整开源的工作（代码 + 权重 + 数据 + 部署框架）。
SENTINEL、LangWBC、ADAPT、SafeFlow、MIND、SCRIPT 全部没有代码。我们需要一个能独立跑起来的外部基线。

**它是什么范式**：两阶段运动学范式，不是端到端 text→action。
```
H-GPT   文本 → 623 维 SMPL-X 动作表示   （VQ-VAE 离散化 + Llama-3.1-8B LoRA + 思维链）
  ↓
retarget  623 维 → 52 关节 → SMPL-X → 机器人关节（G1 / H1，梯度拟合，最多 1000 步 Adam）
  ↓
跟踪      RL 策略（基于 Human2Humanoid，有 G1/H1 预训练权重）
  ↓
RoboJuDo  sim2sim / sim2real
```
对应 ADAPT 表里 `Offline TextOp` (0.522) / `DART` (0.764) 那一类。**他们自己不报在线切换指令的闭环成功率**，
所以我们用自己的 Table-1 协议去跑它（docs/05 §5），拿到那个位置上第一个可独立复现的数字。

## 1. 发布里的缺口与我们的处理

| 问题 | 事实 | 处理 |
|---|---|---|
| **adapter 自带的 `tokenizer.json` 是错的** | 四个 LoRA 的 `tokenizer.json` 都只声明 515 个 `<motion_id_*>`（512 码本），但权重 `lm_head` 是 `[130307, 4096]`，130307 − 128256 = **2051 = 2048 + 3**，对应唯一发布的 VQ-VAE（`codebook (2048, 1024)`）。他们自己的 `lora_merge.py` 和 `load_pretrained_lora_and_merge` 都从 adapter 目录读 tokenizer，**都会踩这个坑** | `scripts/fromw1_merge_lora.py` 按 2048 自建 tokenizer 再合并，并断言合并后词表为 130307。`load_pretrained_vae` 的 `strict=True` 独立复核了码本尺寸 |
| **demo 从不加载 VQ-VAE** | `load_pretrained_vae` 在 demo 路径里没有任何调用点，解码器保持随机初始化 | `scripts/fromw1_gen.py` 显式加载。**这条最危险**：文本与思维链输出完全正常，只有动作是噪声。判据是脚的运动——随机解码器下双脚高度恒为 0.12 m、根部以完全恒定速率平移；正确加载后脚会抬到 0.34 m 再落下 |
| **`deps/` 全套缺失** | GloVe 词向量、T2M 评估器权重、指令模板都不在发布里 | GloVe 用本地 `Umdd/KV-Control/glove`；评估器用他们 HF 上的 `eval/`；**指令模板从 MotionGPT 取**（这套代码派生自它），`class`/`input`/`output` 与占位符逐项核对一致 |
| submodule 用 SSH 地址 | `git@github.com:humanoidintelligence/...`，无密钥拉不动 | 改 HTTPS 克隆到同路径 |
| `main.py` 无条件调手部 retarget | 需要 MANO 模型；且 `SMPLX_OUTPUT_PATH = ""`，脚本按原样跑不起来 | G1 是 29 自由度、末端到 wrist_yaw，没有灵巧手。`scripts/fromw1_retarget.py` 只调 `body_retarget.process_data` |
| `pos2smpl` 注释与实现不符 | 注释写 `np.ndarray`，实现调 `torch.zeros_like` | 传张量 |
| `from_pretrained` 未指定 dtype | 按 fp32 materialise，8B 参数 32 GB | 加 `torch_dtype="auto"`（config 里本来就是 bfloat16）。见 `external/VENDOR_PATCHES.md` |

## 2. 环境
- 新建 conda 环境的 pip **走不了 SOCKS 隧道**（装 PySocks 后内置 urllib3 冲突，报 `PoolKey.__new__() got an
  unexpected keyword argument`）。克隆项目已有的 `uniphys`（pip 24.2 正常），顺带白得 torch / numpy /
  pytorch_lightning / smplx / chumpy / trimesh
- 补装 `transformers==4.45.1` / `peft` / `accelerate` / `rich` / `easydict` / `imageio` / `shapely` /
  `spacy==3.7.5`（必须预编译 wheel）/ `bert_score` / `sacrebleu` / `rouge_score` / `nltk`
- **pip 会把 numpy 升到 1.24 导致 chumpy 挂掉**（`np.bool` 被移除），每次装完都要钉回 `numpy==1.23.5`
- SMPL / SMPL-X 用本地已有的；**MANO 不需要**

## 3. 生成结果的两个不可控之处（影响基线公平性，必须声明）
130 条提示词（`data/g1_prompt_pool_130.txt`，套 `a person {}` 模板，因为 H-GPT 训练用的是描述句不是裸标签）：
- 帧数中位 292（9.7 s @30fps），最小 44（1.5 s），最大 1000
- **23/130 撞到 1000 帧生成上限**（模型没有自然停止）→ 截到 10 s
- **37/130 短于协议最短片段 5 s** → 拼接时循环补齐，逐段在 meta 里标 `looped`

## 4. 接到我们协议上
retarget 输出 `{body_names, root_trans_offset, pose_aa, dof(T,29), root_rot(XYZW), fps:30}`，
**字段与 TextOp 参考动作格式一致**，所以：
1. `scripts/fromw1_build_schedule.py`：按协议采样提示词序列（每 5–10 s 切换，共 20 s），拼接片段，
   边界按偏航角 + XY 对齐使其从上一段末尾续上；高度保持生成值（retarget 已做落地校正）。
   **不做混合平滑**——切换处的姿态跳变是两阶段方法的真实属性，抹平它会美化基线。
   坐标约定：retarget 在置换前的轴 1 上做落地校正、再 `[:, [2,0,1]]`，所以输出中 XY 水平、**索引 2 是高度**（z-up）
2. `pklpack_to_npz.py`（他们的脚本，容器内）：30 → 50 fps 重采样 + Kit 前向运动学 → `artifacts/fromw1/*/motion.npz`
3. `g1_physical_protocol.py --source tracker`：TextOp tracker 跟踪，接触判据摔倒、式 S10/S11、只算未摔的 rollout

**策略模式与 tracker 模式在参考动作上是相反的**：策略不看命令项，所以要冻住它以免参考放完时瞬移；
tracker 必须让参考推进，否则它会一直跟第一帧。调度构建成正好 1000 帧，两种模式下都不会触发重采样；
并加了断言，任何参考动作短于 `--steps` 就直接报错。

## 5. 对表目标（2026-09-30 从 arXiv 2601.12799 原文抄录，此后不再凭印象引用）

### 5.1 生成侧 —— Table 3，HumanML3D-X 基准，Motion-X 训练数据

这是我们唯一能对的那张表。**H-GPT++ 是我们的目标行**（发布的 `hgpt/motionx/lora/llama-3.1-cot`
+ 唯一发布的 `VQVAE_MotionX_2Kx1K`）。

| Model | FID ↓ | R Top-1 ↑ | R Top-2 ↑ | R Top-3 ↑ | MM Dist ↓ | DIV → | MModality ↑ |
|---|---|---|---|---|---|---|---|
| Real | 0.000±0.000 | 0.393±0.005 | 0.573±0.005 | 0.677±0.004 | 3.862±0.009 | 9.811±0.096 | — |
| H-GPT w.o. CoT | 0.229±0.029 | 0.333±0.007 | 0.490±0.003 | 0.588±0.003 | 4.455±0.033 | 9.674±0.021 | 2.754±0.335 |
| H-GPT | 0.255±0.008 | 0.332±0.004 | 0.481±0.004 | 0.573±0.001 | 4.513±0.019 | 9.382±0.182 | 3.256±0.465 |
| H-GPT++ w.o. CoT | 0.337±0.011 | 0.323±0.005 | 0.482±0.005 | 0.581±0.004 | 4.494±0.054 | 9.240±0.104 | 2.664±0.115 |
| **H-GPT++** | **0.312±0.054** | **0.327±0.004** | **0.486±0.010** | **0.583±0.012** | **4.494±0.066** | **9.411±0.062** | **3.153±0.241** |

Table 1（HumanML3D-X 训练数据）的 H-GPT 行与 Table 3 同值，**且复现不了**：它对应
`hgpt/humanml3d-x/lora`（512 码本），配对的 512 VQ-VAE 未发布。只有 motionx 的 2Kx1K 发布了。

### 5.2 跟踪侧 —— **没有表**

**论文的跟踪结果只以柱状图给出（Figure 7、Figure 9），正文与附录都没有数值表。**
Figure 7(a) 在 AMASS 上比较 filtered / unfiltered 数据，G1 的 SR 落在 80–90% 区间、H1 在 40–50%；
Figure 9 比较 from-scratch → pretrain → finetune 的递进。**具体数值只能从图上读，精度约 ±5%。**

这意味着跟踪侧**做不到「严格对表」**，只能做到「落不落在他们图示的区间内」。这是他们的发布问题，
不是我们的选择。汇报时必须写明「原文仅有柱状图，无数值表」。

### 5.3 两处协议差异，必须在汇报里声明

1. **他们报 ± 是多次重复的结果**（Guo 协议惯例 20 次）；本项目 CLAUDE.md §4 永久规定单次。
   我们的数字是**单次 rollout、单次计算**，不写 ±。
2. **MModality 本质上需要对同一文本重复采样**（`METRIC.MM_NUM_REPEATS`），与 §4 直接冲突。
   除非用户另行指示，**这一列不做**，汇报时写「未做，与项目单次评测规定冲突」。

### 5.4 缺口清单（决定哪些格子永远填不上）

| 缺什么 | 卡住什么 | 能否解决 |
|---|---|---|
| Motion-X 授权（623 维 GT 特征） | FID、DIV、以及 Real 参照行 | 用户申请中 |
| humanml3d-x 的 512 码本 VQ-VAE | Table 1 的 H-GPT 行 | 未发布，无解 |
| H-ACT teacher 策略 | 复现「pretrain → finetune」递进 | README 自写 "Teacher (TBD)"，未发布 |
| 跟踪侧数值表 | 严格对表 | 原文只有图，无解 |

## 6. 机器人侧（H-ACT）复现规格 —— 2026-09-30 从代码与 env_cfg 实测抄录

生成侧已封存（CLAUDE.md §6）。以下是机器人侧的全部事实。

### 6.1 他们发布的 G1 策略是什么

`external/fromw1_weights/hact/g1/` 下两个，都是 **OmniH2O student**（LSTM，actor `[512,256,128]`，
`init_noise_std 0.001`，`distill: true`，`obs_v: v-teleop-extend-max-full`，22 个 teleop 关键点）：

| 目录 | 对应 Figure 7(a) |
|---|---|
| `25_12_11_18-16-37_OmniH2O_STUDENT` | G1-Full（未过滤数据） |
| `25_12_11_18-18-10_OmniH2O_STUDENT_FILTER` | G1-Clean（过滤数据） |

**两个都在本地** → Figure 7(a) 的 filtered vs unfiltered 对比**可以复现**。

### 6.2 机器人是 21 自由度，不是 29

`env_cfg.json` 写死：`file: resources/robots/g1/urdf/g1_21dof.urdf`、`dof_num: 21`、
`xml_file: .../g1_21dof.xml`。资产在 `human2humanoid/legged_gym/resources/robots/g1/` 里，齐全。

**注意与我们自己的线不同**：TextOp 那条是 29 dof，`BABEL-AMASS-ROBOT-23dof` 是 23 dof。
三者互不通用，不能拿我们已有的重定向数据代替。

### 6.3 指标定义（`phc/phc/smpllib/smpl_eval.py:100-160`，单位全是 mm）

| 指标 | 定义 |
|---|---|
| `mpjpe_g` | 全局 MPJPE，不做根对齐，`norm(pred-gt).mean()*1000` |
| `mpjpe` | 根相对 MPJPE，先减掉根关节再算 |
| `pa_mpjpe` | Procrustes 对齐后的 MPJPE |
| `vel_dist` | `compute_error_vel` 的均值 ×1000 |
| `accel_dist` | `compute_error_accel` 的均值 ×1000 |
| **`succ`** | **`not fail_safe and percent == 1`** —— 整条参考动作跑完且未触发终止才算成功 |

终止条件在 env_cfg：`terminate_by_ref_motion_distance: true`、`max_ref_motion_distance: 1.5`（米）。
即关键点到参考的距离超过 1.5 m 就判失败。**这是 SR 的全部定义，不需要我们自己发明判据。**

评测入口是 `legged_gym/legged_gym/scripts/play_hydra.py`。

### 6.4 机器人侧的缺口

| 缺什么 | 事实 | 能否解决 |
|---|---|---|
| **重定向后的 AMASS 动作库** | `env_cfg` 要 `resources/motions/g1/amass_all_21dof.pkl`；`resources/motions/` 目录**根本不存在**。human2humanoid README 原话：「We also provide preprocessed training datasets for Unitree G1 and H1 **(TODO: Add download links)**」—— **没发布** | 要自己建：原始 AMASS → `retarget/body_retarget/grad_fit_robot.py`（通用，G1 走 `robot_config.py`） |
| **原始 AMASS** | 集群上**没有**。UniPhys 的 `sample_data/*.pkl` 只是筛选表与 betas，不是动作数据；TextOp 的 `BABEL-AMASS-ROBOT-23dof` 是 23 dof 且走的是别家重定向管线，**不能替代** | 需在 amass.is.tue.mpg.de 注册下载 |
| **teacher 策略** | README 自写 "Teacher (TBD)" | **未发布，无解** → Figure 9 的 from-scratch→pretrain→finetune 递进**复现不了**，只能评 student |
| **数值表** | 只有 Figure 7 / Figure 9 柱状图 | **无解**，只能判断是否落在 SR 80–90% 区间 |

### 6.5 结论：能复现到什么程度

**能做**：用他们发布的两个 G1 student，在我们自建的 AMASS→G1-21dof 动作库上，
按他们的指标定义算 `mpjpe_g / mpjpe / pa_mpjpe / vel_dist / accel_dist / succ`，
复现 Figure 7(a) 的 **filtered vs unfiltered 对比**，并检查 SR 是否落在 80–90%。

**做不了**：Figure 9 的训练递进（缺 teacher）；与他们数值逐位对齐（无表）。

**唯一阻塞项**：原始 AMASS 下载。

### 6.6 补充（2026-10-01）：G1 的 AMASS 重定向脚本不在发布里，但零件齐了

把两个子模块都翻完后，`amass_all_21dof.pkl` 这一步的真实情况：

**他们发布了的零件**

| 零件 | 位置 | 说明 |
|---|---|---|
| 通用 AMASS→机器人梯度拟合 | `retarget/body_retarget/grad_fit_robot.py` | 有 `load_amass_data()`、`process_data(robot="G1")`，**本来就吃 AMASS** |
| **G1 的 SMPL 体型拟合** | `assets/beta/shape_optimized_g1.pkl` | 已发布。体型是肢长比例，21/23/29 dof 是同一台机器人锁不同关节，**这个文件可直接复用** |
| 通用 MJCF 正运动学 | `phc/phc/utils/torch_robot_humanoid_batch.py` | `Humanoid_Batch(cfg: RobotConfig)`，解析任意 MJCF |
| `g1_21dof.xml` | `human2humanoid/legged_gym/resources/robots/g1/xml/` | 策略 env 用的就是它 |
| G1 的 legged_gym 配置 | `legged_gym/legged_gym/cfg/cfg_g1/` | 含 `config_eval.yaml`、`config_play_student.yaml`、`motion/*` |

**没发布、必须我们写的**

1. **21 dof 的 G1 重定向配置**。`retarget/body_retarget/robot_config.py` 里 `G1Config` 是 **29 dof**
   （`xml_file = assets/robot/g1/g1_29dof.xml`，`ROBOT_ROTATION_AXIS` 29 条，含 waist_roll/pitch 与
   两侧 3 自由度腕）。29 = 21 + waist_roll + waist_pitch + 腕×3×2。**21 dof 是 29 dof 的严格子集。**
   retarget 的 assets 里也只有 `g1.xml` / `g1_29dof.xml`，**没有 21dof 的 xml**（但 legged_gym 里有）。
2. **构建动作库的驱动脚本**。`human2humanoid/scripts/data_process/grad_fit_h1.py` 是 **H1 专用**
   （19 dof 写死、H1 关节名、`torch_h1_humanoid_batch`、`data/h1/shape_optimized_v1.pkl`），
   **没有 G1 版本**。

**一处无法消除的歧义（必须在汇报里写明）**

他们没说 `amass_all_21dof.pkl` 是怎么做的，两种可能：
- (a) 直接在 21 dof 上做梯度拟合
- (b) 在 29 dof 上拟合再丢掉 8 个关节

**我们选 (a)**。理由：策略只控 21 个关节；若拟合时放开腕和腰去降误差、再把这些自由度丢掉，
得到的 21 dof 参考轨迹并非该约束下的最优解，会人为抬高跟踪误差。
**这是我们的选择，不是他们的规定** —— 汇报时如实标注。

### 6.7 他们的数据管线原样跑不起来的四处（2026-10-01 实测）

要造 `amass_all_21dof.pkl`，他们发布的三个相关脚本**每一个都有阻断性问题**：

| 文件 | 问题 | 我们怎么处理 |
|---|---|---|
| `retarget/body_retarget/grad_fit_robot.py:100` | `load_amass_data` 把 `"fps": 30` **硬编码**，真实帧率被注释掉（`# framerate`）。于是 `process_data` 里 `skip = fps // 30` 恒为 1，**不做降采样**。AMASS 多数是 120 Hz，直接喂进去得到 4 倍慢动作 | 按 `process_amass_db.py:175` 的 `skip = int(真实帧率 / 30)` 重采样 |
| `retarget/body_retarget/robot_config.py` | `G1Config.FIX_BASE_HEIGHT = True`、`FIX_BASE_HEIGHT_VALUE = 0.75` —— **根高度被钉死成常数 0.75 m**。部署路径合适，但会把下蹲、坐下、起跳全部抹平 | 改用 `grad_fit_h1.py:219` 的逐片落地：最低身体点抬到 0.08 m |
| `scripts/data_process/process_amass_db.py` | ① 循环体内留着 `ipdb.set_trace()`（第 203 行附近），原样跑会停住；② 导入 `uhc.*`（`transform_utils`、`smpl_parser`、`flags`）与 `fix_height_smpl_vanilla`，**`uhc` 不在依赖里也没装** | 把重采样与遮挡过滤的逻辑**转写**进我们的脚本，不调用它 |
| `scripts/data_process/grad_fit_h1.py` | 末尾是 `ipdb.set_trace()` 然后 `joblib.load(...)`，**从不保存 `data_dump`**；且 19 dof / H1 关节名 / H1 的 FK 与体型文件全写死 | 不用。它是调试残骸，不是生产路径 |

另外跳过了 `fix_height_smpl_vanilla`（在未安装的 `uhc` 里）：它给 SMPL 的 trans 加一个常量，
拟合目标与共享的根同时平移，而机器人在之后还要重新落地，所以对拟合结果只差这一个常量。

**坐标约定对比（容易搞错，记一下）**

| | 高度轴 | 落地 | 置换 |
|---|---|---|---|
| retarget（部署） | 置换前索引 1 | `FIX_BASE_HEIGHT` 钉死 0.75 | 输出做 `[:, [2,0,1]]` |
| 动作库（我们要的） | 索引 2（机器人自身 z-up） | 最低点抬到 0.08 m | **不置换** |

### 6.8 我们写的两个脚本

| 脚本 | 作用 |
|---|---|
| `scripts/fromw1_g1_21dof_config.py` | 21 dof 的 G1 重定向配置，**从他们的 `g1_21dof.xml` 解析出来**，不手抄。断言解析出的受驱动关节顺序与策略 `env_cfg.json` 的 dof 顺序逐项一致 |
| `scripts/fromw1_amass_to_g1_21dof.py` | 造动作库。梯度拟合的损失、优化器、调度**逐行转写自他们的 `grad_fit_robot.process_data`**；FK 用他们的 `robot.py`；体型用他们发布的 `shape_optimized_g1.pkl` |

**21 dof 的结构处理**（因为锁关节导致 body 数 ≠ joint 数）：
- `g1_21dof.xml` 有 **24 个 body、21 个 joint** —— `waist_roll_link` 与 `torso_link` 被锁成固定链接
- `fk_batch` 按 **body** 索引，所以轴表必须保持 body 对齐：两个无关节 body 给零轴，`axis * dof` 恒为 0，等价锁死
- MJCF 给的 `joints_range` 只有 21 行而 dof 变量有 23 行，重排成 body 对齐、锁死行填 `[0,0]`，`clamp_` 才能用
- `WRIST_PICK = []` → 他们代码本来就有 `if wrist_pick_idx == []: loss_rot = 0` 的分支，21 dof 没有腕关节可约束，自然走这条

### 6.9 决定性缺口（2026-10-01）：他们的评测动作集没发布、也没描述

把 `cfg_g1/` 全部读完后，G1 侧被引用过的动作文件**只有三个**，一个都没发布：

| 文件 | 出处 | 用途 |
|---|---|---|
| `resources/motions/g1/amass_all_21dof.pkl` | 发布权重的 `env_cfg.json` | **训练**用 |
| `resources/motions/g1/100_100fps_dup20.pkl` | `cfg_g1/config_eval.yaml` | **评测**用 |
| `resources/motions/g1/motion_data.pkl` | `cfg_g1/config_play_student.yaml` | 单条可视化 |

**`100_100fps_dup20.pkl` 是他们算 Figure 7(a) 那些数的动作集。** 文件名能看出是 100 条动作、
源 100 fps、每条复制 20 遍（`num_envs: 406`），但**是哪 100 条、怎么挑的，论文和代码里一个字都没有**。

于是机器人侧的「严格对表」有两重不可能：
1. **没有数值表**（§5.2，只有柱状图）
2. **没有相同的评测动作集** —— 我们只能在自建的 AMASS 子集上评，动作不同，绝对值天然不可比

**实际能做到的**（这就是机器人侧复现的上限，汇报时照此写）：
- 用他们发布的两个 G1 student（Full / Clean）
- 在我们按他们管线自建的 AMASS→G1-21dof 动作库上
- 用他们的指标代码（`phc/phc/smpllib/smpl_eval.py`）算 `mpjpe_g / mpjpe / pa_mpjpe / vel_dist / accel_dist / succ`
- 可对照的只有两条**定性**结论：(a) Clean 是否优于 Full（Figure 7(a) 的主张）；(b) SR 是否落在 80–90% 区间

**不能做**：与他们的绝对数值逐位对齐。

### 6.10 评测配置核对（好消息）

`cfg_g1/config_eval.yaml` 与我们下载的 student 完全对得上，不需要改配置：

| 项 | config_eval.yaml | 发布权重的 env_cfg.json |
|---|---|---|
| `load_run` | `25_12_11_18-16-37_OmniH2O_STUDENT` | 就是这个目录 |
| `teleop_obs_version` | `v-teleop-extend-vr-max-nolinvel` | 同 |
| `num_observations` | 1821 | 1821 |
| `num_privileged_obs` | 1904 | 1904 |
| `short_history_length` | 25 | 25 |
| `extend_head` | False | false |
| `max_ref_motion_distance` | 1.5 | 1.5 |

（`env_cfg.json` 里另有 `obs_v: v-teleop-extend-max-full` / 993 / 1076 / history 5 —— 那是
**teacher** 的 distill 配置，只在 DAgger 训练时用，评 student 不涉及。）

一处要留意：`config_eval.yaml` 的 `train.dagger.load_run_dagger` 指向
`25_12_05_23-30-46_OmniH2O_TEACHER_G1` 且 `dagger_only: True`，而 teacher **没发布**。
若评测路径会去实例化它，需要把这段关掉——待实跑验证。
cat >> docs/09_fromw1_spec.md <<'EOF'

### 6.11 更正：评测用的成功阈值是 0.5 m，不是 1.5 m

§6.3 写「`max_ref_motion_distance: 1.5` 就是 SR 的全部定义」**是错的**。
`rsl_rl/rsl_rl/runners/on_policy_runner.py:276` 的 `eval()` 里把它**硬改成 0.5**：

```python
self.env.cfg.env.test = True
self.env.cfg.env.im_eval = True
self.env.begin_seq_motion_samples()
self.env.cfg.asset.termination_scales.max_ref_motion_distance = 0.5   # 覆盖配置里的 1.5
```

所以：**1.5 m 是训练期的终止阈值，0.5 m 才是评测判成功的阈值**（这也是 PHC / H2O 的惯例值）。
评测结束后再还原成配置值。

`eval()` 的完整流程：
1. 置 `test=True`、`im_eval=True`，阈值改 0.5
2. `begin_seq_motion_samples()` —— 按顺序遍历动作，一个 env 跟一条
3. 跑到所有动作覆盖完
4. `compute_metrics_lite` 算两套：**全部动作**与**仅成功的动作**
5. 打印 `Success Rate`、`All: ...`、`Succ: ...`、以及失败的动作 key 列表

注意第 4 步：他们同时报「全部」与「仅成功」两套。这与本项目 CLAUDE.md §2 的
「截断摔倒 vs 排除摔倒」是同一个问题 —— **「仅成功」那套摔得越多数字越好看**。
汇报时两套都给，并以「全部」为主。

### 6.12 评测入口：不能直接用 config_eval.yaml

`config_eval.yaml` 走的是 `train_hydra.py`（`max_iterations: 1` + `has_eval: True` + `eval_interval: 1`），
但 `learn()` 里顺序是：

```python
if has_eval and it > 0 and it % eval_interval == 0:
    eval_info = self.eval()                 # 先评测
...
self.alg.update(..., dagger_only=self.dagger_only)   # 再训练 -- 这里要 teacher
```

`config_eval.yaml` 的 `dagger_only: True` 且 `load_run_dagger` 指向未发布的
`25_12_05_23-30-46_OmniH2O_TEACHER_G1`。评测在 `alg.update` 之前，所以指标能打印出来、
之后才会因缺 teacher 报错 —— 但这样很脏。

**我们的做法**：写一个只构建 runner 再直接调 `eval()` 的驱动，完全不进 `learn()`，
从而彻底绕开未发布的 teacher。
EOF
echo ok; sed -n '/终止条件在 env_cfg/,+2p' docs/09_fromw1_spec.md | head -3

### 6.13 动作库格式的权威定义：`cfg_g1/phc/phc_base.yaml`（2026-10-01）

这个文件才是 env 读动作库的权威规格，**它推翻了 §6.6 里我自己做的那个选择**。

```yaml
xml_file : "resources/robots/g1/xml/g1_29dof.xml"      # FK 用 29dof，不是 21dof
ROBOT_ROTATION_AXIS : [ ...29 行... ]
JOINT_NUM : 21
picked_joint : [0,1,...,12, 15,16,17,18, 22,23,24,25]  # 从 29 个关节里挑 21 个
picked_link  : [0,...,13, 16,17,18,19, 23,24,25,26, 30,31,32]   # 25 个跟踪关键点
Extend:
  extend_link_name : ["left_hand_link", "right_hand_link", "head_link"]
  extend_parent_idx : [19, 26, 0]        # 注意 head 挂在 pelvis(0)，不是 torso
```

**结论**：动作库的 `pose_aa` 必须是 **29dof 布局，33 行** = 1 根 + 29 非根 body + 3 extend。
`fk_batch` 里 `pose[..., :len(self._parents), :]` 会按 33 个 parent 逐个索引，给 27 行就静默切短，
最后炸成 `IndexError: index 0 is out of bounds for dimension 2 with size 0` —— 报错位置离病因很远。

`picked_joint` 挑的 21 个正是我们独立推出的那组（跳过 waist_roll 13、waist_pitch 14、
左腕 19-21、右腕 26-28）。所以 **§6.6 的歧义不再是歧义**：
他们的库是 29dof 布局，策略靠 `picked_joint` 取 21 个。

**不需要重拟合。** 已核对：21dof MJCF 与 29dof MJCF **共有的 24 个 body，`pos` 与 `quat` 逐位相同**，
因此经过那 21 个受驱动关节的运动链在两个文件里是同一个模型，拟合出的角度可原样搬运，只是行位置变。
被锁的 8 个关节（腰滚、腰俯、两侧腕三自由度）保持单位旋转。

重映射脚本 `scripts/fromw1_relayout_21to29.py`，**按 body 名对齐，不用硬编码索引表** ——
右臂在两种布局间**整体偏移 3 行**（`right_shoulder_pitch_link` 在 21dof 是非根第 19，29dof 是第 22），
硬编码极易差三。脚本里断言重映射后那 21 个受驱动角与原值逐位相等。

**拟合目标仍用 FRoM-W1 retarget 模块的约定**（head 挂 torso + 0.45 m），不用 phc_base 的
（head 挂 pelvis + 0.45 m）。理由：后者是刚固连在根上的点，关节怎么动都改不了它，
对拟合既无梯度也无信息；而 21dof 只有一个 waist_yaw，用挂在 torso 的头部点去约束它是有意义的。
**env 侧的关键点定义照他们的 `picked_link` 原样用，不动。** 这一处差异汇报时注明。

### 6.14 评测侧要修的五处（2026-10-01，全部实跑踩出来）

| # | 问题 | 处理 |
|---|---|---|
| 1 | `legged_gym` 声明依赖 `rich`，requirements 里没有 | 跳板机下 wheel，计算节点离线装 |
| 2 | `pytorch3d` 未声明未安装，但仓库**自己 vendored 了同一模块**，只有 `legged_gym/utils/transform.py` 一个文件去 import 外部包 | 改那一行指向 `phc.utils.pytorch3d_transforms`（VENDOR_PATCHES §2） |
| 3 | `asset_teleop.yaml` 的资产路径是**纯相对路径**无 `{LEGGED_GYM_ROOT_DIR}` 占位符；仓库有两个 resources 树，`human2humanoid/resources` 下**只有 23dof**。从 README 说的 `human2humanoid/` 跑会报 "Failed to parse URDF" 并返回 **0 自由度**资产 | cwd 改为 `legged_gym/`，并加资产存在性断言 |
| 4 | `poselib` 既不自带也不声明，且两个子模块 import 的模块路径还不同（`poselib.skeleton` vs `poselib.poselib.skeleton`）。借用的 UniPhys 那份 `from_mjcf` 读 `pos` 无默认值，而 `g1_21dof.xml`（与 Unitree 官方 `g1_29dof.xml` 一样）有两个 body 省略了它 | 为 h2h 单独 vendored 一份加默认值的，`PYTHONPATH` 遮蔽，UniPhys 环境不动（VENDOR_PATCHES §3） |
| 5 | 动作库必须是 29dof 布局（本节） | `fromw1_relayout_21to29.py` |

加上 §6.7 数据侧的四处，**为了把他们的评测跑通一共要修九处**。

---

## 7. 机器人侧复现结果（2026-10-01）

### 7.1 怎么得到的

| 项 | 取值 |
|---|---|
| 策略 | 他们发布的两个 G1 student，`model_50000.pt`，未做任何微调 |
| 参考动作 | 424 条 AMASS 片段，163575 帧，90.9 分钟 |
| 动作来源 | 原始 AMASS（SMPL+H，gender-specific），19 个子集按 `process_amass_db.py:242-261` 的划分取 train+test（vald 按项目 §1 排除），seed 0 随机抽 500，经他们自己的遮挡过滤表剩 424 |
| 重定向 | 拟合逻辑转写自他们的 `grad_fit_robot.process_data`，FK 用他们的 `robot.py`，体型用他们发布的 `shape_optimized_g1.pkl` |
| 拟合误差 | 均值 0.0507 m，中位 0.0449，p95 0.0904，最大 0.1694 |
| 评测 | 他们的 `OnPolicyRunner.eval()`，指标 `phc.smpllib.smpl_eval.compute_metrics_lite` |
| 成功判据 | 关键点距参考 > 0.5 m 即失败（`eval()` 里硬编码，覆盖配置的 1.5 m） |
| 重复次数 | **单次 rollout、单次计算**（项目 CLAUDE.md §4） |

**产物自检**（用 env 自己的 FK 与 `phc_base` 配置回读，`fromw1_relayout_21to29.py`）：

```
root z            : min 0.177  median 0.795  max 0.844     # 0.795 ≈ URDF 骨盆高 0.793
head above pelvis : median 0.448   frac>0.2 0.998          # ≈ head extend 的 0.45 m
lowest foot       : median 0.080   frac<0.15 0.981         # 落地目标就是 0.08
```

### 7.2 结果

单位均为 mm，口径为**全部动作**（下节说明为何不以「仅成功」为主）。

| 指标 | G1-Full（未过滤） | G1-Clean（过滤） | 差 |
|---|---|---|---|
| **Success Rate ↑** | **0.8821** | **0.9033** | **+0.0212** |
| mpjpe_g ↓ | 253.53 | **238.02** | −15.51 |
| mpjpe_l ↓ | 187.48 | **177.88** | −9.60 |
| mpjpe_pa ↓ | 102.50 | **94.63** | −7.87 |
| accel_dist ↓ | **6.09** | 7.89 | +1.80 |
| vel_dist ↓ | **10.17** | 11.52 | +1.35 |

「仅成功」口径（他们的 `eval_info` 只在这个口径下给 accel / vel / pa）：

| 指标 | G1-Full | G1-Clean |
|---|---|---|
| mpjpe_g | 248.69 | 236.68 |
| mpjpe_l | 193.51 | 184.05 |
| mpjpe_pa | 103.34 | 95.22 |
| accel_dist | 2.59 | 2.69 |
| vel_dist | 7.22 | 7.11 |

### 7.3 对照他们报的内容

他们的跟踪结果**只有柱状图（Figure 7、Figure 9），无数值表**，且**评测动作集
`100_100fps_dup20.pkl` 未发布、未描述**。因此可对照的只有定性结论：

| 他们的主张 | 我们的结果 | 判定 |
|---|---|---|
| Figure 7(a)：过滤数据训出的策略跟踪更好 | SR +2.1 点，三个 MPJPE 全部更低 | **复现** |
| Figure 7(a)：G1 的 SR 在 80–90% | 88.2% / 90.3% | **落在区间**（读图精度约 ±5%） |
| Figure 9：from-scratch → pretrain → finetune 的递进 | 无法评 —— teacher 未发布（README 自写 "Teacher (TBD)"） | **不可复现** |

**accel / vel 在「全部」口径下与上述方向相反**（Clean 更差），但在「仅成功」口径下两者几乎相等
（2.59/2.69、7.22/7.11）。证据不足以支持任何结论，如实列出，不作判断。

### 7.4 口径说明：为何以「全部」为主

「仅成功」的 `mpjpe_l` 反而**比「全部」更差**（Full 193.5 vs 187.5）。原因是失败片段在终止处截断，
只统计到前面那一段，而开头机器人是按参考摆放的、跟得最准，于是把「全部」的均值拉低。

这与本项目 CLAUDE.md §2 记录的「截断摔倒 vs 排除摔倒」是同一个机制，在他们的代码里就是
`metrics_all` 与 `metrics_succ`。按 §2 的纪律，**以「全部」为主口径**，「仅成功」仅附列。

### 7.5 这次复现的边界（汇报时必须一并写出）

**可以说**：用他们发布的 G1 student 权重、他们的指标代码、他们的成功判据，在我们按他们管线
自建的 AMASS→G1-21dof 参考上，**复现了 Figure 7(a) 的两条定性结论**。

**不能说**：与他们的数值逐位对齐。原因全在发布侧 —— 无数值表、评测动作集未发布。

**没做**：Figure 9 的训练递进（缺 teacher）；真机部署。
