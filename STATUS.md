# 现状（唯一权威入口）

> **本文件是「当前进度」和「当前最好结果」的唯一依据。**
> `NOTES.md` 是只往后追加的流水账，里面大量配置已被淘汰，**不得作为现状依据**。
> 每次有新结果或方向变化，**覆盖本文件对应段落**，不要在末尾追加历史。
> 最近更新：2026-10-01

## 1. 核心目标（永远不变）

**端到端生成 Unitree G1 的机器人动作。** 创新就是把这件事本身做好 ——
该领域现有论文基本都不开源，可信度不成立。

## 2. 当前线：FRoM-W1（arXiv 2601.12799）

唯一代码 + 权重 + 数据齐全的外部工作，所以选它做对照。
复现范围**只限机器人侧 H-ACT**，生成侧 H-GPT 已封存（CLAUDE.md §6）。

### 2.1 已完成：H-ACT 复现（2026-10-01）

用他们发布的 G1 student 权重、他们的指标代码、他们的 0.5 m 成功判据，
在我们按他们管线自建的 424 条 AMASS→G1-21dof 参考上。单次 rollout、单次计算。

| 指标 | G1-Full（未过滤） | G1-Clean（过滤） |
|---|---|---|
| **Success Rate ↑** | 0.8821 | **0.9033** |
| mpjpe_g ↓ (mm) | 253.53 | **238.02** |
| mpjpe_l ↓ (mm) | 187.48 | **177.88** |
| mpjpe_pa ↓ (mm) | 102.50 | **94.63** |
| accel_dist ↓ | **6.09** | 7.89 |
| vel_dist ↓ | **10.17** | 11.52 |

**对上了他们报的两条定性结论**：过滤数据更好；G1 的 SR 在 80–90%。
**对不齐绝对数值**，两个原因都在发布侧：跟踪结果只有柱状图无数值表；
评测动作集 `100_100fps_dup20.pkl` 未发布未描述。
Figure 9 的训练递进复现不了 —— teacher 未发布。

细节见 `docs/09_fromw1_spec.md` §7。

### 2.2 可复用的基础设施（这是复现的最大副产品）

全部落盘在 `scripts/`，与 FRoM-W1 解耦，换成我们自己的策略可直接用：

| 脚本 | 作用 |
|---|---|
| `fromw1_amass_download.sh` / `fromw1_amass_extract.sh` | AMASS 19 个子集下载解压（14096 条，25 GB，在 `data/amass/`）|
| `fromw1_h2h_setup.sh` / `activate_h2h.sh` | `h2h` 环境（py3.8 + Isaac Gym + legged_gym + phc + rsl_rl）|
| `fromw1_g1_21dof_config.py` | G1 21dof 重定向配置，从 MJCF 解析并断言关节顺序 |
| `fromw1_amass_to_g1_21dof.py` | SMPL → G1 21dof 动作库（梯度拟合）|
| `fromw1_relayout_21to29.py` | 21dof → 29dof 布局 + **世界系自检**（已验证能拦住坐标系错误）|
| `fromw1_eval_g1_policy.py` | 跑 G1 跟踪策略并出 MPJPE/SR，绕开未发布的 teacher |
| `fromw1_eval_chain.sh` | 上述串成一条链，任一级失败即中止 |

## 3. 方法侧成果：线 1（SMPL 仿真角色）

连续动作流模型（相对 VQ 离散码本的对照胜出，NOTES 2026-09-23）。
**形态是 PHC/UniPhys 的仿真 SMPL 角色（69 维动作、24 关节），不是 G1，上不了真机。**
价值在于方法与消融结论，不在于可部署性。

### 当前最好（截断摔倒口径，HumanML3D 测试集全集 4646，单次 rollout、单次计算）

| 配置 | R@1 | R@2 | R@3 | FID | Duration | 摔倒率 |
|---|---|---|---|---|---|---|
| **无稀疏历史 20万步 + 最优旋钮** | **0.4117** | 0.5866 | 0.6942 | **2.129** | 0.914 | 8.6% |
| 基线 10万 + 最优旋钮 | 0.4021 | 0.5866 | 0.6916 | 2.270 | 0.901 | — |
| 基线 20万 + 最优旋钮 | 0.3894 | 0.5733 | 0.6886 | 2.369 | 0.929 | 7.1% |

最优旋钮 = 采样 10 步 + CFG 2.5 + 意图 s=0.5 + K=2。
真值参照：R@1 0.5244 / R@2 0.7106 / R@3 0.7994。

**不要再引用这些数**（它们是早期扫描里被淘汰的配置，曾被我误当成最好）：
R@1 0.410 / FID 3.51 / Duration 0.760 / 摔倒 24.2%。

### 已定的消融结论
- 连续动作优于 VQ 离散码本（物理闭环控制任务上）
- 稀疏长历史（SCRIPT 式指数采样 + 有符号帧偏移 + L_max 回溯）**删掉是对的**：
  同步数同旋钮干净对照下 R@1 +0.0223、FID −0.240
- 窗口定为 20 帧（16 密集 + 4 未来），与 MIND 的历史结构一致

## 4. 已删除 / 作废，不得再引用

| 东西 | 处置 | 理由 |
|---|---|---|
| ADAPT（arXiv 2609.00677）全部产物 | **已删除**（2026-10-01） | CLAUDE.md §5 永久否决。未开源、无人独立验证、我们严格按规格复现 stage 1 只有 0.044 |
| ADAPT 时代的 G1 实验（`outputs/g1_eval/`、`g1_intent`、`g1_noint` 日志） | **已删除**（2026-10-01） | 评测协议借自 ADAPT，锚点按 §5 作废；20 万步权重已在清磁盘时丢失，数字无法复核 |
| HumanML3D 物理线的 val 相关产物 | 作废 | CLAUDE.md §1：三划分数据集一律忽略 val |
| FRoM-W1 生成侧（H-GPT）产物 | 封存保留不推进 | CLAUDE.md §6 |

## 4b. 改名与保留（2026-10-01）

删 ADAPT 时带出的连锁问题，一并处理掉，**不要再被名字误导**：

| 原名 | 现名 | 说明 |
|---|---|---|
| `scripts/adapt_eval_protocol.py` | **`scripts/g1_physical_protocol.py`** | 我们自己的 G1 物理 rollout 评测台（接触判据摔倒、式 S10/S11、冻结参考）。名字里原来带 adapt，内容不是 ADAPT 的 |
| `scripts/hml_phys/adapt_prompt_pool.py` | **`scripts/hml_phys/g1_prompt_pool.py`** | |
| `data/adapt_{prompt_pool_130,eval_prompts,motion_whitelist_*}.txt` | **`data/g1_*.txt`** | |
| — | **`hml_phys/babel_labels.py`**（新建） | `clean_label` / `labels_overlapping` 从被删的 `adapt/data.py` 取回。纯标签处理，我们的 G1 代码在用 |

**保留的 G1 端到端生成代码**（§1 的核心目标，目前唯一的实现）：
`hml_phys/g1_{data,model,to_smpl}.py`、`hml_phys/babel_labels.py`、
`scripts/g1_eval_rollout.py`、`scripts/hml_phys/g1_*.py`（8 个）。

**两条已失效的代码路径**，误用会立刻报错并指向替代方案：
`g1_physical_protocol.py --source policy` 与 `record_tracker_rollouts.py --policy_ckpt`
都依赖已删的 ADAPT `DiffusionPolicy`。我们自己的策略用 `scripts/g1_eval_rollout.py` 评。

## 5. 下一步（待定）

FRoM-W1 机器人侧复现已收尾。**端到端 G1 生成这件事本身怎么往下做，方向未定，等用户拍板。**
在此之前不启动任何训练（CLAUDE.md §3）。
