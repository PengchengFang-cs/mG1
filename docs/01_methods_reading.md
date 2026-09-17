# 仿真角色线文本驱动物理控制：方法与评测阅读总结（2026-09-15）

## 一、方法对照（SMPL 人物，Isaac Gym / MuJoCo，公开数据）
| | UniPhys (ICCV25) | PDP (SIGGRAPH Asia 24) | CLoSD (ICLR25) | MIND (2605.26006) | SCRIPT (2605.22894, SIGGRAPH Asia 26) |
|---|---|---|---|---|---|
| 范式 | 端到端：状态+动作联合扩散（Diffusion Forcing） | 端到端扩散策略 | 两段：DiP 规划器(MDM) + RL 跟踪器闭环 | 端到端：Intent VAE + 多尺度意图预测 + 动作 DiT | 端到端：JAST-DiT 三流(动作/状态/文本)联合注意力 + RL 后训练 |
| 造数据 | PULSE 跟踪 AMASS | PHC 跟踪 KIT（文本用 HumanML3D） | HumanML3D 训 DiP；跟踪器 AMASS | PHC 跟踪 HumanML3D，只留成功 | PHC 跟踪 HumanML3D + MotionMillion，前后过滤 |
| 规模 | 15.7h (BABEL) | 未报 | HumanML3D | 未报（≈HumanML3D） | 550k 轨迹 ≈1200h |
| 状态/动作 | 366/398 维；32 维 PULSE 隐动作 | 358 状态 / 69 动作 | 跟踪器动作 | 358 / 69 | 358 / 69 |
| 历史 | 4 帧 | 4 帧 | 规划 40 帧 | 16 帧 | 稠密近期 + 指数稀疏远期（式 6） |
| 生成 | DDPM x0，5 步 DDIM，CFG | 扩散策略 | MDM | 扩散 DiT（6 块×768），CFG 3.5 | flow matching，adaLN-Zero，QK-Norm，0.2B–1.2B |
| 后训练 | 无 | 无 | 无 | 无 | PPO：物理跟踪奖励 + 轨迹级文本对比奖励 + BC 锚 |
| 训练 | 10 GPU 天 A100 | 4×A100 32h | 600k 步 | 80k 步 bs448，8×4090 12h | 未报 |
| 代码 | 有 | 无 | 有 | 将发布 | 页面无链接 |

## 二、评测协议（HumanML3D 测试集，Guo et al. 2022 评估器）
- 指标：R-Precision Top-1/2/3、FID、MM-Dist、Diversity、MModality（text_mot_match 评估器，输入 263 维 HumanML3D 特征）；物理指标：Floating（脚离地）、Jerk（加速度变化）、Duration（完成率）、Penetration/Skating（PhysDiff 定义）；鲁棒性：推力后 2 s 内摔倒率（SCRIPT，0–800 N × 0.2 s）。
- 初始化：MIND 统一中性站姿；其余未明说。
- 仿真轨迹 → 263 维特征：各文各文未写清；通行做法是取仿真关节位置（22 关节）按 HumanML3D 流程重算特征，20 fps。
- 数字（Top-1 / FID / Floating mm）：MIND 0.468 / 0.118 / 17.1；SCRIPT 0.435 / 0.164 / 17.6；Kimodo++ 0.386 / 0.661 / 33.7；CLoSD 0.17–0.37 / 0.37–0.73；UniPhys 0.09–0.14 / 0.49–1.15；PDP 0.03–0.21 / 0.97–1.54；Phys-GT 0.559 / 0.0001 / 15.6。同一 baseline 在两篇里数字不同 → 各家自跑、协议细节不一。
- 真机线（仅作参考）：SENTINEL 在 HumanML3D 测试集报成功率 99.45%、R@1 0.582；ADAPT 自定协议 0.804/0.984。

## 三、数据资产核对
- HumanML3D：本地已有（263 维特征、文本、划分）；AMASS 映射 index.csv 需从 HumanML3D 仓库下载。
- UniPhys 公开的 PULSE 跟踪全 AMASS 状态动作对（HF yan0116/SMPL_Humanoid_offline_dataset/amass_state-action-pairs，19 个子数据集，每个约 0.3–0.4 GB，含 is_succ）→ 可按 index.csv 切出 HumanML3D 片段并配文本，得到与 PDP/MIND/SCRIPT 同类的"HumanML3D 物理数据集"，无需自己跑跟踪器。
- MotionMillion：HF InternRobotics/MotionMillion 需申请，CC BY-NC-SA，308 GB，272 维 SMPL-X 表示；HumanML3D/BABEL/AIST 子集不直接发布。要用于物理控制还需自己重定向并跟踪（PHC/PULSE）→ 大工程，第二阶段再议。
- 评估器：KV-Control/checkpoints/t2m/text_mot_match 已在本地。
- 仿真：UniPhys 的 Isaac Gym + SMPL 人偶 + PULSE 环境已装好可用。
