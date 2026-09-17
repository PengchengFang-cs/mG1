# 文本驱动物理人形控制：方向调研（2026-09-15，来源为各 arXiv 页面，未逐一读全文）

## 两条线
A. 仿真角色线（SMPL 人物，Isaac Gym，PHC 造数据，HumanML3D 协议）：UniPhys(ICCV25) → PDP/CLoSD/MaskedMimic → MIND(2605.26006) / SCRIPT(2605.22894, SIGGRAPH Asia 26)
B. 真机线（Unitree G1，IsaacLab，PPO 跟踪器造数据）：LangWBC(RSS25) → TextOp(2602, 两段式, 有代码) → SENTINEL(2511.19236) / ADAPT(2609.00677) / ECHO / RoboForge

## 数据（全部是"动捕 → 重定向 → RL 跟踪器在仿真里执行 → 录 (状态,动作,文本)"）
| 工作 | 动作来源 | 规模 | 跟踪器 |
|---|---|---|---|
| LangWBC | HumanML3D 子集 | "几十段" | 自训 PPO |
| TextOp | AMASS 40.7h + 私有 3h + 合成 31h；生成器 83k 段文本对 | ~75h | 自训 PPO（公开） |
| SENTINEL | AMASS/HumanML3D 12,422 段 ×20 次 DR rollout | ~200k 轨迹 ≈ 1 亿状态动作对 | 自训 PPO |
| ADAPT | AMASS + BABEL | 未报告 | TextOp 跟踪器 |
| SCRIPT | HumanML3D + MotionMillion | 550k 轨迹 ≈ 1200h | PHC |
| MIND | HumanML3D | 未报规模 | PHC |
| 我们现在 | TextOp 公开的 8,266 段 | ≈ 400 万帧 ≈ 23h（+DAgger） | TextOp 跟踪器 |

## 评测
- 仿真角色线：HumanML3D 测试集，R-Precision/FID/MM-Dist（TMR 或 T2M 评估器）+ 物理指标（floating、jerk、foot skate）+ 完成率。可比性最好，数据和 baseline 公开。
- 真机线：成功率（不摔）+ R-Precision（TMR）+ 平滑/脚滑 + 真机演示。协议各家自定：LangWBC 15 条未见指令分三档；SENTINEL HumanML3D 测试集，成功 99.45% vs LangWBC 81.78%；ADAPT 2048×20s、130 指令、5–10s 切换，纯先验 0.804 / 完整 0.984。
- HumanTracker(2608.13555)：跟踪基准（153h，MuJoCo 统一入口），不含文本条件。

## 代码/数据可得性
公开：UniPhys、TextOp（跟踪器+数据子集）、PHC/PULSE、BeyondMimic、MotionMillion/HumanML3D。未公开：LangWBC、SENTINEL、ADAPT（Coming Soon）、SCRIPT（页面无链接）、MIND（将发布）。

## 结论
- ADAPT 是真机线最新工作之一，但不是唯一权威：SENTINEL 更早、数据大 25 倍、成功率 99%，两者都无代码；ADAPT 未报数据规模，复现无法对齐。
- 若要"可比较"，仿真角色线（HumanML3D 协议、公开数据与 baseline）是唯一能严格对齐的战场，且与用户的 MoGeFlow/MotionCraft 直接衔接；但 SCRIPT 已在此做了 JAST-DiT + flow matching + RL 后训练 + 规模化。
