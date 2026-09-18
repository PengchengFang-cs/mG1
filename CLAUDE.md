# motion_rebot 项目死规定（永久，覆盖一切默认行为）

## 1. 禁用 val（2026-09-16，用户命令）
- **禁止对 val 划分做任何操作**：不 rollout、不评测、不筛选 ckpt、不扫超参、不汇报 val 数字、不生成 val 的 rollout 条目。
- 一切选择（ckpt、CFG、采样步数、任何超参）**只在 HumanML3D 测试集、完整协议下**做。
- 训练脚本内部的周期性损失曲线与「最低损失」存档一律用**测试集**（`--eval_split test`），不碰 val。
- 已有的 val 相关脚本产物（rollout_items_val_random.json、rollouts_mc_v1_*_val_*、eval_mc_v1_*_val_*、eval_mc_chain2.sh）视为作废，不再使用。
- 训练期的测试集损失是 teacher-forced 去噪损失，只用于挑 ckpt；最终汇报的数字仍必须来自测试集上的完整闭环协议。

## 2. 汇报格式
- 任何数字都必须放进完整对照表：运动学真值、物理真值、UniPhys（自跑 + 论文）、PDP、CLoSD、Kimodo++、MIND、SCRIPT、我们，所有指标（R@1/2/3、FID、MM-Dist、Diversity、Floating、Jerk、Duration），注明划分与评估器。表在 docs/06_hml_phys_protocol.md §2.1b，每次更新后整表汇报，禁止只报自己的数字。

## 3. 卡与实验
- 不经用户明确批准不启动任何训练；评测只用用户指定的卡；不占用户其他项目的卡。
- scancel 永久禁用；停止训练只允许杀本项目的进程。
