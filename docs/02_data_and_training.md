# UniPhys 数据加工与训练（2026-09-12 核对代码）

## pickle → token（uniphys/datasets/humanoid/babel.py）
1. 只用成功序列、长度 > 32 的序列；窗口 33 帧，stride 16（多 1 帧算速度）
2. 文本：与窗口时间重叠的 BABEL 标签中随机选 1 条；去 "transition to "；"walk back"→"walk"；无标签则丢弃；查 text_embedding_dict_clip.pkl 得 CLIP 512 维
3. 规范化：片段第 0 帧根 xy 归零、髋连线定朝向、整段旋转到面朝 +y（cano_seq_smpl_or_smplx）
4. 状态 366 = 根位姿速度 15 + local_positions 72 + local_vel 72 + dof_pose_6d 138 + dof_vel 69（get_repr）
5. 动作 = z 32
6. 逐维标准化（train_data_stats.npy: Mean/Std 366, zMean/zStd 32）
7. token = [z | state] 398；batch = xs (B,32,398) + text_embedding (B,512)
训练集 98,851 片段。

## 训练一步（df_base.training_step → diffusion.forward）
- 噪声等级 k ~ Uniform{0..49}，对 (帧, 样本) 独立采样，无干净 context（_generate_noise_levels）
- 前向加噪 q_sample：cosine β，50 步；噪声 clamp 到 [-1,1]（df_humanoid clip_noise=1）
- 网络输入 x_k 与 k，因果 mask，文本 memory 以 0.1 概率置零（mask_cond，仅训练时）
- 目标 pred_x0：loss = MSE(pred, x0) 逐元素
- 权重 = min(SNR(k), 5)（Min-SNR），高噪声帧权重小
- 日志拆分 z_loss / root / local_pos / dof 等，只用于观察
- AdamW lr 1e-4 wd 1e-4，warmup 10k，cyclic cosine，grad clip 1.0，batch 512
- 官方 ckpt: 2.93M 步 / 15199 epoch，约 10 GPU 天

## 与标准动作扩散的差异
| | 标准 MDM 类 | UniPhys |
|---|---|---|
| 噪声等级 | 整段一个 t | 每帧独立 k |
| 注意力 | 双向 | 因果 |
| token | 姿势 | 姿势 + 动作 z |
| 目标 | x0 或 ε | x0 + Min-SNR 权重 |
| 条件 dropout | 有 | 有，0.1 |
