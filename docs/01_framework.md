# UniPhys 大框架与模型架构（2026-09-11，基于 wuyan01/UniPhys 代码核对）

## 1. 三个阶段

```
阶段 A  离线造数据（作者已做好，我们直接用 babel_train.pkl）
  AMASS 动捕 ──重定向──> SMPL 仿真人物参考动作
        ──PULSE 跟踪器在 Isaac Gym 里模仿──> 每帧记录 (root_state, dof_state, body_pos, action, z)
        ──BABEL 逐帧文本──> frame_labels
  产物：babel_{train,val}.pkl，一段序列 = 一个 AMASS 文件

阶段 B  训练扩散策略（Diffusion Forcing）
  pkl ──切成 32 帧片段(stride 16) + 选一条文本──> 每帧 398 维 token = [z 32 | state 366]
      ──每帧独立采样噪声等级 k──> 12 层因果 Transformer decoder 预测 x0
      ──CLIP 文本嵌入作 cross-attention memory，10% 概率置零(CFG)

阶段 C  闭环执行（interact / policy）
  仿真返回真实状态 ──缓冲区最近 4 帧(干净) + 未来 8 帧(噪声)──> 5 步 DDIM 去噪
      ──取未来 8 帧 z──> PULSE 解码器 z→69 维关节目标──> PD 控制 ──> 仿真 ──> 新状态回写缓冲区
```

## 2. 每帧 token 的 398 维

| 段 | 维度 | 内容 | 来源 |
|---|---|---|---|
| z | 32 | PULSE 隐动作（归一化） | pkl z_all |
| root_trans | 3 | 根位置（规范化坐标系） | root_state[0:3] |
| root_rot_6d | 6 | 根旋转 6D | root_state[3:7] 四元数 → 旋转矩阵前两列 |
| root_trans_vel | 3 | 根线速度 | root_state[7:10] |
| root_rot_vel | 3 | 根角速度 | root_state[10:13] |
| local_positions | 72 | 24 个关节相对根、朝向 y+ 的位置 | body_pos |
| local_vel | 72 | 上述位置的帧间差分 | body_pos |
| dof_pose_6d | 138 | 23 个驱动关节旋转 6D | dof_state[...,0] 轴角 |
| dof_vel | 69 | 23 个关节角速度 | dof_state[...,1] |

state = 15 + 72 + 72 + 138 + 69 = 366；yaml 里的 351 被 main.py:507-516 覆盖。

## 3. 网络（models/transformer.py TransformerDecoder）

```
输入 x: (T=32, B, 398)      噪声等级 k: (T, B) 整数 0..50
  k ──SinusoidalPosEmb(64)──┐
  x ───────concat───────────┴─> init_mlp: Linear(462→768) ReLU Linear(768→768)
  + 帧位置编码 t_embed(768)
  文本 CLIP(512) ──Linear(512→768)──> memory（训练时 10% 置零）
  12 × nn.TransformerDecoderLayer(d=768, 8 头, ffn 2048, dropout 0.1, 因果 mask)
  out: Linear(768→398)  ──> 预测 x0（objective = pred_x0）
参数量 247M（含 CLIP 不计）。ckpt: global_step 2.93M, epoch 15199。
```

## 4. 关键超参（config/diffusion_forcing/algorithm/df_humanoid.yaml）

- n_frames 32, context_frames 4, chunk_size 8, exec_step 8
- timesteps 50, sampling_timesteps 5 (DDIM, eta 0), beta cosine, stabilization_level 3
- guidance 2.5, cond_mask_prob 0.1, clip_noise 1
- lr 1e-4, batch 512, warmup 10k, cyclic cosine, grad clip 1.0

## 5. 训练数据规模

- babel_train.pkl: 切片后 98,851 个 32 帧片段
- babel_val.pkl: 1815 段序列，1635 段跟踪成功，共 731k 帧 ≈ 6.8 小时
- 同一段序列可能对应多个文本片段，切片时随机选一条与该 32 帧重叠的标签
