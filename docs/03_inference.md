# UniPhys 闭环推理（df_humanoid.policy / interact，diffusion.ddim_sample_step）2026-09-13 核对

## 循环
```
reset → 发 2 步零动作填缓冲区
loop 直到 episode 结束:
  历史 = 缓冲区最近 4 帧 (root_state, dof_state, body_pos, z) → 规范化 → 366 维 + 32 维 → 标准化
  未来 = 8 帧纯噪声 (clamp ±1)
  x = [历史 4 | 未来 8]  (12, B, 398)
  按 scheduling matrix 逐行做 DDIM：每行两次前向（无条件 + 有条件），CFG: out = u + 2.5 (c − u)
  取未来 8 帧的 z，反标准化
  for i in 0..7:  a = PULSE.dec_action(z_i, 当前 obs) → env_step → 真实状态与 z_i 写入缓冲区
```

## 噪声等级
- 调度等级 0..5 映射到真实步 real_steps = [-1, 9, 19, 29, 39, 49]（50 步中取 5 个）
- 历史帧等级 0 → 真实 -1 → 在 ddim_sample_step 中改为 stabilization_level−1 = 2：x_hist ← √ᾱ₂·x_hist（不加随机噪声，只缩放），并告诉模型它处在等级 2
- 未来帧：autoregressive 矩阵，uncertainty_scale = sampling_timesteps = 5，41 行：
  第 t 帧从第 5t 行开始降，每行降 1 级，第 5t+5 行到 0。即帧内严格顺序：第 0 帧完全去噪后第 1 帧才开始。
  每行只更新等级下降的帧，其余保持原值（mask）。
- DDIM eta = 0，确定性；最后一步 alpha_next = 1 直接输出 x̂₀

## 关键超参
context_frames 4，chunk_size 8，exec_step 8，sampling_timesteps 5，guidance 2.5，stabilization_level 3，clip_noise 1
每 8 帧约 40 行 × 2 次前向 = 80 次 247M 模型前向

## 文本切换
缓冲区不清空；下一块生成时换文本嵌入即可。平滑性来自历史帧条件 + 因果注意力。
