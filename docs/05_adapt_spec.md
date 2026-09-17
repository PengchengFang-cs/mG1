# ADAPT 第一阶段（扩散动作先验）实现规格 —— 草案 2026-09-13

来源：ADAPT arXiv 2609.00677 正文 + 附录；对照 UniPhys 代码。未确认项标 [?]。

## 1. 数据（对应 data/g1_rollouts/*.pkl）
- 来源：TextOp tracker 在 IsaacLab 里跟踪 GMR 重定向后的 AMASS，50 Hz；只保留 success 的 rollout
- 每帧记录：proprio 67 = v(3) ω(3) g(3) q(29) q̇(29)；上一步动作 a_{t-1}(29)；动作 a_t(29)
- 观测向量 96 = [v, 0.2 ω, g, q, 0.05 q̇, a_{t-1}]（论文给的缩放）；训练与部署时历史里的 v 置零
- 文本：BABEL frame_ann，按窗口时间重叠选标签（沿用 UniPhys 规则；去 "transition to "）[?] ADAPT 是否也随机选一条
- 片段：T = 20 帧，其中 H = 5 帧历史、15 帧未来；stride [?]（UniPhys 用 T/2）
- 归一化：逐维均值方差（obs 与 action 分开）[?]

## 2. 模型
- token 对齐（2026-09-13 决定）：token_j = [a_{j-1} (29) | o_j (96)]，即"导致 o_j 的动作 + o_j"，125 维。
  片段 20 个 token j = t-4 … t+15：前 5 个是历史（推理时全部已知，含当前 o_t），后 15 个是未来；
  第一个未来 token j=t+1 里的 a_t 就是当前要执行的动作。o_j 末尾 29 维 prev_action 与 a_{j-1} 重复，保留（与论文 96 维一致）。
  这与 UniPhys 缓冲区语义一致（action_buffer 存"刚执行的动作"，state_buffer 存"执行后的状态"）。
- 8 层因果 Transformer decoder，hidden 512，8 头，FFN [?]（UniPhys 2048），dropout [?]
- 文本：冻结 CLIP ViT-B/32，512 → Linear → memory（cross-attention），CFG dropout 0.1
- 噪声等级嵌入：逐 token（沿用 UniPhys），但训练时历史 5 帧保持干净（k=0），只对未来 15 帧采样 k
- 扩散：cosine 调度，训练 20 步，目标 v-prediction；推理 2 步 DDIM，CFG 2.5
- 优化：AdamW lr 1e-5，cyclic cosine，warmup 10k；batch [?]；步数 [?]

## 3. 闭环执行
- 缓冲区存最近 5 帧 (o, a)；每步：历史 5 帧 + 15 帧噪声 → 2 步去噪 → 取第 1 帧动作 a_t → 发给 PD（目标 = default + scale·a_t）→ 新观测写回缓冲区
- 频率 50 Hz；每步 2 次前向（CFG）
- 文本切换：不清缓冲区，换 embedding

## 4. 与 UniPhys 代码的映射
| UniPhys | ADAPT 改动 |
|---|---|
| babel.py 切片/选文本/标准化 | 换字段：proprio+action，T=20/H=5，去掉 cano/get_repr |
| df_base._generate_noise_levels | 历史 5 帧固定 k=0，未来随机 |
| diffusion.py pred_x0 + Min-SNR | 改 v-prediction，timesteps 20 |
| df_humanoid.policy | 去 scheduling matrix（全同步），2 步 DDIM，去 PULSE 解码，exec 1 帧 |
| interact | 换成 IsaacLab 环境 step（容器内） |

## 5. 决定记录（2026-09-15）
- 不做 VQ 码本几何版本（用户决定）。生成器采用 MotionCraft 形式：x0 预测 + 速度空间 loss + logit-normal + Euler。条件注入 adaLN-Zero。

## 6. 待确认 [?] 清单
1. 片段 stride、batch size、训练步数、FFN 宽度
2. token 内 action 与 obs 的时间对齐
3. 文本选择规则是否与 UniPhys 相同
4. 归一化统计的计算范围（只成功片段）
5. 评测协议：2048 rollouts、每 5–10 s 切换提示、成功 = 不摔；R-Precision 用 TMR
