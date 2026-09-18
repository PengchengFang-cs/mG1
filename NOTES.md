# motion_rebot — ADAPT 复现工作区

目标：复现 ADAPT (arXiv 2609.00677) 的文字条件扩散控制器。第一阶段先跑通 UniPhys。

## 目录
- `UniPhys/`            官方仓库 (wuyan01/UniPhys, Apache-2.0)，shallow clone 2026-09-10
- `scripts/setup_uniphys_env.sh`  在计算节点建 `uniphys` conda 环境（自动开 SSH 隧道）
- `logs/`               环境安装与下载日志

## 环境
- conda env: `uniphys` (Python 3.8, torch 2.3.1 + cu121)，位于 /scratch/pf2m24/miniconda3/envs/uniphys
- 计算节点无外网：所有 conda/pip/HF 操作前 `source /iridisfs/scratch/pf2m24/xiabao/PIE-VLA/scripts/net_tunnel.sh`
- 运行方式：`srun --jobid=<my_inter 作业> --overlap --ntasks=1 bash -lc '...'`

## 数据 / 权重
- SMPL:  `UniPhys/data/smpl/SMPL_NEUTRAL.pkl` -> projects/Umdd/KV-Control/body_models/smpl/ ；集群上没有 SMPL_MALE/FEMALE，暂时软链到 NEUTRAL（demo 用固定中性体型，不受影响；训练 shape variation 前需补真文件，来源 https://smpl.is.tue.mpg.de 的 SMPL v1.1.0）
- SMPL-X: `UniPhys/data/smpl/SMPLX_{NEUTRAL,MALE,FEMALE}.pkl` -> data/body_models/models/smplx/
- BABEL state-action-text: `UniPhys/data/babel_state-action-text-pairs/babel_{train,val}.pkl` (HF yan0116/SMPL_Humanoid_offline_dataset, 2.2 GB + 0.8 GB)
- 预训练 ckpt 与 sample_data: `UniPhys/download_data.sh` (gdown)

## 待办
- [x] Isaac Gym Preview 4：已下载解压到 `isaacgym/`，pip -e 装入 uniphys；gymtorch 需 gcc>=9，用 `module load gcc/11.5.0`，扩展缓存放 /scratch/pf2m24/torch_extensions/uniphys（见 scripts/activate_uniphys.sh）
- [x] 跑通预训练模型 headless 文字控制 demo（2026-09-10，"walk"，1 env，300 步，episode 长度 298 未摔倒，约 2 分钟；日志 logs/demo_text_walk.log）
- [x] 训练流程冒烟测试（2026-09-11，GPU 1，babel_train.pkl 全量加载 = 3090 batch/epoch @ bs32，30 步约 9 it/s，epoch 末闭环验证 4 env 平均 133.5 步，ckpt 落在 UniPhys/outputs/2026-09-11/09-00-45/checkpoints，单个 1.5 GB，可删）
- [ ] 精读 df_humanoid.py (877 行) 与 models/diffusion.py (553 行)

## 跑 demo 的命令
```
srun --jobid=<my_inter 作业> --overlap --ntasks=1 bash -lc 'source scripts/activate_uniphys.sh && python main.py phc/env=env_im_vae phc.env.num_envs=1 phc.headless=True phc.env.episode_length=300 diffusion_forcing/algorithm=df_humanoid diffusion_forcing.algorithm.guidance_params=2.5 diffusion_forcing.load=output/UniPhys/checkpoints/uniphys_T32.ckpt diffusion_forcing.algorithm.diffusion.use_ema=False diffusion_forcing.task=interact +diffusion_forcing.algorithm.text_prompt=walk +diffusion_forcing.name=play_t2m_single_text'
```

## 本地修改（相对官方仓库）
- config/diffusion_forcing/dataset/isaac_babel.yaml：加 `load_key_actions: null`，否则训练时 babel.py 报 AttributeError
- 训练命令必须加 `+diffusion_forcing.algorithm.text_prompt=walk`（README 漏了），否则 epoch 末验证 interact() 报 AttributeError: text_prompt

## 训练冒烟命令（GPU 1，30 步）
```
srun --jobid=<my_inter 作业> --overlap --ntasks=1 bash -lc 'export CUDA_VISIBLE_DEVICES=1; source scripts/activate_uniphys.sh && python main.py phc/env=env_im_vae phc.env.num_envs=4 phc.headless=True diffusion_forcing/experiment=exp_isaac diffusion_forcing/dataset=isaac_babel diffusion_forcing/algorithm=df_humanoid diffusion_forcing.dataset.data_path_list=data/babel_state-action-text-pairs/babel_train.pkl diffusion_forcing.dataset.n_frames=32 diffusion_forcing.task=training diffusion_forcing.wandb.mode=disabled diffusion_forcing.experiment.training.max_steps=30 diffusion_forcing.experiment.training.batch_size=32 diffusion_forcing.experiment.training.data.num_workers=4 +diffusion_forcing.algorithm.text_prompt=walk +diffusion_forcing.name=smoke_train'
```
正式训练：去掉 max_steps/batch_size 覆盖（默认 bs512、4M 步、每 100 epoch 存 ckpt），wandb 改 offline。注意 Lightning 会把节点上所有可见 GPU 拿来做 DDP，务必设 CUDA_VISIBLE_DEVICES。

## 线 A：G1 数据环境（2026-09-13 开始）
- TextOp 克隆到 `TextOp/`（TeleHuman/TextOp，子模块已拉）。跟踪器要求 Isaac Lab v2.1.0（Isaac Sim 4.5，Python 3.10）。
- 节点 RHEL8 glibc 2.28 装不了 Isaac Sim → 容器。SIF 在节点上挂不了（无 fusermount），必须用 sandbox 目录；`apptainer exec --nv <sandbox>` GPU 直通已验证。
- 镜像：NGC 公开 `nvcr.io/nvidia/isaac-lab:2.1.0`，在计算节点经隧道拉成 `/iridisfs/scratch/pf2m24/containers/isaaclab_2.1.0.sandbox`（脚本 scripts/build_isaaclab_sandbox.sh，tmux 会话 isaaclab_pull，日志 logs/isaaclab_pull.log）。apptainer 走代理要用 socks5:// 而非 socks5h://。
- HF Yochish/TextOp-Data：跟踪器 ckpt → TextOp/TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/model_75000.pt；G1 URDF 资产 → TextOpTracker/source/.../assets/unitree_description；重定向好的 50fps G1 数据 → TextOp/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps/{train,val}.pkl（4.9+1.8 GB）。HF 上 TextOpTracker/artifacts 为空，跟踪器用的 npz 动作需用 scripts/pklpack_to_npz.py 从上述 pkl 转换。
- 跟踪器动作 npz 键：fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w。
- G1 数据 pkl 格式（list，每项 dict）：feat_p（AMASS 文件名）、babel_sid、frame_ann [(start_s, end_s, label, [proc labels])]、length、motion{root_trans_offset (T,3), root_rot (T,4) xyzw, dof (T,23), pose_aa (T,27,3), smpl_joints (T,29,3), contact_mask (T,2), fps 50}。val 2078 段 7.8 h；train 约 6209 段。23 DoF = 29 去掉两侧腕 3 DoF，pklpack_to_npz 会补零成 29。
- 与 UniPhys val 序列 0 是同一段 AMASS（rub018/0017_lifting_light1），可用于跨数据集对照。
- [x] 2026-09-13 容器就绪、跟踪器评测通过、录制脚本 scripts/record_tracker_rollouts.py 跑通（val 子集 20 段：真实 rollout 成功 16/20；输出 data/g1_rollouts/val_subset20_x2.pkl）。详见 docs/04_g1_pipeline.md。
- [ ] 修复录制脚本显式 env.reset() 后首步误判终止的问题（每个 env 浪费一条队列项）
- [ ] 全量 val/train 参考动作 → npz → 录制（需估算时间：20 env 约 150 env-steps/s）
- [ ] 线 B：ADAPT 规格文档 + 扩散策略代码（改自 UniPhys）

## 线 B 进展（2026-09-13 晚）
- adapt/data.py（片段数据集，token=[a_{j-1},o_j] 125 维，T=20/H=5）、adapt/model.py（8 层 512 因果 decoder，34M）、adapt/diffusion.py（历史干净、未来逐 token 噪声、v-pred、cosine K=20、DDIM+CFG）、adapt/policy.py（闭环缓冲区）
- scripts/adapt_train.py 冒烟：val 子集 16 条 rollout → 828 片段，400 步 loss 1.7→0.37
- scripts/adapt_play.py 闭环冒烟（容器内，忽略参考，自己判摔）：小模型 stand 提示 20 env 全摔（预期，数据太少），机制跑通，推理 33 ms/步（未优化）
- 2026-09-14：val 2071 段 npz 转换完成（约 4 h）；全量 val 录制启动（tmux record_val，128 env，日志 logs/record_val.log）
- 2026-09-14 08:08 val 全量录制完成：2071 rollouts，成功 1775（85.7%），成功帧 1,140,981（≈6.3 h @50Hz），文件 data/g1_rollouts/val_all_x1.pkl（128 env 约 38 min）
- 2026-09-14 08:11 第一次正式训练 v1_valall 启动（tmux train_v1_valall，GPU1，holdout 10%，bs256，30k 步，lr 1e-4），日志 logs/train_v1_valall.log，输出 outputs/v1_valall
- train 集 6195 段 npz 转换在 rose12 GPU0（tmux convert_train，约 10/min，预计 ~10 h）；H200 (pink7011) 上 Kit 的 Vulkan DEVICE_LOST 会让转换卡死，不要在 H200 上跑 Kit
- 2026-09-14 v1_valall 训练中（30k 步，val loss 1k:0.319 → 15k:0.121 → 20k:0.117）。闭环诊断：跟踪器策略在评测环境 8/8 站住（环境与执行链路正确）；零动作 PD 保持默认姿态本身不稳（增益软）；扩散模型 5k/10k/15k ckpt 闭环全摔（平均 50–100 步），动作幅度偏小偏平均；离线一步动作 MSE：copy-prev 基线 0.051，5k 0.058，15k 0.046（开始超过基线）。
- 已加：exec_steps（动作块执行）、hist_noise_k/stab_level（UniPhys 式历史稳定化噪声，待 v2 训练验证）。isaaclab_exec.sh 不再在退出时关隧道（并发进程共用）。
- 2026-09-14 v1 30k 闭环网格（32 env × 400 步，单提示）：全部 fall_rate ≥ 0.97；最佳 walk/ddim10/exec1 成功 3%，平均 143 步；exec_steps=5 更差；ddim 10 比 2 略好。→ 纯 BC 扩散先验闭环不稳。
- 下一步：(a) v2 hist_noise + stab_level 评测；(b) DAgger：record_tracker_rollouts.py 加 --policy_ckpt --mix_prob，执行策略动作、记录跟踪器专家动作 expert_action，数据集 use_expert_label 用专家动作做标签（偏离 ADAPT 的残差 RL 路线，待用户决定）。
- 2026-09-14 v2_histnoise 15k 闭环（walk/ddim10）：stab_level 0/1/2 全摔，平均 76/104/84 步 → 历史噪声无实质帮助。
- DAgger 冒烟通过（子集：20 段 15 成功，策略执行 49% 步，专家/执行动作差 0.224）。第一轮全量采集启动 10:48（tmux dagger1，v1 30k，mix 0.5，128 env，输出 data/g1_rollouts/dagger1_valall_v1_mix05.pkl）。
- 训练脚本新增 --only_success/--drop_tail/--init_ckpt；数据集历史 token 用执行动作、未来 token 用专家动作。
- v2_histnoise 30k 闭环（walk/ddim10/stab1）：成功 6.2%，平均 141 步（v1 30k 同设置 3.1%/143 步）→ 微弱改善。
- DAgger1 采集速度约 1.1 步/s（128 env，含策略推理与逐 env 文本查找），预计 3.5 h；完成后自动启动 v3_dagger1（v1 初始化，val 原始 + DAgger 数据，含失败 rollout 去尾 25 帧，15k 步 lr 5e-5）。
- 2026-09-14 12:35 train 全量录制完成：6195 rollouts，成功 5330（86%），成功帧 3,000,866（≈16.7 h），data/g1_rollouts/train_all_x1.pkl（128 env，GPU0，55 min）
- DAgger1 完成：2071 rollouts，1706 成功，data/g1_rollouts/dagger1_valall_v1_mix05.pkl；v3_dagger1 训练中（191,550 片段，v1 初始化，15k 步）；5k ckpt 自动闭环评测 → logs/play_v3_5k.log
- v4_alldata 排队：纯 BC，train+val 全量，holdout 5%，60k 步，GPU0
- 2026-09-14 13:08 **v3_dagger1 5k 闭环**（32 env × 400 步，单提示，KIT_348 起始）：walk ddim2 成功 43.8%（v1 30k: 0%），walk ddim10 37.5%，stand ddim2 0%，stand ddim10 3.1% → DAgger 第一轮显著有效；stand 提示异常，待查。
- 排队：v3 15k 评测（walk/stand/run/jog + 切换协议）→ logs/play_v3_15k.log；DAgger2 采集（v3 15k 策略，mix 0.5）→ data/g1_rollouts/dagger2_valall_v3_mix05.pkl
- 用户提出的模型方向：flow matching（参考 MoGeFlow：rectified flow、logit-normal t、Euler）、loop transformer（权重共享循环）、"优化对齐策略"（待用户澄清）。计划：闭环做通后再换生成器做对比。
- 新增 adapt/flow.py（rectified flow：逐 token 连续 t，历史 t=1，logit-normal 采样，Euler + CFG），训练/策略 `--gen flow`。排队 v3f_flow（GPU0，v4 后，从零，val+dagger1，30k）。
- 排队 v5_dagger2（GPU1，dagger2 采集后，v3 15k 初始化，val+dagger1+dagger2，15k 步）。
- 排队 stand 提示探针：不同起始动作（KIT_379、EKUT_300）× 提示 stand / stand still / tpose / walk → logs/play_v3_5k_stand_probe.log
- 2026-09-14 13:40 **v3_dagger1 15k 闭环**（ddim2，32 env × 500 步，KIT_348 起始）：walk 71.9%，run 65.6%，jog 15.6%，stand 3.1%；切换协议（64 env，1000 步，5 提示 5–10 s 切换）成功 7.8%，平均 353 步。v1 对照 ≈3%。
- **发现文本条件几乎无效**：同一段 stand 历史下，stand/walk/run/tpose 提示的预测动作差异 ≈0.05（噪声底），且比真实数据更平滑 → 模型基本忽略文本、只延续历史，解释 stand 失败。
- 新增 adapt/model.py AdaLNDenoiser（adaLN-Zero，噪声等级+文本逐 token 调制，可选 n_loops 权重共享循环，2 层×4 循环 10.9M 参数），训练 `--arch adaln --n_loops N --uncond_prob p`。
- GPU0 队列改为：v4 → v3fa_flow_adaln（flow+adaLN，从零，val+dagger1，30k）→ v3f_flow（flow+xattn）。
- stand 探针（v3 5k）：起始 EKUT_300 站立姿态 → stand 96.9% / stand still 90.6% / tpose 81% / walk 87.5%；起始 KIT_379 → stand 0% / walk 78% → 失败由起始状态主导，提示压不过历史（文本条件弱）。
- 引导扫描（离线，归一化单位）：g=2.5 时换提示只改动作 0.05（采样噪声底 0.24），g=15 达 0.3–0.6。排队闭环 g=5/10 测试 → logs/play_v3_15k_guidance.log。
- record_tracker_rollouts.py 新增 --policy_prompt_mode random（策略按随机提示行动、标签仍为参考文本，制造过渡样本），用于 DAgger3。
- v3 15k 闭环引导扫描：stand g5 0% / g10 0%；walk g5 62.5% / g10 0% → 推理侧加大引导无效且伤稳定性；文本问题需训练侧解决（adaLN / 错位提示 DAgger）。
- **v4_alldata 60k 闭环**（纯 BC，train+val 4× 数据，val loss 0.095）：walk 65.6%（v1 0%，v3 DAgger 72%），stand 0% → 数据规模单独也大幅有效；stand 问题与数据量无关。
- 排队 dagger4：v4 60k 策略，train_all + val_all（先 train_all），mix 0.5，错位提示模式，dagger3 完成后启动。
- v3fa_flow_adaln 10k（从零）：文本敏感度 g2.5 |walk-stand| 0.145（v3: 0.054）、g5 0.285（v3: 0.108）→ adaLN 明显增强文本条件；闭环 walk 3.1% / stand 0%（训练不足，待 30k）。
- GPU0 队列追加：v3f_flow（flow+xattn）→ v3a_adaln_ddpm → v3d_xattn_ddpm，四者同数据（val+dagger1）从零 30k，用于拆分 flow / adaLN 的贡献。
- 2026-09-14 16:00 **v3fa_flow_adaln 30k 闭环**（从零，val+dagger1，KIT_348 动态起始）：Euler 10 步 walk 68.8% / **stand 96.9%**（v3 DDPM+xattn: stand 3%）；Euler 2 步 walk 18.8% / stand 28.1%。文本敏感度 g2.5 0.146–0.23。→ adaLN（+flow）解决了"听不见文本"的问题；flow 需要 ≥10 步 Euler（后续可换 Heun/蒸馏）。
- GPU0 队列改为：v3f_flow → v7_all_flow_adaln（flow+adaLN，train+val+dagger1-3 全量，60k）→ v3a_adaln_ddpm → v3d_xattn_ddpm。
- 16:05 GPU0：v3f 训完 → 立即启动 v3a_adaln_ddpm_pre（adaLN+DDPM 对照，val+dagger1，30k）；v7_all_flow_adaln 改为 dagger3 保存后启动（与 v3a 并行于 GPU0）。注意：后续链条会重复启动 v3a_adaln_ddpm，届时跳过/停掉。
- 16:20 **v5_dagger2 15k 闭环**（DDPM+xattn，v3 初始化，val+dagger1+dagger2）：walk 93.8% / run 81.2% / jog 40.6% / stand 31.2%；切换协议 18.8%（平均 425 步）。v3 对照：71.9 / 65.6 / 15.6 / 3.1 / 7.8%。→ DAgger 每轮稳步提升。
- v3fa 30k Euler10：run 25% / jog 15.6% / 切换协议 4.7%（平均 288 步）；推理 300–600 ms/步（10 步 Euler × CFG，GPU 共享）→ 动态提示仍弱（只用 dagger1），且需要更快采样（Heun/蒸馏）。
- **架构消融**（同数据 val+dagger1，从零 30k）：v3f flow+xattn 文本敏感度 g2.5 0.063（≈ v3 DDPM+xattn 0.054），闭环 walk 21.9% / stand 0–3.1%；v3fa flow+adaLN 0.146，stand 96.9% → 文本响应的提升来自 adaLN，flow 本身 ≈ DDPM。待 v3a adaLN+DDPM 补齐 2×2。
- v3fa 求解器对比（walk / stand 成功率，推理 ms 受 GPU 争用影响）：Euler10 68.8/96.9 (~300ms)；Heun5 65.6/81.2 (340ms)；Euler5 62.5/— (150ms)；Heun3 43.8/50.0 (170ms) → 质量随 NFE 单调，Heun 无明显优势；50 Hz 部署需蒸馏/减小模型，后续处理。
- 17:00 dagger3（错位提示）完成：2071 段，1732 成功 → data/g1_rollouts/dagger3_valall_v3_mix05_randprompt.pkl。自动启动：v6_dagger3（GPU1）、v7_all_flow_adaln（GPU0）、dagger4（train 集，v4 策略，错位提示，GPU1）。v3a_adaln_ddpm_pre 30k 训完（val 0.105），评测中。
- 17:10 **v3a_adaln_ddpm 30k 闭环**（从零，val+dagger1）：ddim2 walk 78.1% / stand 93.8%；ddim10 walk 93.8% / stand 96.9%；文本敏感度 g2.5 0.109。2×2 结论：**adaLN 是关键，DDPM ≥ flow（少步时明显更好）**。主线切换为 adaLN+DDPM。
- 启动 v8_all_adaln_ddpm（GPU0，与 v7 并行，train+val+dagger1-3，60k）；GPU0 后续只保留 v3d_xattn_ddpm 对照。
- 18:20 **v6_dagger3 15k 闭环**（xattn DDPM，加错位提示 dagger3）：walk 93.8 / run 84.4 / jog 37.5 / stand 68.8（v5 31.2）/ 切换 32.8%（v5 18.8）→ 错位提示 DAgger 数据显著改善 stand 与切换。
- 排队：v8 训完 + dagger4 保存后 → dagger5（v8 策略，val，错位提示，ddim2）→ v9（v8 初始化，全数据 + dagger4 + dagger5，20k）→ 评测。
- 20:45 v8_all_adaln_ddpm 60k 训完（val 0.0877，最低）；v7_all_flow_adaln 60k 训完（val 0.1287，flow 尺度）；dagger4 完成（6195 段，5168 成功，290 万帧）。自动进行中：v8/v7 评测、dagger5（v8 策略，val，错位提示）→ v9、v3d 对照。
- 2026-09-15 05:30 作业 1476768 到期，会话重启；后台等待链全部失效。**dagger5 在 2067/2071 时丢失（未存盘）** → 录制脚本加 --save_every 定期写 .partial；在新作业 1476769（rose02，GPU0）重跑 dagger5。
- **v8_all_adaln_ddpm 60k 闭环 ddim2**：walk 62.5 / stand 68.8 / run 96.9 / jog 100 / 切换 35.9%（平均 594 步，目前最好）；ddim10 评测因作业到期中断，重跑中。
- v7_all_flow_adaln 60k 闭环全 0%（1 s 内摔，异常）→ 离线文本敏感度 + 闭环复查中。v3d_xattn_ddpm 30k 训完，评测待排。
- 2026-09-15 06:10 用户决定：**不做 VQ 版本**；采用 MotionCraft 形式（github PengchengFang-cs/MotionCraft，已克隆到 vendor_motioncraft/）：rectified flow，网络直接预测 x0，速度 v̂=(x̂0−z_t)/(1−t)（1−t 下限 0.05），loss 在 v 空间，logit-normal t，Euler 32 步，adaLN 注入。已实现 `--flow_pred x0 --flow_loss_space v --flow_v_eps 0.05`。
- 启动 v3m_x0pred_vloss（adaLN + flow x0-pred/v-loss，val+dagger1，30k，GPU0 rose02）作为与 v3a（adaLN+DDPM 78/94%）和 v3fa（adaLN+flow v-pred 69/97% @10 步）的同数据对照；评测 Euler 2/10/32 步。
- **ADAPT 基线（Table 1，2048 rollouts × 20 s，每 5–10 s 切换，130 条提示，摔=非法躯干接触）**：纯扩散先验（DDIM 2 步，w/o residual）成功 0.804，Smooth 0.0100，Trans 0.0130，FeetSlide 0.063；完整 ADAPT 0.984；LangWBC 0.923；Offline TextOp 0.522。DDIM 1/5 步：0.706/0.792。→ 我们 v8（5 提示，64 env，自定摔判据）36%，需按 ADAPT 协议重测。
- 2026-09-15 06:30 **用户要求停止所有实验**（"不允许再继续做实验"，"浪费我的卡"）。已停止：dagger5 采集（1200 段 partial 已存）、v3m 训练、v8/v7/v3d/v3m 评测链、v9 链。GPU 上只剩用户自己的两个作业。此后任何 GPU 任务需用户明确批准。
- 用户要求：对比对象是 ADAPT 论文数字（纯先验 0.804 / 完整 0.984，Table 1 协议），不是我们自己的版本之间；并把 MotionCraft 的 DiT 主干搬过来（用户认为代价不大）。
- 2026-09-15 07:10 **ADAPT 协议评测 v8**（2048 × 20 s，75 条附录指令，5–10 s 切换，摔=除脚踝/手腕外刚体 z<0.06 或倾倒，2 步 DDIM）：成功 **0.074**（151/2048），平均 7.4 s 摔；动作平滑 0.709（归一化动作单位，论文 0.0100 为关节目标单位，不直接可比）；脚滑 0.129 m/s（论文 0.063）；推理 378 ms/步（512 env 共享满载卡）。论文纯先验 0.804，完整 0.984。

## 线 C：仿真角色 HumanML3D 物理基准（2026-09-15 起）
- 决定：目标改为 UniPhys/CLoSD/MIND/SCRIPT 的仿真角色协议（SMPL 人偶，Isaac Gym，HumanML3D 测试集，Guo 评估器 + 物理指标）；模型用 MotionCraft；动作空间先 69 维 PD 目标（32 维 PULSE 隐动作后续也做）；镜像片段第一版弃用；文本 CLIP 先行。用户批准步骤 1–5（数据、切片、物理真值检查、评测流水线、方案文档），不训练；用现有空闲卡（本次 1476738 swarma1005，两张空闲 A100）。
- 步骤 1（数据资产）：HF `yan0116/SMPL_Humanoid_offline_dataset/amass_state-action-pairs` 是**逐 AMASS 序列**的 joblib pkl（11,712 个，字段 body_pos[T,24,3] / dof_state[T,69,2] / root_state[T,13] / action[T,69] / pulse_z[T,32] / is_succ / fps=30）。HumanML3D index.csv（14,616 行）与之匹配：12,150 段命中 9,408 个文件；缺 2,466 段（humanact12 1,191 段不属于 AMASS；1,275 段 AMASS 文件不在 HF 里，如 push_recovery、扶栏行走等不可跟踪动作）。已下载 9,408 个所需文件到 data/humanml3d_phys/uniphys_hf（scripts/hml_phys/01_match_index.py, 02_download_hf.py；hf CLI 批量下载会卡死，改逐文件 hf_hub_download）。
- 评测代码（hml_phys/）：t2m/ 为 KV-Control 的 Guo 评估器与 HumanML3D 263 维处理代码原样搬入；sim2hml.py（Isaac 24 刚体 MuJoCo 顺序 z-up 30fps → SMPL 22 关节 y-up 20fps → 官方 process_file → 263 维）；phys_metrics.py（PhysDiff/CLoSD 定义的 Floating/Penetration/FootSliding、GMD Skating ratio、Jerk 两种单位）；evaluator.py（独立评估器：GT 加载完全复刻 Text2MotionDatasetEval，R-Precision/FID/MM-Dist/Diversity/MModality，batch 32，多次重复取均值 ±95%CI；归一化用评估器自带 meta 均值方差，本地 HumanML3D 目录的 Mean.npy 不是官方值不能用）。
- **评估器验证**（scripts/hml_phys/05_eval_gt_kinematic.py，测试集 4,646 条，3 次重复）：真值 R@1 0.515±0.008 / R@2 0.706 / R@3 0.797 / MM-Dist 2.972 / Diversity 9.468；Guo 论文真值 0.511 / 0.703 / 0.797 / 2.974 / 9.503 → 评估器正确。真值物理指标（运动学数据）：Floating 22.7 mm，Skating ratio 0.073，Jerk 5.37 mm/frame³。
- 协议细节（来源 CLoSD 代码 + MIND/SCRIPT 论文）：每条测试文本闭环跑一段（CLoSD 300 步=10 s，去掉前 16 帧上下文），只保留未摔倒的完整片段，长度取 GT 长度；转 263 维用官方流程；MIND 初始为统一站姿；物理指标定义各文不一，本项目按 PhysDiff/CLoSD 实现并记录定义。注意 MIND 报的 Phys-GT (R@1 0.559, Diversity 1.24) 与 SCRIPT 的 (0.651) 不是 Guo 评估器的数值尺度（Guo 评估器真值 Diversity≈9.5），说明它们用了各自的评估器；我们统一用 Guo 评估器并自报 Phys-GT。
- 切片对齐（scripts/hml_phys/03_build_dataset.py）：HumanML3D 帧 k ↔ 时间 (trim+start+k)/20 s，trim 为 raw_pose_processing 的数据集头部裁剪（Eyes_Japan/HDM05 3 s，TotalCapture/MPI_Limits 1 s，Transitions 0.5 s）；PHC 用 skip=int(fps/30) 重采样，100 fps 源（KIT、MPI_mosh）实际为 33.3 fps 但标 30 → 逐文件用内容对齐（263 维局部关节块与 new_joint_vecs 比较）在 30/33.33 间判定，记录 time_stretch；1,500 行冒烟：中位局部关节误差 0.053 m（PULSE 跟踪误差量级），KIT 多数判 33.33，其余数据集 30；29 个文件切片为空（物理序列比索引范围短）被丢弃。
- 步骤 2 完成（2026-09-15 13:40，scripts/hml_phys/03_build_dataset.py，32 核 3.5 min）：**10,903 段，22.0 h**（train 8,718 / 17.6 h，val 530 / 1.1 h，test 1,655 / 3.3 h）；丢弃 985 段（跟踪失败 is_succ=False）、262 段（物理序列比索引范围短，切片为空）；377 段覆盖不足 90%。逐文件帧率判定（含长度可行性约束）：KIT 1686:21 判 33.33，EKUT 104:1 判 33.33，其余数据集 30，MPI_mosh 混合；2,275 个文件判定不明用数据集多数票；382 个成功文件的物理序列长度装不下索引范围。局部关节误差中位数 0.060 m。输出 data/humanml3d_phys/hml_phys_{train,val,test}.pkl，build_stats.json，fps_alignment.json。
- 转换器验证：官方 263 维 → recover_from_ric → 我们的 sim2hml 转换 → 评估器：R@1 0.515±0.001 / FID 0.003 / MM-Dist 2.970 / Div 9.53（真值同批 0.518）→ 转换器与官方流程等价。本地 new_joints/ 目录与官方不一致（2,098 个测试 id 缺文件、部分文件与 new_joint_vecs 不对应），已改为从 new_joint_vecs/000021 恢复参考骨架，任何地方不再用 new_joints/。
- 步骤 4 流水线跑通：UniPhys 官方权重闭环（UniPhys/main_hml_rollout.py → hml_phys/uniphys_rollout.py，8 env 冒烟 33 s，8 段中 3 段摔倒）。全量测试集 4,646 条 rollout 已在 tmux `hml_roll_uniphys`（loginX002，srun bash step on 1476738 GPU1，256 env）运行，输出 data/humanml3d_phys/rollouts_uniphys_test.pkl，日志 logs/hml_phys/rollout_uniphys_test.log。步骤 3 物理真值检查（scripts/hml_phys/04_physgt_eval.py）在 GPU0 运行，日志 logs/hml_phys/physgt_eval.log。
- **三份独立代码审查（logs/hml_phys/review_{1,2,3}.md）共同发现的关键缺陷**：本地 HumanML3D 副本被重新编号（官方 007975/009707/011059 缺失，之后编号前移，镜像 = id+14613），而 index.csv 用官方 id → 之前构建的数据集约一半片段文本/划分错配，物理真值检查 R@1 只有 0.27。修复：hml_phys/hml_ids.py（用 texts.zip 原始文件核对），03/04 改用映射；其他修复：帧率候选加 31.25（SSM 250 fps）、子片段时间标签按 time_stretch 缩放、物理指标增加"仿真原始地面"口径、评估器子片段键去重、rollout 摔倒判定含最后一帧、06a 默认随机选句 + 固定长度选项 + 同句重复（MModality）、07 接入 MModality、清单加文件数/字节数。
- 重建数据集（v3）：**10,902 段 / 22.0 h**（train 8,734 / val 528 / test 1,640）；9,407 个文件 4.36 GB，8,768 个跟踪成功；对齐后所有文件局部关节误差 ≤0.25 m，中位数 0.042 m；长度不匹配 >15% 的测试条目从 537 降到 35。
- **步骤 4 验证：UniPhys 官方权重，HumanML3D 测试集 4,646 条 rollout（逐条目目标长度，首句文本，256 env，22 min）**：摔倒前未达目标 30.8%（Duration 0.692）。排除摔倒（3,213 条评测）：R@1 0.091±0.001 / R@2 0.160 / R@3 0.218 / FID 14.11 / MM-Dist 7.45 / Div 6.75（真值同批 0.513 / 0.704 / 0.797 / — / 2.98 / 9.52）。截断口径：R@1 0.088 / FID 14.05。物理（仿真原始地面）：Floating 17.6 mm，Skating ratio 0.051，Jerk 4.96 mm/frame³。对照论文：MIND 报 UniPhys R@1 0.0865 / Floating 20.5 mm（与我们一致），SCRIPT 报 0.143；FID 论文 0.49–0.60 与我们的 14.1 不在一个尺度（各家评估器不同，见 docs/06 §2.2）。
- **步骤 3 物理真值检查（修正编号后，scripts/hml_phys/04_physgt_eval.py，20 次重复）**：测试集 4,646 条评估条目中 1,760 条有 PULSE 跟踪物理片段（其余 2,861 条是镜像或未跟踪）。物理真值（fps_eff 口径）R@1 **0.458**±0.004 / R@2 0.655 / R@3 0.760 / MM-Dist 3.11 / FID 2.56 / Div 8.99；同一批片段的运动学真值 R@1 0.500 / MM-Dist 2.89；全测试集真值 0.515。→ PULSE 跟踪保留了约 92% 的文本匹配度，nominal 30 fps 口径几乎相同（0.457）。仿真原始地面物理指标：Floating **15.5 mm**（MIND 报 Phys-GT 15.58 mm，一致）、Skating ratio 0.023、Jerk 5.0 mm/frame³。FID 2.56 相对真值 0.002 偏大，来源待分解（子集选择效应 vs 物理跟踪效应，eval_physgt_test_eff_decomp.json）。
- FID 分解（eval_physgt_test_eff_decomp.json）：同一 1,760 条片段的**运动学**真值对全测试集真值的 FID 已是 1.60（子集选择效应：去掉了镜像和未跟踪的难动作），物理跟踪再加约 1.0 → 2.56。所以物理真值的 FID 上界要按"同子集"读：跟踪本身带来的 FID 增量约 1.0，R@1 损失 0.500 → 0.458。
- 固定 320 步 rollout（随机选句）第一次评测全部判为摔倒：审查建议的"摔倒步 ≤ 目标"把 320 步超时也算进去了。修复：uniphys_rollout.py 改用环境的 `_terminate_buf`（只含摔倒，不含超时）判摔；已录的 pkl 用 scripts/hml_phys/fix_fell_flags.py 修正标志（摔倒步 < 目标 才算摔）后重评。
- **固定长度 rollout（CLoSD 式，随机选句，每段 318 步即 10.6 s，256 env，23 min）**：10.6 s 内摔倒 47.3%（Duration 0.527；逐条目长度版 30.8%）。排除摔倒（2,447 条）：R@1 0.094±0.001 / R@2 0.164 / R@3 0.222 / FID 14.12 / MM-Dist 7.40 / Div 6.82；截断：0.088 / FID 13.94。两种长度设置、两种摔倒口径 R@1 都在 0.088–0.094，与 MIND 报的 UniPhys 0.087 一致。结果表见 docs/06 §2.1b。
- 2026-09-15 14:30 步骤 1–5 完成，三份审查意见已处理（除"每句 5 次 rollout / MModality 正式跑"留待训练后一起做）。GPU 上无本项目进程。下一步（需批准）：步骤 6 实现与训练（docs/07 的第 8 节四项待定）。
- 2026-09-15 晚 用户批准第六步：实现 + 训练。训练约定：bs256 × 30 万步（约 7,700 万样本，MIND 的两倍），每 5 万步存一个 ckpt，另存 val 损失最低的一个；主结果用最终步权重，val 最低权重为第二候选，测试集各评一次。损失两项：flow（速度空间）+ 位置速度一致性 0.01。评测起步统一中性站姿（phc.env.stateInit=Start，即站立片段第 0 帧）；UniPhys 对照行按此重跑。文本 CLIP ViT-L/14（checkpoints/clip/ViT-L-14.pt 已下载）。实现完后 2 个子 agent 审查（bug / 严格按约定 / 不偷工减料）。
- 卡：1476738 两张 A100 被用户的 MRISeg 训练占用后，UniPhys 重跑移到 1398707（blossom04 H200，tmux hml_roll_neutral）。
- 第六步实现（2026-09-15 晚）：hml_phys/tokens.py（token 计算，与 UniPhys get_repr 数值一致，含 UniPhys 的 −y 朝向约定；批量版本）、hml_phys/dataset.py（stride 1 滑窗 1,497,780 个，152 段短于 48 帧跳过；文本分配规则；静止起步 / 中性站姿增强；进度条件）、hml_phys/mc_model.py（复用 MotionCraft dit_blocks 的 FrameMotionTextDiT/TimestepEmbedder，root 15 / body 420，root 预测 token 直接作 body 桥接并 detach，AdaLN 条件 = 时间步 + pooled 文本 + 标量条件）、hml_phys/flow.py（MotionCraft 约定：t=1 干净，x0 预测，速度空间 loss，1−t 下限 0.05，Euler + x0 空间 CFG）、scripts/hml_phys/train_mc.py（AdamW 1e-4/0.01，bf16，EMA 0.995/10，文本 dropout 0.1 → 空句 CLIP 特征，每 5 万步 ckpt + val 最低）、hml_phys/mc_rollout.py + main_hml_rollout.py `+hml.policy=mc`（闭环执行）。CLIP ViT-L/14 缓存 32,332 句（GPFS memmap 写入会卡死，改为内存累积后一次写出）；token 统计 data/humanml3d_phys/token_stats.npz。
- 冒烟通过：train_mc.py（60 段、60 步、bs16，损失下降，ckpt 保存）；闭环 mc_rollout（8 env × 8 条，9 s，全摔，未训练模型的预期）。**正式训练 mc_v1 启动**（2026-09-15 19:5x，tmux mc_train_v1，1398708 blossom04 H200，bs256 × 30 万步，约 180 ms/步 ≈ 15 h；模型 181M 参数：hidden 768，root 2+4 / body 3+6，双流块的文本流使参数量高于 MIND 的 6×768）。输出 outputs/mc_v1，日志 logs/hml_phys/train_mc_v1.log。两个独立审查（review_step6_A/B.md）与训练并行；若审查发现实质 bug 则修复后重启训练。
- **UniPhys 对照行（中性站姿起步 stateInit=Start，随机选句，逐条目长度，H200 256 env 13 min）**：摔倒 28.9%（Duration 0.711）；排除摔倒 R@1 0.093±0.001 / FID 14.71；截断 R@1 0.088 / FID 14.32。与随机站姿起步的 0.091 / 0.088 基本一致，起步方式对 UniPhys 影响很小。这一行是最终对照表里 UniPhys 的口径。
- 2026-09-15 20:00 用户收回 1398707（空闲 H200）自用；本项目只占 1398708（训练 mc_v1）。训练结束后的闭环评测需另行申请一张卡（约 30 min）。
- **第六步两份独立审查（logs/hml_phys/review_step6_A.md、review_step6_B.md）结论与处理**：
  - 严重：闭环 mc_rollout 未关闭模仿环境的参考跟踪终止（termination_distances），任何不跟踪隐藏参考动作的策略 1 s 内被判摔倒 → 已加与 uniphys_rollout 相同的覆盖，并校验 ckpt 内 PD offset/scale 与环境一致、起始为 Start。
  - 主要：中性站姿增强只平移不对齐朝向（拼接处出现随机偏航跳变）→ 未来段绕竖直轴旋转到中性姿态髋部朝向，且只对开头静止且直立的片段做；验证损失未用固定随机数（best_val 是噪声）→ 固定 generator；无描述窗口用零特征而非 CLIP("") → 统一为 CLIP("")；root 流只看 15 维 root（MotionCraft 的 root 阶段看完整状态）→ 改为完整 token 输入；闭环预热 2 步被记录计数（与 UniPhys 不一致）→ 不记录不计数；增强概率分支使 rest 实际 0.15 → 独立判定；增强窗口的文本重叠区间 → 改为片段 [0:32)。
  - 审查 A 的 M1（动作与状态错位）经核对不成立：PHC 记录器在物理步之后写状态并与同步施加的动作配对，训练 token 与闭环缓冲区配对一致；已在 tokens.py/docs/07 写明约定。
  - 归一化：局部骨盆 xy 恒零、部分 6D 分量固定使 std 到 sqrt(1e-5)，local_vel 逐帧位移 std 0.006–0.05 是真实尺度，不加下限（UniPhys 同样不加）。
  - 未改（记录在 docs/07 §13）：一致性损失只覆盖 root 平移/速度；CFG 两分支；进度条件对 KIT 片段 11% 偏快；模型 182M。
- 原 mc_v1（旧代码，跑到约 8,000 步）已终止并归档 outputs/mc_v1_aborted_step*/；**修正后的 mc_v1 于 20:1x 重启**（同一命令，tmux mc_train_v1，1398708，182M 参数，约 180 ms/步）。冒烟：40 步训练 + 8 env 闭环均通过。
- 2026-09-16 09:05 训练 mc_v1 到 261k/300k 步（约 2 h 后结束）。**过拟合明显**：val 损失（固定噪声、EMA 权重、2,048 个 val 窗口）在 25k 步最低 0.254，之后单调上升到 260k 的 0.537；训练损失持续下降到 0.06。原因：182M 参数对 17.6 h 数据（stride 1 滑窗，30 万步 ≈ 51 遍）记忆。已存 ckpt：best_val（25k）、50k/100k/150k/200k/250k、300k（待）。
- 计划：训完后用 val 划分（1,4xx 条、Euler 10 步）筛 ckpt 与 CFG，只把最终步和 best_val 两个权重在测试集按协议（32 步、CFG 3.5）各评一次；同时准备 v2：模型缩到 512 维、训练步数按 val 曲线定（约 5–8 万步）。
- 2026-09-16 09:15 用户指示停止训练直接验证：mc_v1 在 277,900 步停止（最后存档 250k）。**val 划分筛选**（1,530 条，Euler 10，CFG 3.5，中性站姿，排除摔倒，5 次重复）：

| ckpt | 摔倒率 | R@1 | FID | MM-Dist | Floating mm | Jerk mm/f³ |
|---|---|---|---|---|---|---|
| best_val (25k) | 42.2% | **0.264** | 7.95 | 4.74 | 15.8 | 4.81 |
| 50k | 33.4% | 0.232 | 9.68 | 5.18 | 15.6 | 4.74 |
| 100k | **30.8%** | 0.215 | 8.65 | 5.36 | 14.6 | 4.03 |
| 250k | 37.1% | 0.197 | 9.88 | 5.79 | 13.4 | 2.96 |

  文本匹配随训练单调下降（与 val 损失曲线一致：过拟合），摔倒率在 100k 最低；越训越平滑（Jerk 降）但越不听文本。所有 ckpt 的 R@1 都远高于 UniPhys（同口径约 0.09）。测试集正式评测（32 步）对 250k 和 best_val 进行中。
- 2026-09-16 用户死规定（已写入项目 CLAUDE.md）：永久禁用 val 的任何操作，所有选择只在测试集完整协议下做；汇报必须是含全部基线的完整表。val 筛选（4 个 ckpt，40 min 卡时）作废。测试集评测顺序：250k、best_val（进行中）→ 50k、100k（已排队）。
- 2026-09-16 14:00 **测试集正式评测 mc_v1 250k**（4,646 条，Euler 32，CFG 3.5，中性站姿，256 env 90 min）：摔倒 29.6%（Duration 0.704）；排除摔倒 R@1 0.183±0.002 / R@2 0.293 / R@3 0.376 / FID 9.66 / MM-Dist 5.71 / Div 8.33 / Floating 13.9 mm / Jerk 3.21；截断口径 R@1 0.184 / FID 7.85。UniPhys 同口径 0.093 / FID 14.7 / Duration 0.711。best_val、50k、100k 测试集评测排队中。
- 2026-09-16 14:15 **测试集 mc_v1 best_val（25k）**：摔倒 38.8%（Duration 0.612）；排除摔倒 R@1 0.254±0.003 / R@2 0.394 / R@3 0.494 / FID 8.59 / MM-Dist 4.88 / Div 8.00 / Floating 17.0 / Jerk 5.85；截断 0.240 / FID 6.75。比 250k 的 0.183 高，摔倒更多。50k、100k 测试集评测进行中（tmux mc_eval3）。
- 2026-09-16 16:50 **测试集 mc_v1 50k**：摔倒 30.4%（Duration 0.696）；排除摔倒 R@1 0.230±0.002 / R@2 0.357 / R@3 0.445 / FID 10.68 / MM-Dist 5.28 / Div 7.98 / Floating 16.6 / Jerk 5.57；截断 0.223 / FID 8.59。100k 评测中。
- 2026-09-16 18:25 **测试集 mc_v1 100k**：摔倒 29.2%（Duration 0.708）；排除摔倒 R@1 0.221±0.002 / R@2 0.342 / R@3 0.428 / FID 8.98 / MM-Dist 5.33 / Div 8.23 / Floating 15.4 / Jerk 4.52；截断 0.216 / FID 7.85。四个 ckpt 测试集齐：R@1 25k 0.254 > 50k 0.230 > 100k 0.221 > 250k 0.183；摔倒 25k 38.8%，其余 29–30%。v1 结论：最佳 ckpt 25k（R@1 0.254，物理真值 0.458，UniPhys 0.093）；过拟合与"不听文本"随训练加重；摔倒率是机器人侧短板。GPU 已空闲，本项目无进程。
- v2 待批准方案（用户提议）：整段预测（历史 16 + 未来到片段末尾，最长约 300 帧，长度掩码，进度条件由剩余长度替代）、模型缩到 512 维、按测试集早停。
- 2026-09-16 19:0x **v2 实现并启动**（用户定：模型缩小、5 万步、每 1 万步存、整段预测）：dataset/flow/model/train/rollout 加 `whole_sequence` 模式——未来 = 历史末尾到片段末尾（上限 304 帧，长度可变，valid 掩码；损失和采样只作用于有效帧；闭环每次生成剩余帧数）；增强窗口的未来为整段。token 统计按整段窗口重算（token_stats_v2.npz：root_trans std 0.49/0.77 m，v1 为 0.18/0.34）。训练窗口 1,705,457 个。模型 hidden 512、root 2+4 / body 3+6，81.6M 参数。冒烟：40 步训练 + 8 env 闭环通过。正式训练 mc_v2：bs256 × 5 万步，ckpt 每 1 万步 + best_val，约 414 ms/步 ≈ 6 h，GPU 59 GB（tmux mc_train_v2，1398708）。日志 logs/hml_phys/train_mc_v2.log。
- 2026-09-17 05:00 **v2 训练完成并评测第一个 ckpt**。训练：val 损失 5k 0.829 → 20k **0.275（最低）** → 50k 0.332，比 v1 平坦得多（v1 25k 0.254 → 250k 0.537），小模型 + 整段预测确实抑制了过拟合。
  **测试集 mc_v2 50k**（Euler 32，CFG 3.5，中性站姿）：摔倒 **75.4%**（Duration 0.246，v1 同设置 29%），排除摔倒 R@1 0.230 / R@2 0.367 / R@3 0.462 / FID 9.11 / MM-Dist 5.17 / Div 7.50 / Floating 15.9 / Jerk 5.45；截断 0.213。→ **文本匹配与 v1 相当，但稳定性严重退化**：摔倒中位步 117（v1 170），即约 3.9 s 就倒，占目标长度 53%（v1 69%）。rollout 耗时也从 90 min 涨到 3.5 h（未来长度最长 304 帧）。
  可能原因待查：(a) 整段预测把模型容量分散到远期帧，近期 4 帧动作精度下降；(b) 长窗口里 root 位置尺度大（std 0.49/0.77 m），归一化后近期误差被压小，损失被远期主导；(c) 模型从 182M 缩到 81.6M。其余 ckpt（best_val 20k、10k、30k、40k）测试集评测排队中，先看曲线再定归因。
- 2026-09-17 06:00 停止 v2 剩余 ckpt 的 rollout 评测（用户：rollout 浪费时间），GPU 空闲。用户指出矛盾：模型缩小但评测成本从 90 min 涨到 3.5 h ——根因是整段预测每 4 步生成最多 320 个 token（v1 为 48），注意力平方增长，且 300 多个远期帧生成后即丢弃；训练损失里远期帧与被执行的 4 帧同权，正是摔倒率升到 75% 的原因。结论：要整段语义应走 MIND 的"整段压成低维意图向量作条件"，而不是逐帧生成整段。
- 代码已推送 GitHub：https://github.com/PengchengFang-cs/mG1 （main，85 个文件，仅代码与文档；data/outputs/logs/checkpoints 与三方克隆 UniPhys/TextOp/isaacgym/vendor_* 由 .gitignore 排除）。新增 README.md 说明基准、数据、模型与当前结果表。
- 2026-09-17 06:30 **清理磁盘，释放 48 GB**（79.7 → 31.7 GB）。删除：G1/ADAPT 线全部训练输出（v1–v8 等 14 个目录）、冒烟与中止的运行（mc_smoke、mc_smoke_v2、mc_v1_aborted）、mc_v1 的 50k–250k 与 mc_v2 的 10k/30k/40k/50k 权重、被禁用的 val 产物、已评完的 rollout pkl。保留：outputs/mc_v1/best_val.pt（2.8 GB，当前最好结果 R@1 0.254）、outputs/mc_v2/best_val.pt（1.2 GB，整段预测版留档）、全部评测 JSON（数字都在里面）、rollouts_uniphys_test_neutral.pkl 与 rollouts_mc_v1_best_val_test.pkl（基线行与最好行，可离线重算指标）。未动：data/humanml3d_phys（数据集、HF 源文件、CLIP 缓存）、data/g1_*（G1 线数据 14.5 GB，待用户决定是否删）。
- 2026-09-17 物理指标跨论文可比性核对（docs/06 §2.1c）：Floating 定义一致（MIND 物理真值 15.58 vs 我们 15.51），**我们 250k 的 13.9 mm 是该表最低，优于 MIND 17.1 与所有已发表数字**；Jerk 归一化不同（差约 1800 倍）不可比；Duration 只有 SCRIPT 报且口径未明；Penetration/Skating 各家称可忽略。§2.1d 记录训练时长与"听文本 vs 动得干净"的权衡。MIND 只用 HumanML3D、无数据增强 → 与 MIND 的差距是纯方法差距；数据量议题搁置直至追平 MIND（用户指出 SCRIPT 的规模来自 MotionMillion 而非镜像，我此前把两者并列是误导）。MIND 消融准确值：仅 AdaLN 0.174 → +文本交叉注意力 0.316 → +近期意图 0.323 → +整段意图 0.360 → +VAE 0.468。
- 2026-09-17 **Jerk 与 Duration 口径对齐（docs/06 §2.1e、§2.1f）**。Jerk：SCRIPT 给出定义（三阶有限差分，评测省略 Δt³ 并 ×10³，24 刚体、30 Hz 原始仿真输出，L2）；MIND 表值小 1000 倍是米未换算成毫米。经验验证：我们物理真值按此口径 2.289 mm/frame³ @30 Hz，对 SCRIPT 2.941 / MIND 2.717，同量级；原来的 20 fps/22 关节口径是 5.27（帧率三次方差别）。Duration：SCRIPT 为帧加权（Σ有效帧/Σ参考帧，根高 <0.15 m 判摔），我们重算 UniPhys 0.871、mc_v1 25k 0.818，对 SCRIPT II 0.981。新增 hml_phys/phys_metrics.py 的 *_raw 系列与 scripts/hml_phys/08_paper_metrics.py（纯离线重算，不占卡）。原始仿真口径下 Floating：物理真值 15.49、UniPhys 17.60、我们 17.19。
- 2026-09-17 **ARDY 代码核对**（NVIDIA，SIGGRAPH 2026，arXiv 2607.08741，克隆到 vendor_ardy/，33 MB）。**仓库是纯推理版**：没有训练脚本、损失、优化器、数据集；模型超参在 HF 权重目录的 config.yaml 里。
  结构与我们同源：两段去噪器（根 Transformer → 身体 Transformer，训练时 detach 桥接，历史帧保留干净值），预测 x0，窗口内双向注意力，无 KV 缓存（每个去噪步都全量重算），滑窗自回归。
  我们更强的地方：文本用 50 个 CLIP token 走联合注意力（它是 LLM2Vec 均值池化成**一个** 4096 维 prefix token）；历史是精确物理量（它把身体历史每窗**重新量化**回 FSQ 格点）；有进度条件（它没有）；采样器 rectified flow 32 步（它 DDIM 约 100 步子采样）。
  它更强的三处，都便宜：(1) **相对位置索引**——token index 减去 history_len//P，使第一个生成 token 恒为 0、历史为负，模型对历史长度不变，推理时可自由伸缩历史（auto_latent_twostage_denoiser.py:190-193）；(2) **每窗把坐标系重心移到最新帧**（我们是窗口最旧帧），并用一个 first_heading_angle prefix token 保留世界朝向；(3) **根→身体桥接送的是局部速度根（4 维）而非绝对根**，去掉了平移/朝向这个无关变量。
  不可移植：FSQ 量化。其身体隐 token 只有 **5 维、每维 16 级**（16^5≈100 万码）覆盖 4 帧全身，整个 token = 5·P+5 = 25 维，这是它实时的根本原因；但量化 69 维 PD 动作等于量化力矩，姿态错是视觉瑕疵，动作错是摔倒。其"生成后修正"路线（C++ 脚滑/IK 后处理、约束填充、重量化）对物理策略全部不可用。
  另注：交互演示的默认历史裁剪是 **1 个 token = 4 帧**，8 秒/10 秒是训练上限与 TRT 预算，不是默认值；其防漂移手段是重量化 + 硬裁剪（注释原话：超过训练窗口的生成会退化成抖动）。
- 2026-09-18 **v3 代码实现完成**（docs/07 §15），改动四处：
  1. **局部根桥接**（tokens.root_to_local_root + mc_model.to_local_root）：预测出的 15 维全局根 → 反归一化 → 4 维局部根 [偏航角速度, dx/dt, dy/dt, 根高] → 用单独的 local_root 统计归一化 → 送 body 流；训练时 detach，测试时可导；最后一有效行复制前一行。因为窗口内的行可能不连续（稀疏历史），有限差分按真实帧间隔 dt 归一，dt=1 时退化为 KiMoDo 公式。body_input_proj 输入维度 15+420*2 → 4+420*2。
  2. **短未来**：whole_sequence 默认关闭，F=32、K=4。
  3. **长历史**：窗口布局 [16 稀疏 | 16 稠密 | 32 未来]，稀疏帧按 SCRIPT 式(6) 从前 138 帧中抽（tokens.sample_sparse_history，alpha=0 退化为均匀）；token 先在**连续跨度**上算再按行 gather，保证每行的瞬时速度正确；每行带**带符号 frame_index**（第一个生成帧为 0，历史为负、真实帧偏移），RoPE 直接吃负数（_rope_cos_sin 是解析式，无需查表）；训练时按样本随机抽 n_sparse 与 alpha（15% 概率完全无稀疏历史），使历史长度成为测试时可扫的旋钮。
  3b. **坐标原点移到最新历史帧**（canonicalize(origin=...)）；重算统计 token_stats_v3.npz（含 local_root_mean/std）。
  验证（CPU）：稀疏采样器 alpha 0/3/5 的均值索引 73/99.5/110（越大越偏近期）、16 个互异且在界内；canonicalize 任意 origin 处 root_trans=(0,0,h)；局部根按 gather + frame_index 与连续计算逐值一致；数据集逐样本不变量（掩码布局、frame_index 单调且边界为 −1/0、有效行数、有限值）全部通过；H_sparse=0 精确退化为旧的 48 token 窗口。
  验证（GPU）：训练冒烟 40 步（81.6M，[16|16|32]，local_root=True）损失正常下降、ckpt 正常；闭环冒烟 8 env×8 条通过（未训练模型全摔，预期）；per-window 与 batched 分词一致（root 0，body 5e-4 float32 误差）；torch 局部根与 numpy 参考一致 1.4e-6。
