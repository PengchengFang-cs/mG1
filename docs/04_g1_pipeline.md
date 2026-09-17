# 线 A：G1 数据管线（TextOp tracker in Isaac Lab 2.1.0）2026-09-13

## 环境
- 容器 sandbox：/iridisfs/scratch/pf2m24/containers/isaaclab_2.1.0.sandbox（NGC isaac-lab:2.1.0，Isaac Sim 4.5，Python 3.10）
- 启动：`bash scripts/isaaclab_exec.sh [--writable] [--offline] <cmd>`；默认开 SSH 隧道并把代理传进容器（Kit 需要下载 Nucleus 资产和注册表扩展）。HOME 映射到 /iridisfs/scratch/pf2m24/isaaclab_home（扩展缓存、Kit 数据都在那）。
- 容器 python：/workspace/isaaclab/_isaac_sim/python.sh。已装 textop_tracker、rsl_rl 2.3.3 分支（editable，指向 TextOp/），joblib、tqdm、pysocks。
- 踩坑：pip 会把 numpy/torch 换成 PyPI 版本 → 必须保持 numpy==1.26.0、torch==2.5.1+cu118、torchvision==0.20.1+cu118（Isaac Sim 自带的 ml_archive 是 cu118，两处不一致会报 torchvision::nms）。写模式 (--writable) 下绑定挂载点必须先在 sandbox 里 mkdir。SIF 在节点上挂不了（无 fusermount），只能用 sandbox。

## 数据
- 参考动作：TextOp/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps/{train,val}.pkl（list of dict：feat_p, frame_ann, motion{dof(T,23), root_trans_offset, root_rot(xyzw), fps 50}）
- scripts/make_motion_subset.py → name→motion dict + meta（frame_ann）
- TextOpTracker/scripts/pklpack_to_npz.py（容器内，需 Kit 做 FK）→ artifacts/<set>/<name>/motion.npz（fps, joint_pos/vel (T,29), body_*_w (T,30,...)）
- 23 DoF 数据在转换时腕关节补零到 29。

## 跟踪器评测（scripts/track_eval.py）
- 任务 Tracking-Flat-G1-ProjGravObs-MNMLP-v0，ckpt model_75000.pt，须传 anchor_body_name=pelvis, future_steps=5, actor/critic [2048,1024,512]
- 观测 431 = 参考未来 5 帧关节位置速度 290 + 锚点未来位置 15 + 朝向 30 + 投影重力 3 + 根线速度 3 + 角速度 3 + 关节位置 29 + 关节速度 29 + 上一步动作 29
- 动作 29 维，目标 = default_joint_pos + scale × action，scale = 0.25 × effort_limit / stiffness
- 50 Hz（dt 0.005 × decimation 4）；失败终止：锚点 z 偏差 > 0.25 m、任一关键刚体 z 偏差 > 0.25 m、锚点朝向偏差 > 0.8
- val 子集 20 段、从第 0 帧开始、无随机化：成功 42 / 失败 35（≈55%）。转身、走、推撑成功率高；举物、坐下失败多。
- 速度：20 env 约 150 env-steps/s（GPU 与他人训练共享）

## 录制（scripts/record_tracker_rollouts.py）
- 输出 pkl：rollouts[i] = {proprio (T,67), prev_action (T,29), action (T,29), joint_target (T,29), root_state (T,13), ref_t (T,), motion, feat_p, frame_ann, success}
- proprio 布局 = ADAPT 的 67 维本体感知；+ prev_action = 96 维
