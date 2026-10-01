# Local edits to vendored third-party code

Everything else in `external/` is an unmodified clone. These are the only changes, kept here so the
reproduction is auditable.

## FRoM-W1

### `H-GPT/hGPT/models/archs/hgpt_lm.py` — load the LM at its own dtype
`AutoModelForCausalLM.from_pretrained(model_path, ...)` was called without `torch_dtype`, so transformers
materialises fp32 (32 GB for 8B parameters) even though the checkpoint's `config.json` declares
`bfloat16`. Added `torch_dtype="auto"`.

## Not modified, worked around instead

* **Submodules use SSH URLs** (`git@github.com:humanoidintelligence/...`). Cloned over HTTPS into the same
  paths instead of changing `.gitmodules`.
* **Config paths** point at the authors' cluster. Our copies live in `configs_fromw1/` (a copy of
  `configs/archs` plus our own `assets.yaml` and exp config); the vendored `configs/` is untouched.
* **`lora_merge.py` builds the tokenizer from the adapter directory**, whose `tokenizer.json` declares only
  515 motion tokens (a 512-entry codebook) while the adapter weights are `[130307, 4096]`, i.e.
  128256 + 2051 = a 2048-entry codebook matching the only released VQ-VAE. We do not use their script;
  `scripts/fromw1_merge_lora.py` rebuilds the tokenizer at the correct size and asserts the result.
* **`H-ACT/retarget/main.py` calls the MANO hand retargeting unconditionally.** G1 here has no dexterous
  hands and we have no MANO models, so the body path (`body_retarget.process_data`) is called directly.

## 2. `H-ACT/human2humanoid/legged_gym/legged_gym/utils/transform.py` (2026-10-01)

**改动**：第 1 行的 `from pytorch3d.transforms import (...)` → `from phc.utils.pytorch3d_transforms import (...)`。
原文件备份为 `transform.py.orig`。

**原因**：`pytorch3d` 不在 `requirements.txt` 里，也没装，评测路径一 import `legged_gym.envs` 就
`ModuleNotFoundError`。而**仓库自己已经 vendored 了同一个模块** —— `phc/phc/utils/pytorch3d_transforms.py`
（727 行），并且仓库里其他所有调用点都用的是这个副本：

- `phc/phc/utils/torch_utils.py:34`
- `phc/phc/env/tasks/humanoid_im_mcp_demo.py:21`
- `scripts/data_process/grad_fit_h1_shape.py:24`

**只有 `legged_gym/utils/transform.py` 这一个文件伸手去拿外部包**。它需要的四个函数
（`quaternion_apply`、`axis_angle_to_matrix`、`matrix_to_quaternion`、`quaternion_multiply`）
在 vendored 副本里全都有，已逐个核对。

所以这是把一处漏改的 import 对齐到仓库自身的既有约定，不是引入新实现 ——
真装 pytorch3d 反而会让这个文件用上与仓库其余部分不同的代码路径。

## 3. `poselib`（为 h2h 环境单独 vendored，2026-10-01）

**改动**：`external/vendored_for_h2h/poselib/poselib/skeleton/skeleton3d.py` 里
`xml_node.attrib.get("pos")` → `xml_node.attrib.get("pos", "0 0 0")`。
UniPhys 自己的 `UniPhys/poselib` **未改动**；h2h 通过 `activate_h2h.sh` 里的 `PYTHONPATH` 遮蔽
（PYTHONPATH 的搜索先于 `.pth` 条目）。

**原因**：human2humanoid 的 `legged_gym/envs/base/legged_robot.py:25` 写
`from poselib.skeleton.skeleton3d import SkeletonTree`，`phc` 则写
`from poselib.poselib.skeleton.skeleton3d import ...` —— **两个不同的模块路径，而仓库既不自带
poselib 也没在 requirements 里声明它**。h2h 是从 `uniphys` 克隆来的，于是继承了 UniPhys 的
editable poselib。那份的 `from_mjcf` 直接 `np.fromstring(attrib.get("pos"), ...)`，body 没有 `pos`
就 `TypeError: a bytes-like object is required, not 'NoneType'`。

**为什么这个默认值是对的，不是掩盖问题**：MuJoCo 规定 body 省略 `pos` 即 (0,0,0)。
`g1_21dof.xml` 有两个 body 省略了它 —— `waist_yaw_link`、`torso_link` —— 而 **Unitree 官方的
`g1_29dof.xml` 是完全相同的模式**，所以这不是哪个资产被改坏了。已用偏移累加独立验证：

```
21/29dof: waist_yaw(0,0,0) + waist_roll(-0.0039635,0,0.044) + torso(0,0,0) + shoulder(…,0.24778)
23dof:    torso(-0.0039635,0,0.054)                                        + shoulder(…,0.23778)
pelvis→left_shoulder 合计均为 (-0.0000072, 0.10022, 0.29178)，逐位相同
```

FRoM-W1 自己的 `retarget/body_retarget/robot.py:from_mjcf` 与 phc 的
`torch_robot_humanoid_batch.py:from_mjcf` **都已经写了 `.get("pos", "0 0 0")`** ——
只有外部 poselib 这一条路径没有。我们的 `scripts/fromw1_g1_21dof_config.py` 同样用了该默认值，
因此已建成的动作库不受影响。

## 4. `H-ACT/human2humanoid/legged_gym/legged_gym/envs/base/legged_robot.py` (2026-10-01)

**改动**：`LeggedRobot.__init__` 第 151 行
`if self.cfg.train.distill:` → `if self.cfg.train.distill and not self.cfg.env.test:`
（守卫 `self.load_expert()`）。原文件备份为 `legged_robot.py.orig`。

**原因**：评测一个**已发布的 student** 会在建环境时就死掉：

```
[EXPERT] loading expert policy: .../logs/robot:teleop/25_12_05_23-30-46_OmniH2O_TEACHER_G1/model_50000.pt
FileNotFoundError: [Errno 2] No such file or directory
```

`config_eval.yaml` 同时设了 `train.distill: True`、`env.test: True`，并把
`dagger.load_run_dagger` 指向 `25_12_05_23-30-46_OmniH2O_TEACHER_G1` —— **teacher 权重没发布**
（他们 README 里 G1/H1 的 teacher 都写 "Teacher (TBD)"），所以这个文件不存在。

**为什么这不是绕过**：expert 只在 `step()` 里被消费，而那里**已经写着完全相同的条件**：

```python
if self.cfg.train.distill and not self.cfg.env.test:
    if "expert_policy" in self.__dict__:
        ...
        gt_actions = self.expert_policy(full_obs)
```

即 test 模式下 expert 被加载却永不调用。`__init__` 的加载条件只是漏了 `test` 这一项。
这个改动把**加载条件对齐到他们自己已经写明的使用条件**，没有跳过任何会被用到的东西 ——
`distill` 的其余作用（如 `setup_kin_info()`）保持原样生效，所以没有用
`train.distill=False` 去粗暴关掉。
