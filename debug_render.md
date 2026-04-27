# `debug_render.py` 说明

这个脚本的目标是把同一个视角下的三类结果按列对齐：

- `GT`
- `Original Render`
- `GT_reshade`

这样每一行只看一种量，方便直接判断：

- 原始渲染到底错在哪
- 把材质替换成 GT 之后，这一项有没有明显变化
- 问题主要来自材质，还是来自 normal / visibility / env lookup


## 1. 脚本做了什么

`debug_render.py` 会：

1. 读取实验目录里的 `cfg_args`，复用训练配置。
2. 加载指定 iteration 的模型。
3. 直接调用项目原始 `render_surfel(...)` 得到 `Original Render`。
4. 从数据集读取 GT：
   - `RGB`
   - `albedo`
   - `roughness`
   - `metallic`
   - `normal`
   - `envmap`
5. 可选地用数据集 GT envmap 覆盖模型当前 envmap。
6. 可选地做一次 `GT_reshade`：
   - 保持原始几何 / alpha / depth / indirect / visibility 计算链路
   - 只把材质输入换成 GT
   - 再调用原项目的 `get_specular_color_surfel(...)`
7. 把每个量整理成固定三列：
   - 第 1 列：`GT`
   - 第 2 列：`Original Render`
   - 第 3 列：`GT_reshade`


## 1.1 严格复现训练可视化

如果你的目标不是做诊断，而是尽量复现 `train.py::save_training_vis(...)` 在某一轮真正看到的 render，推荐使用：

- `--reproduce_training_vis`

这个模式会额外做两件事：

1. 用和训练一致的 render function 选择逻辑  
   也就是按 iteration 自动选择：
   - `render_initial`
   - `render_volume`
   - `render_surfel`

2. 恢复“该轮保存训练图时实际还在用的 mesh”  
   训练里 `save_training_vis` 是在 mesh 更新之前执行的，所以：
   - 普通模式会恢复 `<= iteration` 的最新 mesh
   - `--reproduce_training_vis` 会恢复 `< iteration` 的最新 mesh

这在 indirect / visibility 对最终颜色影响很大时，通常是最关键的差别。


## 2. 三列各表示什么

### 第 1 列 `GT`

- 数据集直接提供的 GT 量。
- 如果该行没有 GT，就放黑图占位，保证三列始终对齐。

### 第 2 列 `Original Render`

- 当前 checkpoint 直接跑 `render_surfel(...)` 得到的原始结果。

### 第 3 列 `GT_reshade`

- 不是“完整 GT 重渲染”。
- 它仍然使用当前模型的：
  - 几何
  - alpha
  - 深度
  - visibility / indirect 链路
  - env 查询逻辑
- 只是把材质输入替换成 GT 后，再走一次原始着色函数。

对纯材质行来说，例如：

- `albedo`
- `roughness`
- `metallic`

第三列显示的是 `GT_reshade` 实际使用的输入，所以通常会和第一列相同。


## 3. 当前 debug 图的固定布局

每张 `*_debug.png` 现在固定为 `3` 列：

```python
make_grid(..., nrow=3)
```

列顺序始终是：

1. `GT`
2. `Original Render`
3. `GT_reshade`

行顺序如下。

### 第 1 行

1. `GT RGB`
2. `render`
3. `gt_reshade render`

### 第 2 行

1. `GT albedo`
2. `predicted albedo / albedo_map` (`base_color_map` now means rendered `base_color`)
3. `albedo used in gt_reshade`

### 第 3 行

1. `GT roughness`
2. `predicted roughness`
3. `roughness used in gt_reshade`

### 第 4 行

1. `GT metallic`
2. `predicted refl_strength`
3. `metallic used in gt_reshade`

说明：

- 当前项目真正参与着色的是 `refl_strength`
- 所以这里第 2 列是模型预测的 `refl_strength_map`
- 第 3 列是 `GT metallic` 作为 `gt_reshade` 的 `refl_strength` 输入

### 第 5 行

1. `(blank)`
2. `diffuse_map`
3. `gt_reshade diffuse`

### 第 6 行

1. `(blank)`
2. `specular_map`
3. `gt_reshade specular`

### 第 7 行

1. `GT normal_cam_vis`
2. `rend_normal_cam_vis`
3. `normal used in gt_reshade`

说明：

- 默认情况下，第 3 列通常就是原始渲染使用的 normal 可视化
- 如果加了 `--use_gt_normal_in_reshade`，那第 3 列会变成 GT normal

### 第 8 行

1. `(blank)`
2. `rend_alpha`
3. `alpha used in gt_reshade`

### 第 9 行

1. `(blank)`
2. `direct_light`
3. `gt_reshade direct_light`

### 第 10 行

1. `(blank)`
2. `indirect_light`
3. `gt_reshade indirect_light`

### 第 11 行

1. `(blank)`
2. `visibility`
3. `gt_reshade visibility`


## 4. 额外输出

脚本还会额外保存：

- `env_debug.png`
  - 上下两张 env 可视化
- `grid_layout.txt`
  - 当前 `*_debug.png` 的行列定义，方便对照查看


## 5. 如何看图

推荐按“逐行横向比较”的方式看：

### 看最终结果

先看第 1 行：

- `GT RGB`
- `render`
- `gt_reshade render`

如果第三列比第二列明显更接近第一列，说明：

- 材质是重要问题来源

如果第三列和第二列都离第一列很远，说明：

- 主问题更可能在 normal / visibility / indirect / env sampling / geometry

### 看材质

重点看第 2 到第 4 行：

- `albedo`
- `roughness`
- `metallic / refl_strength`

如果第 2 列和第 1 列差很多，说明模型预测材质本身就偏了。

### 看光照分解

重点看第 5 到第 6 行：

- `diffuse`
- `specular`

如果 `diffuse_map` 本身已经带明显天空色或地平线色，说明错色不只是 specular 导致的。

### 看法线

重点看第 7 行：

- `GT normal`
- `Original Render normal`
- `GT_reshade normal used`

如果原始法线和 GT 差很多，那么即使材质换成 GT，最终结果也未必会好很多。

### 看遮挡和间接光

重点看第 9 到第 11 行：

- `direct_light`
- `indirect_light`
- `visibility`

如果这里的颜色或分块模式很怪，通常会直接反映到最终地板偏蓝 / 偏暖 / 分块不均的问题上。


## 6. 命令示例

你之前能跑的命令可以直接继续用：

```bash
CUDA_VISIBLE_DEVICES=7 /nfs/508_users/disk5/wsq/miniconda3/envs/ref/bin/python debug_render.py \
  --model_path /nfs/508_users/disk5/wsq/ENVS/ref-xjm/output/table_reflect/table_reflect-0330_1108 \
  --iteration 30001 \
  --all_cameras \
  --max_cameras 60
```

如果你想强制只看训练相机，也可以显式加上：

```bash
--split train
```

如果你想尽量复现 `train.py` 里某一轮 `save_training_vis` 的 render，推荐用：

```bash
CUDA_VISIBLE_DEVICES=7 /nfs/508_users/disk5/wsq/miniconda3/envs/ref/bin/python debug_render.py \
  --model_path /nfs/508_users/disk5/wsq/ENVS/ref-xjm/output/table_reflect/table_reflect-0330_1108 \
  --iteration 30000 \
  --split train \
  --reproduce_training_vis \
  --all_cameras \
  --max_cameras 60
```

如果你想手动指定 ray tracer mesh，也可以用：

```bash
--mesh_iteration 28000
```

如果你想验证“是不是 mesh / visibility 导致颜色漂移”，可以直接关闭 mesh 恢复做对照：

```bash
--disable_restore_mesh
```


## 7. 这个脚本的边界

这个脚本仍然是诊断工具，不是严格意义上的 GT forward renderer。

它不会：

- 用 GT 几何替换当前几何
- 用 GT 深度替换 `surf_depth`
- 用 GT visibility 替换当前 ray tracing 结果

所以 `GT_reshade` 更准确的理解是：

- `same geometry`
- `same camera`
- `same env sampling path`
- `same visibility / indirect path`
- `only swap material inputs to GT`

这正是它最适合拿来回答的问题：

- 错误主要来自材质，还是来自更上游的法线 / 光照链路
