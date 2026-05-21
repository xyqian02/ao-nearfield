# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

自适应光学(AO)近场分布复原仿真系统，使用哈特曼-夏克波前传感器测量子孔径光强，通过极限学习机(ELM)从光强分布反演泽尼克系数，进而重构近场振幅和复原入射波前。

核心创新点：利用 ELM 预测的近场振幅分布，通过"差分斜率"方法消除光强-波前耦合带来的斜率串扰，提升波前复原精度。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 配件预生成（按当前配置生成 subcfg + modes 缓存到 accessories/）
python generate_accessories.py
python generate_accessories.py --show   # 生成并可视化子孔径布局 + 泽尼克模式

# 数据生成（生成训练数据集到 data/）
python main_data_generation.py

# ELM 训练与近场复原
python main_elm_train.py

# 完整仿真（单样本波前复原全流程对比：传统哈特曼 vs 本方法）
python main_simulation.py

# 超参数优化（4种激活函数 × 神经元数网格搜索，多次运行取均值±标准差）
python main_optimization.py

# 泽尼克阶数对 MSE 影响分析
python main_suremax.py

# 批量波前复原测试（三种模式）
python main_batch_test.py rc   # 随机组合：波前和光强均随机变化
python main_batch_test.py fi   # 固定光强：同一光强 + 不同随机波前
python main_batch_test.py fw   # 固定波前：同一波前 + 不同数据集光强
```

没有测试套件。验证修改正确性的方式是运行对应的 main 脚本并检查输出。

## 统一配置系统 (`configs.py`)

所有主脚本通过 `configs.py` 获取配置，集中管理所有参数。

```python
from configs import get_config

cfg = get_config()                    # 默认读取 ./config_exp.json
cfg = get_config("config_sim.json")   # 指定配置文件
```

### 配置文件

| 文件 | 用途 | 典型场景 |
|------|------|----------|
| `config_exp.json` | 实验参数预设 | ELM训练、超参数优化 |
| `config_sim.json` | 仿真参数预设 | 波前复原仿真、批量测试 |
| `config.json` | 临时激活配置 | `python configs.py` 生成，不入库 |

### 关键参数

```python
# === 光学系统 ===
wavelength: float = 635e-6       # 波长 (mm)
focal_length: float = 20.0       # 微透镜焦距 (mm)
pixel_pitch: float = 12.8e-3     # 像元尺寸 (mm/pixel)

# === 探测器靶面 ===
image_size: int = 400            # 全图尺寸 (像素)
beam_size: int = 400             # 光束直径 (像素)
sub_ap_pixels: int = 25          # 子孔径边长 (像素)
n_sub_dim: int = 16              # 子孔径阵列维度
sub_valid_ratio: float = 0.6     # 有效子孔径能量阈值

# === 泽尼克模式 ===
n_modes_total: int = 36          # 预生成的模式总数 (Noll 1 ~ N)
n_amp_modes: int = 35            # 光强(振幅)泽尼克阶数 — ELM 预测目标
n_wf_modes: int = 35             # 波前(相位)泽尼克阶数 — Z2S 响应矩阵行数
wf_skip_count: int = 3           # 波前排除的低阶模式数 (3=去piston+tip+tilt)

# === 数据生成 ===
n_samples: int = 1000            # 样本总数
flag_noise: bool = True          # 是否添加探测器噪声
flag_wf: bool = False            # 是否叠加随机波前扰动

# === ELM ===
activation: str = "softplus"     # 激活函数: sigmoid/relu/softplus/tanh/sin/rbf
n_hidden: int | None = None      # 隐藏层神经元数 (None=自动搜索)
test_ratio: float = 0.1          # 测试集比例
```

**重要**：`n_amp_modes` 和 `n_wf_modes` 是两套独立的泽尼克阶数，分别服务于光强重构和波前传感，不可混淆。

## 核心架构

### 模块依赖

```
main_*.py / generate_accessories.py
    → configs.py
    → src/optics.py        光学计算
    → src/elm.py           ELM 模型
    → src/hartmann.py      哈特曼传感器
    → src/evaluation.py    评估指标
    → src/utils.py         工具函数
```

### 数据布局约定

| 数据 | 形状 | 说明 |
|------|------|------|
| 输入 X / 输出 y | `(n_samples, n_features)` | sklearn 原生布局 |
| 泽尼克模式 modes | `(H, W, n_modes)` | 第3维 = 模式索引，modes[:,:,k] = Noll k+1 阶 |
| 子孔径坐标 subcfg | `(2, n_sub)` | row0=y, row1=x，原点为图像中心 |
| 斜率向量 slopes | `(2*n_sub,)` | 奇偶交错: [Δx₀, Δy₀, Δx₁, Δy₁, ...] |

`build_response_matrix()` 和 `load_accessories()` 会自动检测并转置 `(n_modes, H, W)` 布局。

### 各模块详解

**`src/optics.py`** — 纯光学计算，所有方法为静态或基于构造参数，无外部状态依赖。

| 方法 | 功能 |
|------|------|
| `dl_propagate(field, distance)` | 角谱法衍射传播（FFT → 频域传输函数 → IFFT） |
| `zernike(mode, Na, pupil)` | 单阶泽尼克多项式（Noll 索引方案） |
| `zernike_modes(n_modes, Na)` | 批量生成前 n_modes 阶泽尼克模式 |
| `pupil_circle(N)` | 圆形光瞳函数，二值掩模 |
| `std_beam(N, rec, cir, m)` | 标准近场光束分布（支持矩形+圆形+中心遮拦） |

**`src/elm.py`** — 极限学习机，继承 `sklearn.base.BaseEstimator` + `RegressorMixin`，兼容 `Pipeline` / `GridSearchCV`。

- 输入权重和偏置随机生成（uniform [-1, 1]）
- 输出权重通过伪逆（`np.linalg.pinv`）解析求解，无需反向传播
- 支持 L2 正则化（`alpha > 0` 时使用岭回归 `np.linalg.solve`）
- 6 种激活函数：`sigmoid`, `relu`, `softplus`, `tanh`, `sin`, `rbf`
- `fit(X, y)` 接受 `(n_samples, n_features)`，内部转置运算

**`src/hartmann.py`** — 哈特曼-夏克波前传感器。

自由函数：
- `create_sub_valid(n_dim, near_field, n_sub_pix, ratio)` — 有效子孔径判定，返回布尔矩阵
- `create_subcfg(sub_valid_mat, n_pixel, n_sub_pix)` — 坐标矩阵生成，原点对齐图像中心

`HartmannSensor` 类工作流：
```
calibrate(mask)           → 平面波标定，记录参考质心 origin_hs
build_response_matrix()   → 逐阶泽尼克模式 → Z2S 斜率响应矩阵
measure_slopes(field)     → 入射场 → 各子孔径斜率向量
reconstruct(slopes, R)    → 斜率 @ 复原矩阵 → 泽尼克系数
```

**`src/evaluation.py`** — 评估指标，兼容 `(n_samples, n_outputs)` 和 `(n_outputs, n_samples)` 两种布局。

- `compute_mse(y_true, y_pred)` / `compute_rmse` / `compute_r2` — 回归指标
- `compute_rms_wavefront(wf_pred, wf_true, mask)` — 波前残差 RMS（单位 λ）
- `compute_pv_wavefront(wf_pred, wf_true, mask)` — 波前残差 PV（单位 λ）

**`src/utils.py`** — 工具集。

数据 IO：
- `load_mat(path)` / `save_mat(path, **kv)` — .mat 文件读写

配件管线：
- `generate_mask(cfg)` → 圆形光瞳
- `generate_modes(cfg)` → 泽尼克模式矩阵 `(H, W, n_modes_total)`
- `generate_subcfg(cfg)` → 子孔径坐标 `(2, n_valid_sub)`
- `load_accessories(cfg)` → 自动加载或生成 (subcfg, modes)
- `save_accessories(cfg)` → 缓存为 .mat

泽尼克重构：
- `reconstruct_from_zernike(coeffs, modes, start_order=1, end_order=None)` — `np.tensordot` 向量化重构，`start_order`/`end_order` 为 1-based Noll 索引（含）。`reconstruct_from_zernike(coe, modes)` 使用全部阶数。

其他：
- `set_seed(seed)` — NumPy 随机种子
- `configure_chinese_font()` — matplotlib 中文字体自动检测
- `get_data_filename(cfg)` — 根据参数生成数据文件后缀

### 配件管线

`load_accessories(cfg)` 统一管理 subcfg + modes 的加载：

1. 优先从 `accessories/` 目录的 .mat 缓存加载（文件名含关键配置参数，不同配置不会混淆）
2. 缓存不存在时自动生成并保存
3. 也可预生成：`python generate_accessories.py`

## 关键约定与陷阱

### 运行顺序

数据生成 → ELM 训练/仿真/批量测试。后三者依赖 `data/` 和 `accessories/` 目录下的 .mat 文件。

### 波前低阶模式排除

波前复原中通过 `wf_skip_count`（默认 3）排除 piston + tip + tilt（Noll 1~3）。排除方式：生成全阶系数后，将前 `wf_skip_count` 个系数置零，再通过 `reconstruct_from_zernike(coe, modes)` 全阶重构。数据生成、单样本仿真、批量测试三处统一采用此方式。

光强重构**不**排除低阶 —— `wf_skip_count` 仅影响波前相关代码。

### 数据文件

`accessories/`、`data/`、`result/` 被 .gitignore 排除。首次运行需先执行 `main_data_generation.py` 生成数据，或放置预生成的 .mat 文件。

### 中文字体

所有绘图脚本须在 `plt.show()` 前调用 `configure_chinese_font()`，否则图表中文可能显示为方框。

### ELM 可复现性

- 设置相同 `random_state` 可保证 ELM 权重可复现
- 不同 PRNG 实现（如 NumPy vs 其他库）会导致略有不同的随机权重和训练结果，属正常现象
- 数据生成阶段（噪声序列、随机系数）也受随机种子控制
