# 实验数据 ELM 预测与波前复原 — 详细说明

## 1. 概述

`main_experiment.py` 对实验采集的哈特曼-夏克波前传感器光斑阵列图像进行处理，使用极限学习机 (ELM) 从子孔径总光强预测光强泽尼克系数，进而通过**差分斜率法**分离光强-波前耦合，实现高精度波前复原。

**核心创新**：传统哈特曼传感器假设近场光强均匀分布，直接由质心斜率反演波前。实际系统中近场光强不均匀（激光束轮廓、光学元件缺陷等），导致光强-波前耦合串扰。本方法用 ELM 从子孔径总光强预测近场振幅分布，仿真计算"纯振幅诱导斜率"，从测量斜率中扣除后得到"纯波前斜率"，消除耦合误差。

## 2. 实验数据

### 2.1 数据来源

| 文件 | 说明 |
|------|------|
| `data/experimental_data/250616/0000000.bmp` | 标定图像（平面波入射，无波前像差） |
| `data/experimental_data/250616/0000001~0000500.bmp` | 500 张波前扰动图像，每 100 张对应同一组 100 个波前条件（共 5 组重复测量） |
| `data/experimental_data/f1000d_r012.mat` | 波前真值 `totalProj` (36, 100)：36 阶泽尼克系数 × 100 组 |

### 2.2 图像参数

| 参数 | 值 |
|------|-----|
| 原始分辨率 | 512×512 RGB |
| 光斑阵列中心 | (x_center=272, y_center=251) — MATLAB 1-based |
| 裁剪尺寸 | 400×400 |
| 子孔径网格 | 16×16 @ 25 px/子孔径 |
| 有效子孔径 | ~202 个（圆形光瞳内，重叠率 > 60%） |
| 波长 | 635 nm |
| 微透镜焦距 | 20 mm |
| 像元尺寸 | 12.8 μm |

## 3. 实验图像处理管线

### 3.1 预处理 (`preprocess_experimental_image`)

```
BMP 图像 (512×512 RGB)
    │
    ▼ Image.open().convert("L")
灰度图像 (512×512 uint8)
    │
    ▼ np.float64, np.maximum(img - 10, 0)
去基底/去噪 (threshold=10)
    │
    ▼ img[72:472, 51:451]  (Python 0-based)
裁剪光斑阵列区域 (400×400)
  - MATLAB 等价: F(73:472, 52:451) 1-based
  - row_start = x_center - 200 = 272 - 200 = 72
  - col_start = y_center - 200 = 251 - 200 = 51
    │
    ▼ np.fliplr(img)
左右翻转 (与 MATLAB fliplr 一致)
    │
    ▼
处理后图像 (400×400 float64)
```

**MATLAB 参考**：`SHackHartmann_oneFrame.m` 第 16-57 行。

### 3.2 子孔径提取（坐标约定）

子孔径坐标矩阵 `subcfg` (2, n_sub)，以图像中心为原点：
- `subcfg[0, i]`：第 i 个子孔径左下角的 y 坐标（相对图像中心）
- `subcfg[1, i]`：第 i 个子孔径左下角的 x 坐标（相对图像中心）

像素索引转换：
```python
y_px = int(round(subcfg[0, i] + image_size / 2))  # Python 0-based
x_px = int(round(subcfg[1, i] + image_size / 2))
```

16×16 规则网格，25px 间距。例如：
- 网格行 0（图像顶部）：y_px = 0，覆盖像素行 0~24
- 网格行 1：y_px = 25，覆盖像素行 25~49
- ...
- 网格行 15（图像底部）：y_px = 375，覆盖像素行 375~399

这与 MATLAB 1-based 索引 `Row_s = (nRow-1)*25 + 1` 完全对齐。

### 3.3 质心提取 (`extract_sub_spot_centroids`)

从光斑图像直接提取每个有效子孔径的焦斑质心（不通过衍射仿真）：

```python
X_g, Y_g = np.meshgrid(arange(25), arange(25))  # 0~24 坐标网格
sub_spot = spot_image[y:y+25, x:x+25]
cx = sum(sub_spot * X_g) / sum(sub_spot)
cy = sum(sub_spot * Y_g) / sum(sub_spot)
```

### 3.4 总光强提取 (`extract_sub_ap_total_intensity`)

```python
intensity[i] = sum(spot_image[y:y+25, x:x+25])
```

每个子孔径 25×25 像素区域的总和作为该子孔径的光强特征值。这决定了 ELM 的输入维度 = n_sub（有效子孔径数，~202）。

## 4. 波前复原流程

### 4.1 整体流程图

```
                        ┌──────────────────────────┐
                        │  实验哈特曼光斑图像        │
                        │  (已包含物理衍射结果)      │
                        └────────────┬─────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
              ┌──────────┐   ┌────────────┐   ┌──────────────┐
              │ 标定图像  │   │ 测试图像    │   │ 测试图像      │
              │ (0号)    │   │ 质心提取    │   │ 子孔径总光强  │
              └────┬─────┘   └─────┬──────┘   └──────┬───────┘
                   │               │                  │
                   ▼               ▼                  ▼
            ┌────────────┐  ┌───────────┐    ┌───────────────┐
            │ 标定质心    │  │ 测量质心   │    │ 总光强向量     │
            │ calib_cent  │  │ test_cent  │    │ I (n_sub,)    │
            └─────┬──────┘  └─────┬─────┘    └──────┬────────┘
                  │               │                  │
                  │        ┌──────▼──────┐    ┌──────▼────────┐
                  │        │ measured_   │    │ ELM 预测       │
                  │        │ slopes      │    │ 光强泽尼克系数  │
                  │        │ = test -    │    │ coeff_amp      │
                  │        │   calib     │    └──────┬────────┘
                  │        └──────┬──────┘           │
                  │               │                  ▼
                  │               │          ┌───────────────┐
                  │               │          │ 重构近场振幅   │
                  │               │          │ A_pred =      │
                  │               │          │ sum(coeff *   │
                  │               │          │  modes) + bias│
                  │               │          └──────┬────────┘
                  │               │                 │
                  │               │                 ▼
                  │               │          ┌───────────────┐
                  │               │          │ 仿真衍射传播   │
                  │               │          │ hs.measure_   │
                  │               │          │ slopes(A_pred)│
                  │               │          └──────┬────────┘
                  │               │                 │
                  │               │                 ▼
                  │               │          ┌───────────────┐
                  │               │          │ intensity_    │
                  │               │          │ slopes        │
                  │               │          │ (纯振幅诱导)   │
                  │               │          └──────┬────────┘
                  │               │                 │
                  │        ┌──────▼─────────────────▼──────┐
                  │        │        斜率分离                │
                  │        │  wf_slopes = measured_slopes  │
                  │        │            - intensity_slopes  │
                  │        └──────────────┬────────────────┘
                  │                       │
                  ▼                       ▼
           ┌─────────────┐        ┌─────────────┐
           │ 传统方法     │        │ 本方法       │
           │ Recon @      │        │ Recon @      │
           │ measured_    │        │ wf_slopes    │
           │ slopes       │        │              │
           └──────┬──────┘        └──────┬───────┘
                  │                      │
                  ▼                      ▼
           ┌─────────────┐        ┌─────────────┐
           │ recoe_trad   │        │ recoe_prop   │
           │ 波前系数      │        │ 波前系数      │
           └──────┬──────┘        └──────┬───────┘
                  │                      │
                  ▼                      ▼
           ┌─────────────┐        ┌─────────────┐
           │ wf_trad 重构  │        │ wf_prop 重构  │
           └──────┬──────┘        └──────┬───────┘
                  │                      │
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ 与真值比较        │
                    │ RMS/PV 残差评估   │
                    └─────────────────┘
```

### 4.2 传统方法

```
measured_slopes → Recon(=pinv(Z2S)) → recoe_trad → reconstruct(wf)
```

直接对测量斜率做伪逆求解，假设光强均匀、斜率仅由波前引起。

### 4.3 本方法（差分斜率法）

```
step 1: ELM(I) → coeff_amp_pred              # 从总光强预测振幅系数
step 2: A_pred = reconstruct(coeff_amp_pred)  # 重构近场振幅
step 3: intensity_slopes = simulate(A_pred)   # 仿真纯振幅引起的斜率
step 4: wf_slopes = measured_slopes - intensity_slopes  # 扣除振幅贡献
step 5: recoe_prop = Recon @ wf_slopes        # 复原纯波前系数
```

关键：步骤 3 使用与训练数据相同的衍射仿真参数，输入为纯振幅（无波前相位），输出即为"如果只有振幅不均匀会产生多少斜率"。

### 4.4 Z2S 响应矩阵

Z2S 通过仿真构建：对每一阶泽尼克模式（波前相位），计算其在各子孔径处产生的斜率偏移。

```
for k in range(n_wf_modes):
    wf = modes[:,:,k]           # 第 k 阶泽尼克波前
    field = mask * exp(-1j*wf)  # 平面波 + 该阶波前
    slopes_k = measure(field) - origin  # 该阶引起的斜率
    Z2S[k, :] = slopes_k
```

Z2S 形状: (n_wf_modes, 2*n_sub)。Recon = pinv(Z2S) 形状: (2*n_sub, n_wf_modes)。

### 4.5 波前低阶模式排除

波前复原中排除 piston + tip + tilt (Noll 1~3)，因为这些模式对应整体相位偏移和光束指向抖动，不影响波前像差评估：

```python
recoe[:cfg.wf_skip_count] = 0.0  # wf_skip_count=3
```

该排除仅作用于波前复原阶段，光强重构不受影响。

### 4.6 RMS/PV 计算

```python
rms = std(wf_pred[mask==1] - wf_true[mask==1]) / (2π)   # 单位: λ
pv  = (max(diff) - min(diff)) / (2π)                      # 单位: λ
```

仅计算圆形光瞳 mask 内的有效区域。

## 5. ELM 模型

### 5.1 训练数据

ELM 输入：子孔径总光强向量 (n_sub,)
ELM 输出：光强泽尼克系数 (n_amp + 1,)，前 n_amp 个为系数，最后一个为偏移量

数据由 `generate_simulation_data()` (`src/utils.py`) 统一生成，`main_data_generation.py` 和 `main_experiment.py` 共用同一函数。

生成的数据自动保存到 `data/` 目录：
- 标准模式：`data/InputData_s{image_size}_a{n_amp}.mat`
- 增强模式：`data/InputData_s{image_size}_a{n_amp}_enh.mat`

后续运行自动检测并加载缓存，避免重复生成。其他主脚本（`main_simulation.py`、`main_batch_test.py`）也可直接使用这些数据文件。

### 5.2 模型架构

```
输入层 (n_sub) → 隐藏层 (N 个神经元，随机权重+偏置) → 输出层 (n_amp+1)
                                  │
                            activation (softplus/relu/sigmoid/tanh/sin/rbf)
                                  │
                            输出权重 = pinv(H) @ Y  (解析求解，无需反向传播)
```

### 5.3 模型持久化

```python
elm.save("models/elm_xxx.npz")    # 保存 IW_, B_, LW_, 超参数
elm = ELM.load("models/elm_xxx.npz")  # 恢复模型

joblib.dump(scaler_X, "models/scaler_X_xxx.joblib")  # 保存归一化器
joblib.dump(scaler_Y, "models/scaler_Y_xxx.joblib")
```

## 6. 域偏移优化策略

仿真训练数据与实验测试数据之间存在分布差异（域偏移），提供 4 种可选策略缓解。

### 6.1 策略 1：标定图像归一化 (`-n` / `--normalize`)

**原理**：标定图像（平面波）记录了系统固有的微透镜透过率差异、探测器 PRNU 等固定图案。用标定图像对每个子孔径做除法归一化，消除固定增益差异。

```
I_norm[i] = I_test[i] / I_calib[i]
```

仿真侧同理：生成"标定样本"（均匀近场 + 平面波），所有训练数据除以该参考值。

**启用效果**：消除子孔径间固定增益差异，提高仿真→实验的特征一致性。

### 6.2 策略 2：增强仿真训练数据 (`-e` / `--enhanced-data`)

**原理**：标准仿真用 N(0,1) 泽尼克系数和简单偏移，分布单一。增强模式引入三种随机变化，扩大训练分布覆盖范围：

1. **超高斯包络**：`envelope = exp(-(r/w)^n)`，w∈[0.6, 1.0]，n∈[2, 8]
   - 模拟真实激光束的非均匀轮廓（平顶、超高斯等）
2. **变系数方差**：`sigma ∈ [0.5, 2.0]`
   - 模拟不同强度的振幅扰动
3. **随机偏置**：`Ampl += bias ∈ [0, 0.2*max(Ampl)]`
   - 模拟不同曝光/增益条件

**启用效果**：ELM 学习到更鲁棒的映射，对实验数据中未见过的振幅分布泛化更好。

### 6.3 策略 3：比强度特征 (`-r` / `--ratio-features`)

**原理**：用 `I_i / mean(I)` 替代原始 `I_i` 作为 ELM 输入。子孔径总光强的绝对尺度受曝光时间、激光功率、探测器增益影响，其相对分布才是振幅信息的载体。

```
I_ratio[i] = I[i] / mean(I)
```

**启用效果**：对全局亮度变化（曝光、增益波动）完全不变。训练和测试都在"比强度空间"进行，消除尺度差异。

**注意**：`--ratio-features` 同时应用于仿真训练数据和实验测试数据。训练时即使用比强度特征训练 ELM。

### 6.4 策略 4：波前系数后校准 (`-p` / `--post-calibrate`)

**原理**：仿真 Z2S 与真实物理 Z2S 之间存在系统性的增益偏差。MATLAB 参考代码中有一个硬编码的 `0.78` 比例因子。本方法用前 N 组（默认 20）实验样本的波前真值，通过 Ridge 回归自动学习校正矩阵。

```python
# 对前 n_calib 个样本，收集复原波前系数和真值
A, b = Ridge(alpha=0.1).fit(reco_prop[:n_calib], true_coeffs[:n_calib])

# 对剩余样本应用校正
reco_corrected = A @ reco_prop + b
```

**启用效果**：自动补偿仿真-实验的系统增益偏差，等效于学习 MATLAB 的 `0.78` 因子但更灵活（允许模式间不同的缩放）。

**注意**：校准样本不参与最终评估（数据泄漏防护）。

## 7. 命令行参数

```
用法: python main_experiment.py [选项]

实验模式选项:
  -n, --normalize         标定图像归一化 (策略1)
  -e, --enhanced-data     增强仿真训练数据 (策略2)
  -r, --ratio-features    比强度特征 (策略3)
  -p, --post-calibrate    波前系数后校准 (策略4)
  --n-calib N             校准样本数 (配合 -p，默认 20)
  --n-test N              测试样本数 (默认 100)
  --n-samples N           仿真训练样本数 (默认 5000)
  --model-dir DIR         模型保存目录 (默认 models/)
  --exp-dir DIR           实验数据目录
  --truth-file FILE       波前真值 .mat 文件

可视化模式选项:
  --show N                可视化第 N 帧实验图像 (0=标定帧, 1~N=测试帧)
  --animate M-N           生成第 M 到 N 帧的连续动画 (如 --animate 1-50)
  --fps N                 动画帧率 (配合 --animate，默认 10)

示例:
  python main_experiment.py                                    # 基础运行
  python main_experiment.py -n -r                              # 归一化 + 比强度
  python main_experiment.py -e -p --n-calib 30                 # 增强数据 + 后校准
  python main_experiment.py -n -e -r -p                        # 全部启用
  python main_experiment.py --show 0                           # 查看标定帧
  python main_experiment.py --show 5                           # 查看第5帧测试图像
  python main_experiment.py --animate 1-100 --fps 10           # 生成前100帧动画
```

## 8. 输出文件

所有结果保存在 `result/experiment/`：

| 文件 | 内容 |
|------|------|
| `rms_pv_comparison.png` | RMS/PV 散点图（传统 vs 本方法），含均值和提升幅度 |
| `wavefront_sample_000.png` ~ `002.png` | 前 3 个样本的波前图对比（真值/传统/本方法） |
| `results.npz` | 定量结果数组（rms_trad, rms_prop, pv_trad, pv_prop 等） |

## 9. 与 MATLAB 参考代码的差异

| 项目 | MATLAB (`SHackHartmann_oneFrame.m`) | Python (`main_experiment.py`) |
|------|-------------------------------------|-------------------------------|
| 波前复原 | 仅传统方法（斜率→pinv→系数） | 传统 + 本方法（差分斜率）对比 |
| 光强处理 | 不做光强预测 | ELM 预测光强系数 → 差分斜率 |
| 比例因子 | 硬编码 `0.78` | 可选 `-p` 自动学习 |
| 批量处理 | 单帧手动 | 批量自动化，支持 100+ 帧 |
| Z2S 构建 | `HartmannImage_calc` 外部函数 | `optics.dl_propagate` 角谱法 |
| 泽尼克模式 | Noll 2~36 (35 阶) | Noll 1~35 (35 阶)，零化 1~3 |
| 有效阶数 | 33 阶 (零化后) | 32 阶 (零化后) |

## 10. 运行前置条件

```bash
# 1. 生成配件 (subcfg + modes)
python generate_accessories.py

# 2. 生成仿真训练数据 (若使用标准模式)
python main_data_generation.py --config config_exp.json

# 3. 运行实验预测
python main_experiment.py --n-samples 5000 --n-test 100

# 4. 全优化运行
python main_experiment.py -n -e -r -p --n-test 100
```

## 11. 关键配置参数

来自 `config_exp.json`，需与实验系统匹配：

```python
wavelength = 635e-6       # mm, 635nm
focal_length = 20.0       # mm, 微透镜焦距
pixel_pitch = 12.8e-3     # mm/pixel, 12.8μm
image_size = 400          # 像素
beam_size = 400           # 像素
sub_ap_pixels = 25        # 像素/子孔径
n_sub_dim = 16            # 子孔径阵列维度
n_amp_modes = 25          # 光强泽尼克阶数 (ELM 输出)
n_wf_modes = 35           # 波前泽尼克阶数 (Z2S 行数)
wf_skip_count = 3         # 排除低阶模式数 (piston+tip+tilt)
flag_wf = True            # 训练时叠加波前扰动
flag_noise = True         # 训练时添加探测器噪声
wf_decay_scheme = "power_law"  # 波前系数衰减方案: "none"|"power_law"|"kolmogorov"
wf_decay_exponent = 1.6   # power_law 指数 (仅power_law方案)
```

### 11.1 波前系数衰减模型

真实光学系统中波前泽尼克系数随阶数增大而衰减（低阶占比大，高阶占比小）。

通过实验数据 (`f1000d_r012.mat`, 36阶×100样本) 分析，系数标准差与径向阶数满足幂律关系：`std ∝ n^(-1.61)`，即 `variance ∝ n^(-3.21)`。

衰减方案：

| 方案 | 公式 | 说明 |
|------|------|------|
| `"none"` | weight = 1.0 | 无衰减（旧版行为） |
| `"power_law"` | `weight = 1/(n+0.5)^exponent` | 可调幂律，默认 exponent=1.6 匹配实验数据 |
| `"kolmogorov"` | `weight = 1/(n+1)^(4/3)` | Kolmogorov 大气湍流近似 (Noll 1976) |

其中 n 为 Noll 径向阶数（n=0=piston, n=1=tip/tilt, n=2=defocus+astig, ...）。

衰减权重应用于所有生成随机波前系数的位置（数据生成、仿真、批量测试）。

## 12. 哈特曼阵列可视化

### 12.1 三图对比 (`--show N`)

在同一张画布 (1×3) 上并列对比三幅哈特曼光斑图像：

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  实验图像             │  仿真全场             │  仿真纯振幅           │
│  (预处理后)           │  A_pred ×            │  A_pred              │
│                      │  exp(−i·WF_true)     │  (无波前相位)         │
│  +网格 +光瞳圆        │  +网格 +光瞳圆        │  +网格 +光瞳圆        │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

- **左图**：实验采集的哈特曼光斑，经预处理（灰度→减阈值→裁剪→翻转）
- **中图**：用 ELM 预测的振幅 × 真值波前 → 衍射传播 → 仿真光斑（完整模型估计）
- **右图**：仅用 ELM 预测的振幅 → 衍射传播 → 仿真光斑（纯振幅贡献，无波前）

对比逻辑：
- 左 vs 中 → 验证 ELM+衍射模型的准确性
- 中 vs 右 → 直观展示波前对光斑图案的影响

所有子图叠加：
- 红色细线矩形：有效子孔径边界
- 白色虚线圆：圆形光瞳边界 (`beam_size/2` 半径)

需要已训练的 ELM 模型。输出 `result/experiment/hartmann_compare_NNN.png`。

### 12.2 连续帧动画 (`--animate M-N`)

生成第 M 到第 N 帧实验图像的连续动画，叠加子孔径网格，观察光斑图案动态变化。

- 优先输出 MP4（需 FFmpeg），不可用时回退 GIF
- 通过 `--fps` 控制帧率

输出：`result/experiment/hartmann_anim_M-N.mp4` 或 `.gif`

### 12.3 实现细节

`plot_hartmann_grid()` (`src/utils.py`)：
1. `imshow` 显示光斑图像（jet colormap）
2. 遍历有效子孔径，`plt.Rectangle` 绘制红色边框
3. `plt.Circle` (白色虚线) 绘制光瞳边界（`pupil_radius=beam_size/2`）
4. 子孔径位置：`subcfg[0,i] + image_size/2` → 像素坐标

仿真图像生成：
```python
hs.calibrate(mask)
_, sim_full = hs.measure_slopes(A_pred * np.exp(-1j * true_wf))  # 全场
_, sim_amp  = hs.measure_slopes(A_pred)                            # 纯振幅
```
`measure_slopes()` 内部对每个子孔径做衍射传播，将焦斑填入 `img_mat` 画布。```
