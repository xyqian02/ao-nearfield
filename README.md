# 自适应光学 (AO) 近场分布复原仿真系统

基于哈特曼-夏克波前传感器和极限学习机 (ELM) 的近场振幅分布与波前复原仿真平台，使用 Python 从 MATLAB 原版重构。

## 目录结构

```
Dnunf/
│
├── main_data_generation.py        # 数据生成：泽尼克系数 → 哈特曼子孔径光强
├── main_elm_train.py              # ELM训练 + 近场复原 + 可视化
├── main_simulation.py             # 完整仿真：哈特曼标定 + 波前复原方法对比
├── main_optimization.py           # 超参数优化：不同激活函数 × 神经元数
├── main_suremax.py                # 泽尼克阶数对MSE的影响分析
├── main_batch_test.py             # 批量随机波前测试（传统方法 vs 本方法）
├── main_experiment.py             # 实验数据处理：动态哈特曼光斑阵列生成
│
├── src/                           # 核心功能模块
│   ├── __init__.py                # 模块入口，导出核心类
│   ├── optics.py                  # 光学计算（衍射传播、泽尼克、光瞳）
│   ├── elm.py                     # ELM极限学习机 + 神经网络抽象基类
│   ├── hartmann.py                # 哈特曼波前传感器
│   └── utils.py                   # 数据IO、归一化、评估工具
│
├── accessories/                   # 辅助数据（.mat 格式）
│   ├── Subcfg.mat                 # 子孔径配置矩阵 (112子孔径坐标)
│   ├── modes.mat                  # 泽尼克模式矩阵
│   └── modes250.mat               # 泽尼克模式矩阵 (250阶, 240×240)
│
├── data/                          # 仿真生成数据目录
├── result/                        # 运行结果目录
├── figure_data/                   # 图表数据目录
├── requirements.txt               # Python 依赖
└── README.md                      # 本文档
```

## 环境要求

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.7
- scikit-learn >= 1.2

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 仿真流程总览

整个仿真系统包含三个核心阶段：

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  阶段1: 数据生成      │ ──→ │  阶段2: ELM 训练      │ ──→ │  阶段3: 波前复原对比   │
│                      │     │                      │     │                      │
│  随机泽尼克系数        │     │  子孔径强度 → ELM     │     │  哈特曼标定            │
│  → 近场振幅分布        │     │  → 泽尼克系数预测      │     │  真实光强 → 传统方法    │
│  → 波前扰动          │     │  → 近场复原           │     │  ELM光强 → 差分方法    │
│  → DL衍射 → 子孔径光强 │     │  → 可视化评估         │     │  → RMS/PV 残差对比     │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### 2. 运行各阶段

**生成仿真数据**（先运行此步骤，或在 MATLAB 中已生成则可跳过）：

```bash
python main_data_generation.py
```

**ELM 训练与近场复原**：

```bash
python main_elm_train.py
```

**完整仿真（波前复原对比）**：

```bash
python main_simulation.py
```

**超参数优化（多激活函数对比）**：

```bash
python main_optimization.py
```

## 模块说明

### `src/optics.py` — 光学计算核心

| 类/方法 | 对应 MATLAB | 功能 |
|---------|------------|------|
| `Optics` | — | 统一管理光学参数 (λ, f, pixel_pitch) |
| `Optics.dl_propagate()` | `DL.m` | 角谱法衍射传播 |
| `Optics.zernike()` | `Zernike.m` | 生成单阶泽尼克多项式 (Noll索引) |
| `Optics.zernike_modes()` | — | 批量生成多阶泽尼克模式 |
| `Optics.pupil_circle()` | `Pupil.m` | 圆形光瞳函数 |
| `Optics.std_beam()` | `StdBeamFunc.m` | 标准光束/环形域生成 |

使用示例：

```python
from src.optics import Optics

# 初始化光学系统（参数可随时替换）
optics = Optics(wavelength=1.064e-3, focal_length=21.0, pixel_pitch=14e-3)

# 衍射传播
output_field = optics.dl_propagate(input_field)

# 生成泽尼克模式
mode_5 = Optics.zernike(5, 240)  # 第5阶, 240×240
modes = Optics.zernike_modes(25, 240)  # 前25阶, 形状 (25, 240, 240)
```

### `src/elm.py` — 极限学习机

| 类 | 对应 MATLAB | 功能 |
|---|------------|------|
| `BaseModel` | — | 神经网络抽象基类 (fit/predict 统一接口) |
| `ELM` | `elmtrain.m`, `elmpredict.m` | 极限学习机实现 |

支持的激活函数：`sigmoid`, `relu`, `softplus`, `tanh`, `sin`, `rbf`

使用示例：

```python
from src.elm import ELM

# 创建 ELM 模型
elm = ELM(n_hidden=850, activation='softplus')

# 训练（数据格式: (n_features, n_samples)）
elm.fit(X_train, y_train)

# 预测
y_pred = elm.predict(X_test)

# 后续替换为其他网络（U-Net、Transformer 等）只需继承 BaseModel
class MyUNet(BaseModel):
    def fit(self, X, y): ...
    def predict(self, X): ...
```

### `src/hartmann.py` — 哈特曼波前传感器

| 类/函数 | 对应 MATLAB | 功能 |
|---------|------------|------|
| `HartmannSensor` | Main 脚本中的波前传感逻辑 | 完整的哈特曼传感器 |
| `HartmannSensor.calibrate()` | 平面波标定部分 | 计算各子孔径质心参考原点 |
| `HartmannSensor.measure_slopes()` | 斜率测量部分 | 测量子孔径x/y方向波前斜率 |
| `HartmannSensor.build_response_matrix()` | Z2S 构建部分 | 构建斜率响应矩阵 |
| `HartmannSensor.reconstruct()` | 波前复原部分 | 从斜率反演泽尼克系数 |
| `create_subcfg()` | `CrtSubcfg.m` | 计算子孔径坐标 |
| `create_sub_valid()` | `CrtSubValid.m` | 判定有效子孔径 |

使用示例：

```python
from src.optics import Optics
from src.hartmann import HartmannSensor

optics = Optics(wavelength=1.064e-3, focal_length=12.0, pixel_pitch=14e-3)
hs = HartmannSensor(subcfg, optics)

# 1. 标定
hs.calibrate(mask, noise_sigma=0.5)

# 2. 构建响应矩阵
Z2S = hs.build_response_matrix(modes, mask, n_wf_modes=15, noise_sigma=0.5)

# 3. 波前复原
Recon = np.linalg.pinv(Z2S)
slopes, _ = hs.measure_slopes(field, noise_sigma=0.5)
coeffs = hs.reconstruct(slopes, Recon)
```

### `src/utils.py` — 工具函数

| 函数 | 替代方案 | 功能 |
|------|---------|------|
| `load_mat()` / `save_mat()` | `scipy.io` | .mat 文件读写 |
| `normalize_data()` | `sklearn.MinMaxScaler` | 归一化到[-1,1] |
| `compute_mse()` | `sklearn.metrics.mean_squared_error` | 批量MSE计算 |
| `split_data()` | `sklearn.model_selection.train_test_split` | 训练/测试划分 |
| `set_seed()` | `np.random.seed` | 设置随机种子 |
| `circ_mask()` | — | 单位圆掩模 |

## 关键参数说明

每个主脚本顶部都有一个 `Config` dataclass，所有可调参数集中在此：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `wavelength` | 1.064e-3 mm | 激光波长 |
| `focal_length` | 21.0 / 12.0 mm | 微透镜焦距 |
| `pixel_pitch` | 14e-3 mm | 相机像元尺寸 |
| `sub_ap_pixels` | 20 | 子孔径边长 (像素) |
| `image_size` | 256 | 图像尺寸 (像素) |
| `n_zernike` | 25 | 近场振幅泽尼克阶数 |
| `n_wf_modes` | 15 | 波前传感泽尼克阶数 |
| `n_samples` | 10000 | 训练样本总数 |
| `n_hidden` | 850 | ELM 隐藏层神经元数 |
| `activation` | "softplus" | ELM 激活函数 |
| `noise_sigma` | 0.5 | 相机噪声 RMS |

## MATLAB ↔ Python 关键差异

| 项目 | MATLAB | Python |
|------|--------|--------|
| 索引 | 1-based | 0-based |
| 数组存储 | 列优先 (Fortran) | 行优先 (C) |
| .mat 读写 | `load`/`save` | `scipy.io.loadmat`/`savemat` |
| FFT | `fft2`/`ifft2` | `numpy.fft.fft2`/`ifft2` |
| 伪逆 | `pinv()` | `numpy.linalg.pinv()` |
| 归一化 | `mapminmax` | `sklearn.preprocessing.MinMaxScaler` |
| MSE | `mse()` | `sklearn.metrics.mean_squared_error` |
| 索引提取 | `A(y1:y1+19, x1:x1+19)` | `A[y1-1:y1+19, x1-1:x1+19]` |

**维度注意事项**：
- .mat 文件加载后保持原始 MATLAB 形状（无需转置）
- `modes` 在 MATLAB 中 `modes(:,:,k)` 对应 Python 中 `modes[:,:,k-1]`
- `Subcfg` 坐标以图像中心为原点，Python 中提取区域时需 `int(coord + image_size/2) - 1` 转为 0-based 索引

## 结果输出

所有脚本的运行结果（图表）保存在 `result/` 目录，包括：
- 泽尼克系数预测对比图
- 近场分布 2D/3D 对比图
- 波前复原残差图 (RMS / PV)
- 超参数搜索曲线
- 批量测试 RMS 统计图
- 哈特曼光斑阵列图

## 算法参考文献

1. Huang, G. B., Zhu, Q. Y., & Siew, C. K. (2006). Extreme Learning Machine: Theory and Applications. *Neurocomputing*, 70(1-3), 489-501.
2. Noll, R. J. (1976). Zernike Polynomials and Atmospheric Turbulence. *Journal of the Optical Society of America*, 66(3), 207-211.
3. Goodman, J. W. (2017). *Introduction to Fourier Optics* (4th ed.). W. H. Freeman.
