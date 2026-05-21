"""
工具函数模块

提供数据IO（.mat文件读写）、随机种子设置、中文字体配置、
泽尼克重构、配件生成（subcfg / modes / mask）等基础功能。
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

from .optics import Optics
from .hartmann import create_sub_valid, create_subcfg, HartmannSensor


def configure_chinese_font() -> None:
    """
    配置 matplotlib 中文显示支持

    自动检测系统中可用的中文字体并设为默认，
    同时修复负号显示问题。在所有绘图脚本开头调用一次即可。
    """
    # Windows 系统下常见的中文字体优先级列表
    _CN_FONT_CANDIDATES = [
        "Microsoft YaHei",   # 微软雅黑 (Windows 优先)
        "SimHei",            # 黑体
        "SimSun",            # 宋体
        "KaiTi",             # 楷体
        "FangSong",          # 仿宋
        "WenQuanYi Micro Hei",  # Linux
        "Noto Sans CJK SC",  # Linux / 跨平台
        "PingFang SC",       # macOS
        "Heiti SC",          # macOS
        "Arial Unicode MS",  # 通用回退
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    selected_font = None
    for font_name in _CN_FONT_CANDIDATES:
        if font_name in available_fonts:
            selected_font = font_name
            break

    if selected_font is not None:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected_font] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False  # 修复负号 '-' 显示为方块
        print(f"[字体] 已启用中文字体: {selected_font}")
    else:
        print("[字体] 警告: 未找到中文字体，图表中的中文可能无法正常显示")
        print("        建议安装中文字体或将图表标签改为英文")
        plt.rcParams["axes.unicode_minus"] = False


def load_mat(filepath: str, key: str | None = None) -> np.ndarray:
    """
    加载 MATLAB .mat 文件

    参数:
        filepath: .mat 文件路径
        key: 指定变量名。若为 None，自动获取第一个非元数据变量

    返回:
        numpy 数组，保持 MATLAB 中的维度排列
    """
    data = sio.loadmat(filepath)
    if key is not None:
        return data[key]
    # 自动获取第一个非系统变量
    for k, v in data.items():
        if not k.startswith("__"):
            return v
    raise KeyError(f"未在 {filepath} 中找到有效变量")


def save_mat(filepath: str, **kwargs) -> None:
    """
    保存数据为 MATLAB .mat 文件

    参数:
        filepath: 保存路径
        **kwargs: 变量名=数据 的键值对
    """
    sio.savemat(filepath, kwargs)


def set_seed(seed: int = 42) -> None:
    """设置 NumPy 随机种子，确保可重复性"""
    np.random.seed(seed)


def circ_mask(size: int = 240) -> np.ndarray:
    """
    创建圆形掩模（单位圆内为1，外为0）

    参数:
        size: 掩模尺寸 (size × size)

    返回:
        二维圆形掩模数组
    """
    X, Y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    mask = np.ones((size, size))
    mask[np.hypot(X, Y) > 1] = 0.0
    return mask


def get_data_filename(
    cfg,
    flag_noise: bool | None = None,
    flag_wf: bool | None = None,
) -> str:
    """
    根据配置生成数据文件名后缀，包含图像尺寸和光强阶数避免不同配置混淆

    参数:
        cfg: SystemConfig 对象，或兼容的含有 image_size / n_amp_modes 的对象
        flag_noise: 覆盖 cfg.flag_noise
        flag_wf: 覆盖 cfg.flag_wf
    """
    fn = flag_noise if flag_noise is not None else cfg.flag_noise
    fw = flag_wf if flag_wf is not None else cfg.flag_wf
    if cfg.beam_size < cfg.image_size:
        suffix = f"_s{cfg.image_size}b{cfg.beam_size}_a{cfg.n_amp_modes}"
    else:
        suffix = f"_s{cfg.image_size}_a{cfg.n_amp_modes}"
    if fn:
        suffix += "_noise"
    if fw:
        suffix += "_wf"
    return suffix


def _noll_radial_order(j: int) -> int:
    """Noll 索引 j (1-based) 对应的径向阶数 n"""
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    return n


def get_zernike_decay_weights(n_modes: int, cfg) -> np.ndarray:
    """
    返回各阶泽尼克波前系数的衰减权重

    参数:
        n_modes: 模式数
        cfg: SystemConfig (使用 wf_decay_scheme / wf_decay_exponent)

    返回:
        (n_modes,) 权重数组，用于乘以随机系数

    scheme:
      "none"        → 全 1.0 (无衰减)
      "power_law"   → weight[j] = 1 / (radial_order(j+1) + 0.5)^exponent
      "kolmogorov"  → Noll 1976 大气湍流近似: weight ∝ 1/(n+1)^(5/3)
    """
    scheme = getattr(cfg, "wf_decay_scheme", "none")
    exponent = getattr(cfg, "wf_decay_exponent", 1.6)

    weights = np.ones(n_modes)
    if scheme == "none":
        return weights

    for j in range(n_modes):
        n = _noll_radial_order(j + 1)  # j is 0-based, convert to 1-based Noll
        if scheme == "power_law":
            weights[j] = 1.0 / ((n + 0.5) ** exponent)
        elif scheme == "kolmogorov":
            # Noll 1976: variance ∝ (n+1)^(-8/3) for Kolmogorov turbulence
            weights[j] = 1.0 / ((n + 1.0) ** (4.0 / 3.0))

    # 归一化到 weight[0] ≈ 1.0 (对 n≥1 的模式)
    if n_modes > 1 and weights[1] > 0:
        weights = weights / weights[1]

    return weights


def reconstruct_from_zernike(
    coeffs: np.ndarray,
    modes: np.ndarray,
    start_order: int = 1,
    end_order: int | None = None,
) -> np.ndarray:
    """
    从泽尼克系数向量化重构空间分布

    使用 np.tensordot 一次性完成所有模式的加权叠加，
    比逐个模式循环快 10-30 倍。

    参数:
        coeffs: 泽尼克系数向量，coeffs[k] 对应 Noll 第 k+1 阶
        modes: 泽尼克模式矩阵，形状 (H, W, N)，modes[:,:,k] 为 Noll 第 k+1 阶
        start_order: 起始 Noll 阶数 (1-based，含)，默认 1
        end_order: 结束 Noll 阶数 (1-based，含)，默认使用 coeffs 全部长度

    返回:
        (H, W) 的空间分布

    示例:
        reconstruct_from_zernike(coe, modes)          # 全部阶数
        reconstruct_from_zernike(coe, modes, 2, 15)   # 第2~15阶
    """
    if end_order is None:
        end_order = len(coeffs)
    start_idx = start_order - 1
    # tensordot: (n_modes,) · (H, W, n_modes) → (H, W)
    return np.tensordot(
        coeffs[start_idx:end_order],
        modes[:, :, start_idx:end_order],
        axes=([0], [2]),
    )


def create_embedded_mask(
    image_size: int = 256,
    beam_size: int = 240,
) -> np.ndarray:
    """
    创建嵌入在 image_size×image_size 图像中的圆形光束掩模（旧版兼容）

    当 beam_size == image_size 时等价于全图圆形光瞳。
    新代码请使用 generate_mask(cfg)。
    """
    offset = (image_size - beam_size) // 2
    mask = np.zeros((image_size, image_size))
    mask[offset : offset + beam_size, offset : offset + beam_size] = (
        Optics.std_beam(beam_size, 100, 100, 1e99)
    )
    return mask


# =========================================================================
# 配件生成函数 — 按配置自动生成 subcfg / modes / mask
# =========================================================================


def generate_mask(cfg) -> np.ndarray:
    """
    按配置生成圆形光瞳掩模

    beam_size < image_size 时光束嵌入靶面中心，使用 std_beam (匹配 MATLAB StdBeamFunc.m)。
    beam_size == image_size 时使用 pupil_circle 全图光瞳。

    参数:
        cfg: SystemConfig

    返回:
        (image_size, image_size) 的二值圆形光瞳
    """
    if cfg.beam_size < cfg.image_size:
        return create_embedded_mask(cfg.image_size, cfg.beam_size)
    return Optics.pupil_circle(cfg.image_size)


def generate_modes(cfg) -> np.ndarray:
    """
    按配置生成泽尼克模式矩阵

    模式在 beam_size×beam_size 网格上生成，嵌入到 image_size 靶面中心。
    modes[:,:,k] = Noll 索引 k+1 阶泽尼克多项式 (保持 MATLAB (H,W,N) 布局)。

    参数:
        cfg: SystemConfig (使用 image_size, beam_size, n_modes_total)

    返回:
        (image_size, image_size, n_modes_total) 的泽尼克模式数组
    """
    n_total = cfg.n_modes_total
    BS = cfg.beam_size
    IS = cfg.image_size
    modes_nhw = Optics.zernike_modes(n_total, BS)  # (n_total, BS, BS)
    if BS < IS:
        offset = cfg.beam_offset
        padded = np.zeros((n_total, IS, IS))
        padded[:, offset:offset + BS, offset:offset + BS] = modes_nhw
        return np.transpose(padded, (1, 2, 0))   # (IS, IS, n_total)
    return np.transpose(modes_nhw, (1, 2, 0))     # (IS, IS, n_total)


def generate_subcfg(cfg) -> np.ndarray:
    """
    按配置生成子孔径坐标矩阵

    使用 generate_mask 作为有效区域判定依据，自动筛选有效子孔径。

    参数:
        cfg: SystemConfig (使用 image_size, n_sub_dim, sub_ap_pixels, beam_size)

    返回:
        (2, n_valid_sub) 的坐标矩阵，第0行=y坐标，第1行=x坐标，
        坐标以图像中心为原点
    """
    near_field = generate_mask(cfg)
    sub_valid = create_sub_valid(cfg.n_sub_dim, near_field, cfg.sub_ap_pixels,
                                 ratio=cfg.sub_valid_ratio)
    subcfg = create_subcfg(sub_valid, cfg.image_size, cfg.sub_ap_pixels)
    return subcfg


def generate_simulation_data(
    cfg,
    modes: np.ndarray,
    subcfg: np.ndarray,
    optics: Optics,
    hs: HartmannSensor,
    n_samples: int,
    n_amp: int | None = None,
    enhanced: bool = False,
    seed: int | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    统一仿真数据生成函数

    随机生成光强泽尼克系数 → 重构近场振幅 → 叠加随机波前 →
    子孔径衍射传播 → 各子孔径总光强。

    参数:
        cfg: SystemConfig
        modes: 泽尼克模式矩阵 (H, W, n_modes_total)
        subcfg: 子孔径坐标 (2, n_sub)
        optics: Optics 光学系统
        hs: HartmannSensor (仅用于 _extract_sub_ap_field)
        n_samples: 样本数
        n_amp: 光强泽尼克阶数，None=使用 cfg.n_amp_modes
        enhanced: 是否启用增强模式 (超高斯包络+变方差+随机偏置)
        seed: 随机种子，None=使用 cfg.seed
        show_progress: 是否显示进度条

    返回:
        InputData: (n_samples, n_sub) 子孔径总光强
        OutputData: (n_samples, n_amp + 1) 光强泽尼克系数 + 偏移量
    """
    from tqdm import tqdm

    IS = cfg.image_size
    if n_amp is None:
        n_amp = cfg.n_amp_modes
    n_sub = subcfg.shape[1]
    rng = np.random.RandomState(seed if seed is not None else cfg.seed)

    InputData = np.zeros((n_samples, n_sub))
    OutputData = np.zeros((n_samples, n_amp + 1))

    # 预生成所有光强泽尼克系数 (向量化)
    if enhanced:
        sigmas = 0.5 + 1.5 * rng.random(n_samples)
        temp_coeffs = sigmas[:, np.newaxis] * rng.randn(n_samples, n_amp)
    else:
        temp_coeffs = rng.randn(n_samples, n_amp)
    OutputData[:, :n_amp] = temp_coeffs

    # 预生成所有波前系数
    wf_coeffs_all = None
    if cfg.flag_wf:
        wf_weights = get_zernike_decay_weights(cfg.n_wf_modes, cfg)
        wf_coeffs_all = cfg.wf_coeff_std * wf_weights * rng.randn(
            n_samples, cfg.n_wf_modes)
        wf_coeffs_all[:, :cfg.wf_skip_count] = 0.0

    iterator = range(n_samples)
    if show_progress:
        desc = "  生成增强仿真数据" if enhanced else "  生成仿真数据"
        iterator = tqdm(iterator, desc=desc, unit="样本")

    for i in iterator:
        coeffs_i = OutputData[i, :n_amp]

        if enhanced:
            Ampl = _generate_enhanced_amplitude(coeffs_i, modes, IS, rng)
            OutputData[i, n_amp] = 0.0
        else:
            Ampl = reconstruct_from_zernike(coeffs_i, modes)
            min_val = np.min(Ampl)
            OutputData[i, n_amp] = -min_val
            Ampl = Ampl - min_val

        # 波前
        wf = np.zeros((IS, IS))
        if cfg.flag_wf:
            wf = reconstruct_from_zernike(wf_coeffs_all[i], modes)

        InputField = Ampl * np.exp(-1j * wf)

        for i_sub in range(n_sub):
            sub_field = hs._extract_sub_ap_field(InputField, i_sub)
            result = optics.dl_propagate(sub_field)
            spot = np.abs(result) ** 2

            if cfg.flag_noise:
                spot = spot + 1.0 + cfg.noise_sigma * rng.randn(
                    cfg.sub_ap_pixels, cfg.sub_ap_pixels)

            InputData[i, i_sub] = np.sum(spot)

    return InputData, OutputData


def _generate_enhanced_amplitude(
    coeffs: np.ndarray, modes: np.ndarray, image_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """增强近场振幅: 超高斯包络 + 随机偏置 (方案2 内部函数)"""
    Ampl = reconstruct_from_zernike(coeffs, modes)

    Y, X = np.meshgrid(
        np.linspace(-1, 1, image_size),
        np.linspace(-1, 1, image_size),
    )
    r = np.hypot(X, Y)
    w = 0.6 + 0.4 * rng.random()
    n_sg = 2.0 + 6.0 * rng.random()
    envelope = np.exp(-((r / w) ** n_sg))
    envelope = np.where(r <= 1.0, envelope, 0.0)

    Ampl = Ampl * envelope
    Ampl = Ampl - np.min(Ampl)
    bias = 0.2 * rng.random() * np.max(Ampl)
    return Ampl + bias


def save_accessories(cfg, modes=None, subcfg=None) -> None:
    """
    将 modes 和 subcfg 缓存为 .mat 文件，加速后续加载

    文件名自动包含配置关键参数以避免混淆。

    参数:
        cfg: SystemConfig
        modes: 泽尼克模式矩阵，None=自动生成
        subcfg: 子孔径坐标矩阵，None=自动生成
    """
    os.makedirs(cfg.accessories_dir, exist_ok=True)

    if subcfg is None:
        subcfg = generate_subcfg(cfg)
    bs_tag = f"b{cfg.beam_size}_" if cfg.beam_size < cfg.image_size else ""
    subcfg_path = os.path.join(
        cfg.accessories_dir,
        f"Subcfg_{bs_tag}s{cfg.image_size}_d{cfg.n_sub_dim}_p{cfg.sub_ap_pixels}.mat",
    )
    save_mat(subcfg_path, Subcfg=subcfg)
    print(f"  子孔径配置已保存: {subcfg_path} ({subcfg.shape[1]} 个有效子孔径)")

    if modes is None:
        modes = generate_modes(cfg)
    modes_path = os.path.join(
        cfg.accessories_dir,
        f"modes_{bs_tag}s{cfg.image_size}_n{cfg.n_modes_total}.mat",
    )
    save_mat(modes_path, modes=modes)
    print(f"  泽尼克模式已保存: {modes_path} ({modes.shape})")


def load_accessories(cfg) -> tuple[np.ndarray, np.ndarray]:
    """
    加载或自动生成配件数据 (subcfg, modes)

    优先从 .mat 缓存加载，缓存不存在时自动生成并保存。

    参数:
        cfg: SystemConfig

    返回:
        (subcfg, modes): 子孔径坐标 (2, n_sub) 和泽尼克模式 (H, W, n_modes_total)
    """
    bs_tag = f"b{cfg.beam_size}_" if cfg.beam_size < cfg.image_size else ""
    subcfg_path = os.path.join(
        cfg.accessories_dir,
        f"Subcfg_{bs_tag}s{cfg.image_size}_d{cfg.n_sub_dim}_p{cfg.sub_ap_pixels}.mat",
    )
    modes_path = os.path.join(
        cfg.accessories_dir,
        f"modes_{bs_tag}s{cfg.image_size}_n{cfg.n_modes_total}.mat",
    )

    if os.path.exists(subcfg_path) and os.path.exists(modes_path):
        subcfg = load_mat(subcfg_path, "Subcfg")
        modes = load_mat(modes_path, "modes")
        print(f"  已加载配件: {subcfg.shape[1]} 子孔径, modes {modes.shape}")
    else:
        print("  配件缓存不存在，自动生成...")
        subcfg = generate_subcfg(cfg)
        modes = generate_modes(cfg)
        save_accessories(cfg, modes=modes, subcfg=subcfg)

    return subcfg, modes


# =========================================================================
# 实验图像处理函数
# =========================================================================


def preprocess_experimental_image(
    bmp_path: str,
    threshold: float = 10.0,
    crop_center_xy: tuple = (272, 251),
    crop_size: int = 400,
    flip_lr: bool = True,
) -> np.ndarray:
    """
    BMP → 灰度 → 减阈值去负值 → 裁剪 → 左右翻转

    crop_center_xy 为 MATLAB 1-based 坐标 (x_center, y_center):
    - x_center (272) → MATLAB rows → numpy rows (第0维)
    - y_center (251) → MATLAB cols → numpy cols (第1维)

    返回:
        (crop_size, crop_size) float64 数组
    """
    img = np.array(Image.open(bmp_path).convert("L"), dtype=np.float64)
    img = np.maximum(img - threshold, 0.0)

    x_center, y_center = crop_center_xy
    half = crop_size // 2
    # MATLAB: F(x_center-half+1:x_center+half, y_center-half+1:y_center+half)
    # MATLAB 1-based → Python 0-based: row_start = x_center - half
    row_start = x_center - half   # 272 - 200 = 72
    col_start = y_center - half   # 251 - 200 = 51
    img = img[row_start : row_start + crop_size,
              col_start : col_start + crop_size]

    if flip_lr:
        img = np.fliplr(img)

    return img


def extract_sub_spot_centroids(
    spot_image: np.ndarray,
    subcfg: np.ndarray,
    sub_ap_pixels: int,
    image_size: int,
) -> np.ndarray:
    """
    从哈特曼光斑图像直接提取各子孔径质心（不做衍射仿真）

    参数:
        spot_image: (image_size, image_size) 光斑图像
        subcfg: (2, n_sub) 子孔径坐标矩阵
        sub_ap_pixels: 子孔径边长 (像素)
        image_size: 图像尺寸 (像素)

    返回:
        (2, n_sub) 质心坐标，第0行=x，第1行=y
    """
    n_sub = subcfg.shape[1]
    centroids = np.zeros((2, n_sub))
    X_g, Y_g = np.meshgrid(np.arange(sub_ap_pixels), np.arange(sub_ap_pixels))

    for i in range(n_sub):
        y1_py = int(np.round(subcfg[0, i] + image_size / 2))
        x1_py = int(np.round(subcfg[1, i] + image_size / 2))
        y1_py = max(0, min(y1_py, image_size - sub_ap_pixels))
        x1_py = max(0, min(x1_py, image_size - sub_ap_pixels))
        sub_spot = spot_image[y1_py : y1_py + sub_ap_pixels, x1_py : x1_py + sub_ap_pixels]

        total = np.sum(sub_spot)
        if total > 0:
            centroids[0, i] = np.sum(sub_spot * X_g) / total  # cx
            centroids[1, i] = np.sum(sub_spot * Y_g) / total  # cy

    return centroids


def extract_sub_ap_total_intensity(
    spot_image: np.ndarray,
    subcfg: np.ndarray,
    sub_ap_pixels: int,
    image_size: int,
) -> np.ndarray:
    """
    从光斑图像提取各子孔径总光强值

    参数:
        spot_image: (image_size, image_size) 光斑图像
        subcfg: (2, n_sub) 子孔径坐标矩阵
        sub_ap_pixels: 子孔径边长 (像素)
        image_size: 图像尺寸 (像素)

    返回:
        (n_sub,) 各子孔径总光强
    """
    n_sub = subcfg.shape[1]
    intensity = np.zeros(n_sub)

    for i in range(n_sub):
        y1_py = int(np.round(subcfg[0, i] + image_size / 2))
        x1_py = int(np.round(subcfg[1, i] + image_size / 2))
        y1_py = max(0, min(y1_py, image_size - sub_ap_pixels))
        x1_py = max(0, min(x1_py, image_size - sub_ap_pixels))
        sub_spot = spot_image[y1_py : y1_py + sub_ap_pixels, x1_py : x1_py + sub_ap_pixels]
        intensity[i] = np.sum(sub_spot)

    return intensity


# =========================================================================
# 哈特曼阵列可视化
# =========================================================================


def plot_hartmann_grid(
    spot_image: np.ndarray,
    subcfg: np.ndarray,
    sub_ap_pixels: int,
    image_size: int,
    title: str = "",
    ax=None,
    cmap: str = "jet",
    rect_color: str = "r",
    rect_linewidth: float = 0.5,
    show_colorbar: bool = True,
    pupil_radius: float | None = None,
):
    """
    绘制哈特曼光斑图像，叠加有效子孔径网格 + 可选圆形光瞳边界

    参数:
        spot_image: (image_size, image_size) 光斑图像
        subcfg: (2, n_sub) 子孔径坐标矩阵
        sub_ap_pixels: 子孔径边长 (像素)
        image_size: 图像尺寸 (像素)
        title: 图表标题
        ax: 指定坐标轴，None=创建新图
        cmap: 颜色映射
        rect_color: 网格线颜色
        rect_linewidth: 网格线宽
        show_colorbar: 是否显示 colorbar
        pupil_radius: 光瞳半径 (像素)，None=不绘制

    返回:
        ax
    """
    from matplotlib.patches import Circle

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 7))

    im = ax.imshow(spot_image, cmap=cmap)
    n_sub = subcfg.shape[1]
    for i in range(n_sub):
        y1 = subcfg[0, i] + image_size / 2
        x1 = subcfg[1, i] + image_size / 2
        rect = plt.Rectangle(
            (x1, y1), sub_ap_pixels, sub_ap_pixels,
            fill=False, edgecolor=rect_color, linewidth=rect_linewidth,
        )
        ax.add_patch(rect)

    if pupil_radius is not None:
        center = image_size / 2
        circle = Circle(
            (center, center), pupil_radius,
            fill=False, edgecolor="white", linewidth=1.5, linestyle="--",
        )
        ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.set_title(title)
    if show_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8)

    return ax
