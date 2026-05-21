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

from .optics import Optics
from .hartmann import create_sub_valid, create_subcfg


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
