"""
工具函数模块

提供数据IO（.mat文件读写）、随机种子设置、中文字体配置、
泽尼克重构等基础功能。
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


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
    n_zernike: int,
    flag_noise: bool = True,
    flag_wf: bool = True,
) -> str:
    """
    根据泽尼克阶数和噪声/波前标记生成数据文件名后缀

    此函数替代各主脚本中重复定义的 _get_data_name 函数。
    """
    suffix = f"_{n_zernike}"
    if flag_noise:
        suffix += "_noise"
    if flag_wf:
        suffix += "_wf"
    return suffix


def reconstruct_from_zernike(
    coeffs: np.ndarray,
    modes: np.ndarray,
    n_modes: int | None = None,
    start_mode: int = 0,
) -> np.ndarray:
    """
    从泽尼克系数向量化重构空间分布（单行替代整个 for 循环）

    使用 np.tensordot 一次性完成所有模式的加权叠加，
    比逐个模式循环快 10-30 倍。

    参数:
        coeffs: 泽尼克系数向量，形状 (n_modes,) 或更长
        modes: 泽尼克模式矩阵，形状 (H, W, N)
        n_modes: 使用的阶数，默认使用 coeffs 长度
        start_mode: 起始阶数 (0-based)，默认 0

    返回:
        (H, W) 的空间分布
    """
    if n_modes is None:
        n_modes = len(coeffs)
    # tensordot: (n_modes,) · (H, W, n_modes) → (H, W)
    return np.tensordot(
        coeffs[start_mode : start_mode + n_modes],
        modes[:, :, start_mode : start_mode + n_modes],
        axes=([0], [2]),
    )


def create_embedded_mask(
    image_size: int = 256,
    beam_size: int = 240,
) -> np.ndarray:
    """
    创建嵌入在 image_size×image_size 图像中的圆形光束掩模

    光束区域 (beam_size×beam_size) 居中嵌入，外部为 0。

    参数:
        image_size: 全图尺寸
        beam_size: 光束区域尺寸

    返回:
        (image_size, image_size) 的二值掩模
    """
    from .optics import Optics

    offset = (image_size - beam_size) // 2
    mask = np.zeros((image_size, image_size))
    mask[offset : offset + beam_size, offset : offset + beam_size] = (
        Optics.std_beam(beam_size, 100, 100, 1e99)
    )
    return mask
