"""
工具函数模块

提供数据IO（.mat文件读写）、评估指标、随机种子设置、中文字体配置等基础功能。
使用 sklearn 的 MinMaxScaler 替代 MATLAB 的 mapminmax，
使用 sklearn.metrics.mean_squared_error 替代 MATLAB 的 mse。
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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


def normalize_data(
    X: np.ndarray, y: np.ndarray, feature_range: tuple = (-1, 1)
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    对输入和输出数据进行归一化处理

    参数:
        X: 输入数据，形状 (n_features, n_samples)
        y: 输出数据，形状 (n_outputs, n_samples)
        feature_range: 归一化范围，默认 (-1, 1)

    返回:
        X_norm: 归一化后的输入
        y_norm: 归一化后的输出
        scaler_X: 输入归一化器（用于后续 apply）
        scaler_y: 输出归一化器（用于后续 reverse）
    """
    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)

    # MinMaxScaler 期望 (n_samples, n_features)，这里数据按 (n_features, n_samples) 组织
    # 需要转置适配，再转置回来
    X_norm = scaler_X.fit_transform(X.T).T
    y_norm = scaler_y.fit_transform(y.T).T

    return X_norm, y_norm, scaler_X, scaler_y


def apply_normalize(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """对数据应用已有的归一化器"""
    return scaler.transform(X.T).T


def reverse_normalize(y_norm: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """将归一化后的数据逆变换回原始范围"""
    return scaler.inverse_transform(y_norm.T).T


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算多输出样本集的平均 MSE

    参数:
        y_true: 真实值，形状 (n_outputs, n_samples)
        y_pred: 预测值，形状 (n_outputs, n_samples)

    返回:
        所有样本的平均 MSE
    """
    n_samples = y_true.shape[1]
    errors = np.zeros(n_samples)
    for i in range(n_samples):
        errors[i] = mean_squared_error(y_true[:, i], y_pred[:, i])
    return float(np.mean(errors))


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.1,
    shuffle: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    划分训练集和测试集

    参数:
        X: 输入数据 (n_features, n_samples)
        y: 输出数据 (n_outputs, n_samples)
        test_ratio: 测试集比例
        shuffle: 是否打乱数据
        seed: 随机种子

    返回:
        X_train, X_test, y_train, y_test
    """
    n_samples = X.shape[1]
    n_test = int(np.round(test_ratio * n_samples))

    if shuffle:
        # sklearn 的 train_test_split 在 feature 维度操作
        X_train, X_test, y_train, y_test = train_test_split(
            X.T, y.T, test_size=test_ratio, random_state=seed
        )
        return X_train.T, X_test.T, y_train.T, y_test.T
    else:
        # 不 shuffle，直接按顺序划分（与 MATLAB 原代码一致）
        X_train = X[:, : n_samples - n_test]
        X_test = X[:, n_samples - n_test :]
        y_train = y[:, : n_samples - n_test]
        y_test = y[:, n_samples - n_test :]
        return X_train, X_test, y_train, y_test


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
