"""
评估指标模块

提供回归评估指标和波前复原质量评估函数。
评估指标兼容 (n_samples, n_outputs) 和 (n_outputs, n_samples) 两种数据布局。
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def _ensure_samples_first(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """若数据为 (n_outputs, n_samples) 布局，转置为 (n_samples, n_outputs)"""
    if y_true.ndim == 2 and y_true.shape[0] < y_true.shape[1]:
        y_true = y_true.T
    if y_pred.ndim == 2 and y_pred.shape[0] < y_pred.shape[1]:
        y_pred = y_pred.T
    return y_true, y_pred


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    均方误差 (Mean Squared Error)

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        MSE 值
    """
    y_true, y_pred = _ensure_samples_first(y_true, y_pred)
    return float(mean_squared_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    均方根误差 (Root Mean Squared Error)

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        RMSE 值
    """
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    决定系数 (R² Score)

    参数:
        y_true: 真实值
        y_pred: 预测值

    返回:
        R² 值
    """
    y_true, y_pred = _ensure_samples_first(y_true, y_pred)
    return float(r2_score(y_true, y_pred))


def compute_rms_wavefront(
    wf_pred: np.ndarray,
    wf_true: np.ndarray,
    wavelength: float,
    mask: np.ndarray | None = None,
) -> float:
    """
    计算波前残差 RMS，单位 λ (waves)

    参数:
        wf_pred: 复原波前 (2D, 单位: 弧度)
        wf_true: 真实波前 (2D, 单位: 弧度)
        wavelength: 波长 (mm)
        mask: 可选掩模，仅计算掩模内有效区域

    返回:
        RMS 波前残差 (λ)
    """
    residual = wf_pred - wf_true
    if mask is not None:
        residual = residual[mask != 0]
    return float(np.std(residual) / (2 * np.pi) * wavelength * 1e3)


def compute_pv_wavefront(
    wf_pred: np.ndarray,
    wf_true: np.ndarray,
    wavelength: float,
    mask: np.ndarray | None = None,
) -> float:
    """
    计算波前残差 PV (峰谷值)，单位 λ (waves)

    参数:
        wf_pred: 复原波前 (2D, 单位: 弧度)
        wf_true: 真实波前 (2D, 单位: 弧度)
        wavelength: 波长 (mm)
        mask: 可选掩模，仅计算掩模内有效区域

    返回:
        PV 波前残差 (λ)
    """
    residual = wf_pred - wf_true
    if mask is not None:
        residual = residual[mask != 0]
    return float((np.max(residual) - np.min(residual)) / (2 * np.pi) * wavelength * 1e3)
