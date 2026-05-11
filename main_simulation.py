"""
完整仿真主脚本: 哈特曼波前传感 + ELM近场复原 + 波前复原对比

实现完整的自适应光学仿真流程:
1. 加载数据和 ELM 模型
2. 哈特曼传感器标定 (平面波入射)
3. 构建斜率响应矩阵 Z2S (逐阶泽尼克模式波前)
4. 测试: 随机波前 + 真实光强 → 测量斜率
5. 传统方法: 斜率 → 波前复原
6. 本方法: ELM预测光强 → 光强差分解耦 → 差分斜率 → 波前复原
7. 比较两种方法的 RMS/PV 残差

MATLAB 对应: Main_A_ELM_simulation.m
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optics import Optics
from src.elm import ELM
from src.hartmann import HartmannSensor
from src.utils import (
    load_mat,
    set_seed,
    normalize_data,
    apply_normalize,
    reverse_normalize,
    split_data,
    circ_mask,
    configure_chinese_font,
)


# ===========================================================================
# 参数配置
# ===========================================================================
@dataclass
class Config:
    """完整仿真参数"""

    # ---- 光学系统参数 ----
    wavelength: float = 1.064e-3
    focal_length: float = 12.0              # 微透镜焦距 (mm) - 与 MATLAB 一致
    pixel_pitch: float = 14e-3
    sub_ap_pixels: int = 20
    image_size: int = 256

    # ---- 数据参数 ----
    n_zernike_amp: int = 25                 # 光强分布泽尼克阶数 (AN)
    flag_noise: bool = True
    flag_wf: bool = True

    # ---- 波前参数 ----
    n_wf_modes: int = 15                    # 波前泽尼克阶数 (WN)
    wf_coeff_std: float = 0.2

    # ---- ELM 参数 ----
    activation: str = "softplus"
    n_hidden: int | None = None             # None=自动搜索

    # ---- 噪声 ----
    noise_sigma: float = 0.5

    # ---- 测试样本索引 ----
    test_index: int = 21

    # ---- 随机种子 ----
    seed: int = 1

    # ---- 路径 ----
    accessories_dir: str = "accessories"
    data_dir: str = "data"
    result_dir: str = "result"


def _get_data_name(cfg: Config) -> str:
    if not cfg.flag_noise and not cfg.flag_wf:
        return f"_{cfg.n_zernike_amp}"
    elif cfg.flag_noise and not cfg.flag_wf:
        return f"_{cfg.n_zernike_amp}_noise"
    elif cfg.flag_noise and cfg.flag_wf:
        return f"_{cfg.n_zernike_amp}_noise_wf"
    else:
        return f"_{cfg.n_zernike_amp}_wf"


def main():
    cfg = Config()
    set_seed(cfg.seed)
    configure_chinese_font()
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ---- 1. 加载数据 ----
    print("加载数据...")
    name = _get_data_name(cfg)
    subcfg = load_mat(os.path.join(cfg.accessories_dir, "Subcfg.mat"), "Subcfg")
    modes = load_mat(os.path.join(cfg.accessories_dir, "modes250.mat"), "modes")
    InputData = load_mat(os.path.join(cfg.data_dir, f"InputData{name}.mat"))
    OutputData = load_mat(os.path.join(cfg.data_dir, f"OutputData{name}.mat"))

    n_sub = subcfg.shape[1]
    nZer = OutputData.shape[0] - 1
    nZerRecon = nZer
    print(f"  子孔径数: {n_sub}, 泽尼克阶数: {nZer}")

    # ---- 2. ELM 训练 ----
    print("训练 ELM 模型...")
    zer_indices = list(range(nZerRecon)) + [OutputData.shape[0] - 1]
    X = InputData
    y = OutputData[zer_indices, :]

    X_train, X_test, y_train, y_test = split_data(X, y, test_ratio=0.1, shuffle=False)
    X_norm, y_norm, scaler_X, scaler_y = normalize_data(X_train, y_train)
    X_test_norm = apply_normalize(X_test, scaler_X)

    # 搜索最优隐藏层神经元
    if cfg.n_hidden is None:
        hidden_list = list(range(100, 1550, 50))
        mse_list = np.zeros(len(hidden_list))
        for idx, n_hid in enumerate(
            tqdm(hidden_list, desc="  搜索最优隐藏层神经元数", unit="个")
        ):
            elm_tmp = ELM(n_hidden=n_hid, activation=cfg.activation)
            elm_tmp.fit(X_norm, y_norm)
            pred_norm = elm_tmp.predict(X_test_norm)
            pred_tmp = reverse_normalize(pred_norm, scaler_y)
            mse_list[idx] = np.mean([
                np.mean((pred_tmp[:, k] - y_test[:, k]) ** 2)
                for k in range(y_test.shape[1])
            ])
        best_hidden = hidden_list[np.argmin(mse_list)]
        print(f"  最优神经元数: {best_hidden}, MSE={np.min(mse_list):.4e}")
    else:
        best_hidden = cfg.n_hidden

    # 训练最终模型
    elm = ELM(n_hidden=best_hidden, activation=cfg.activation)
    elm.fit(X_norm, y_norm)
    pred_norm = elm.predict(X_test_norm)
    T_sim = reverse_normalize(pred_norm, scaler_y)

    # ---- 3. 初始化光学系统和哈特曼传感器 ----
    optics = Optics(
        wavelength=cfg.wavelength,
        pixel_pitch=cfg.pixel_pitch,
        focal_length=cfg.focal_length,
        n_pixels=cfg.image_size,
        sub_ap_pixels=cfg.sub_ap_pixels,
    )
    hs = HartmannSensor(subcfg, optics)

    # 生成标准圆域掩模
    mask = np.zeros((cfg.image_size, cfg.image_size))
    temp = Optics.std_beam(240, 100, 100, 1e99)
    mask[8:248, 8:248] = temp

    # ---- 4. 哈特曼传感器标定 (平面波) ----
    print("哈特曼传感器标定 (平面波)...")
    hs.calibrate(mask, noise_sigma=cfg.noise_sigma)
    print("  标定完成")

    # ---- 5. 构建斜率响应矩阵 Z2S ----
    print(f"构建斜率响应矩阵 (前{cfg.n_wf_modes}阶泽尼克)...")
    Z2S = hs.build_response_matrix(modes, mask, cfg.n_wf_modes, noise_sigma=cfg.noise_sigma)
    Recon = np.linalg.pinv(Z2S)  # 波前复原矩阵 (2S, N)
    print(f"  Z2S 形状: {Z2S.shape}")

    # ---- 6. 测试: 随机波前 → 对比两种复原方法 ----
    idx = min(cfg.test_index, y_test.shape[1] - 1)
    print(f"\n测试样本 #{idx}:")

    # 6.1 生成随机波前 (不含前2阶 piston/tilt)
    np.random.seed(21)
    coe = cfg.wf_coeff_std * np.random.randn(cfg.n_wf_modes)
    wf = np.zeros((cfg.image_size, cfg.image_size))
    tempW = np.zeros(240)
    for i in range(2, cfg.n_wf_modes):  # 从第3阶开始 (跳过piston和tilt)
        tempW = tempW + coe[i] * modes[:, :, i]
    wf[8:248, 8:248] = tempW

    # 6.2 重构真实近场振幅 A0
    np.random.seed(2)
    A0_mat = np.zeros(240)
    for i in range(nZer):
        A0_mat = A0_mat + y_test[i, idx] * modes[:, :, i]
    A0_mat = A0_mat + y_test[-1, idx]  # 最小值偏移

    # 6.3 传统方法: 真实光强 + 波前 → 测量斜率 → 复原
    IntensityMat = mask.copy()
    IntensityMat[8:248, 8:248] = A0_mat
    Input_trad = IntensityMat * np.exp(-1j * wf)

    slopes_int, _ = hs.measure_slopes(Input_trad, noise_sigma=cfg.noise_sigma)
    recoe = hs.reconstruct(slopes_int, Recon)  # 传统方法复原系数

    # 6.4 本方法: ELM 预测光强 → 测斜率 → 差分
    A1_mat = np.zeros(240)
    for i in range(nZer):
        if i < nZerRecon:
            A1_mat = A1_mat + T_sim[i, idx] * modes[:, :, i]
    A1_mat = A1_mat + T_sim[-1, idx]

    # ELM预测光强单独入射
    IntensityMat_R = mask.copy()
    IntensityMat_R[8:248, 8:248] = A1_mat
    Input_R = IntensityMat_R  # 纯振幅，无波前

    slopes_r, _ = hs.measure_slopes(Input_R, noise_sigma=cfg.noise_sigma)
    # 差分斜率: 含波前斜率 - 纯光强斜率 = 纯波前斜率
    recoeR = hs.reconstruct(slopes_int - slopes_r, Recon)

    # ---- 7. 结果对比 ----
    # 泽尼克系数对比 (从第3阶开始)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    orders = np.arange(2, cfg.n_wf_modes)
    ax.plot(orders, coe[2:], "ko-", linewidth=1.5, label="真实值")
    ax.plot(orders, recoe[2:], "r.-", linewidth=1.5, label="传统方法")
    ax.plot(orders, recoeR[2:], "b.-", linewidth=1.5, label="本方法")
    ax.set_xlabel("泽尼克阶数", fontsize=13)
    ax.set_ylabel("泽尼克系数", fontsize=13)
    ax.set_title("泽尼克系数对比 (三种方法)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "coefficients_comparison.png"), dpi=150)
    plt.show()

    # 波前残差计算 RMS 和 PV
    residual_trad = np.std(np.abs(recoe[2:] - coe[2:]))
    residual_proposed = np.std(np.abs(recoeR[2:] - coe[2:]))
    print(f"  传统方法 RMS 残差: {residual_trad:.4f}")
    print(f"  本方法 RMS 残差:   {residual_proposed:.4f}")

    # 重构波前分布图
    wf_0 = np.zeros(240)  # 真实波前
    wf_1 = np.zeros(240)  # 传统方法复原
    wf_r = np.zeros(240)  # 本方法复原
    for i in range(2, cfg.n_wf_modes):
        wf_0 = wf_0 + coe[i] * modes[:, :, i]
        wf_1 = wf_1 + recoe[i] * modes[:, :, i]
        wf_r = wf_r + recoeR[i] * modes[:, :, i]

    mask_240 = circ_mask(240)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    vlim = 2.0
    im0 = axes[0].imshow(wf_0 * mask_240, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[0].set_title("入射波前 (真实)", fontsize=13)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(wf_1 * mask_240, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[1].set_title("传统方法复原波前", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(wf_r * mask_240, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[2].set_title("本方法复原波前", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle("波前复原对比", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "wavefront_comparison.png"), dpi=150)
    plt.show()

    # 残差图
    diff_trad = (wf_1 - wf_0) * mask_240
    diff_prop = (wf_r - wf_0) * mask_240
    rms_trad = np.std(diff_trad[diff_trad != 0]) / (2 * np.pi)
    pv_trad = (np.max(diff_trad) - np.min(diff_trad)) / (2 * np.pi)
    rms_prop = np.std(diff_prop[diff_prop != 0]) / (2 * np.pi)
    pv_prop = (np.max(diff_prop) - np.min(diff_prop)) / (2 * np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im3 = axes[0].imshow(diff_trad, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[0].set_title(
        f"传统方法残差\nRMS ≈ {rms_trad:.3f}λ, PV ≈ {pv_trad:.3f}λ", fontsize=13
    )
    axes[0].axis("off")
    plt.colorbar(im3, ax=axes[0], shrink=0.8)

    im4 = axes[1].imshow(diff_prop, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[1].set_title(
        f"本方法残差\nRMS ≈ {rms_prop:.3f}λ, PV ≈ {pv_prop:.3f}λ", fontsize=13
    )
    axes[1].axis("off")
    plt.colorbar(im4, ax=axes[1], shrink=0.8)

    fig.suptitle("波前复原残差对比", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "wavefront_residual.png"), dpi=150)
    plt.show()

    # 近场振幅二维对比
    NF0 = A0_mat * mask_240
    NF1 = A1_mat * mask_240
    cMax = max(np.max(NF0), np.max(NF1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    im5 = axes[0].imshow(NF0, cmap="jet", vmin=0, vmax=cMax)
    axes[0].set_title("原始近场分布", fontsize=13)
    axes[0].axis("off")
    plt.colorbar(im5, ax=axes[0], shrink=0.8)

    im6 = axes[1].imshow(NF1, cmap="jet", vmin=0, vmax=cMax)
    axes[1].set_title("ELM 复原近场分布", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im6, ax=axes[1], shrink=0.8)

    im7 = axes[2].imshow(NF1 - NF0, cmap="jet", vmin=-cMax, vmax=cMax)
    axes[2].set_title("残差", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im7, ax=axes[2], shrink=0.8)

    fig.suptitle(f"近场分布对比 ({cfg.n_zernike_amp}阶泽尼克)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "nearfield_comparison.png"), dpi=150)
    plt.show()

    print(f"\n仿真完成! 结果图片已保存至 '{cfg.result_dir}/'")


if __name__ == "__main__":
    main()
