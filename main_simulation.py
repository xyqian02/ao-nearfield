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
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import get_config
from src.optics import Optics
from src.elm import ELM
from src.hartmann import HartmannSensor
from src.evaluation import compute_mse, compute_rmse, compute_r2, compute_rms_wavefront, compute_pv_wavefront
from src.utils import (
    load_mat,
    set_seed,
    configure_chinese_font,
    get_data_filename,
    reconstruct_from_zernike,
    load_accessories,
    generate_mask,
)


def main():
    cfg = get_config()
    set_seed(cfg.seed)
    configure_chinese_font()
    os.makedirs(cfg.result_dir, exist_ok=True)
    IS = cfg.image_size

    # ---- 1. 加载数据 ----
    print("加载数据...")
    name = get_data_filename(cfg)
    subcfg, modes = load_accessories(cfg)
    InputData = load_mat(os.path.join(cfg.data_dir, f"InputData{name}.mat"))
    OutputData = load_mat(os.path.join(cfg.data_dir, f"OutputData{name}.mat"))

    n_sub = subcfg.shape[1]
    n_amp = OutputData.shape[1] - 1
    print(f"  子孔径数: {n_sub}, 光强泽尼克阶数: {n_amp}")
    print(f"  图像尺寸: {IS}×{IS}")

    # ---- 2. ELM 训练 ----
    print("训练 ELM 模型...")
    X = InputData
    y = OutputData

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=cfg.test_ratio, shuffle=False
    )
    val_size = cfg.val_ratio / (1 - cfg.test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, shuffle=False
    )

    scaler_X = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(y_train)
    X_train_norm = scaler_X.transform(X_train)
    y_train_norm = scaler_y.transform(y_train)
    X_val_norm = scaler_X.transform(X_val)

    # 搜索最优隐藏层神经元
    if cfg.n_hidden is None:
        hidden_list = list(range(100, 1550, 50))
        mse_list = np.zeros(len(hidden_list))
        for idx, n_hid in enumerate(
            tqdm(hidden_list, desc="  搜索最优隐藏层神经元数", unit="个")
        ):
            elm_tmp = ELM(n_hidden=n_hid, activation=cfg.activation, random_state=cfg.seed)
            elm_tmp.fit(X_train_norm, y_train_norm)
            y_val_pred_norm = elm_tmp.predict(X_val_norm)
            y_val_pred = scaler_y.inverse_transform(y_val_pred_norm)
            mse_list[idx] = compute_mse(y_val, y_val_pred)
        best_hidden = hidden_list[np.argmin(mse_list)]
        print(f"  最优神经元数: {best_hidden}, 验证集 MSE={np.min(mse_list):.4e}")
    else:
        best_hidden = cfg.n_hidden

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.vstack([y_train, y_val])
    X_train_full_norm = scaler_X.transform(X_train_full)
    y_train_full_norm = scaler_y.transform(y_train_full)

    X_test_norm = scaler_X.transform(X_test)

    elm = ELM(n_hidden=best_hidden, activation=cfg.activation, random_state=cfg.seed)
    elm.fit(X_train_full_norm, y_train_full_norm)
    T_sim_norm = elm.predict(X_test_norm)
    T_sim = scaler_y.inverse_transform(T_sim_norm)

    test_mse = compute_mse(y_test, T_sim)
    test_rmse = compute_rmse(y_test, T_sim)
    test_r2 = compute_r2(y_test, T_sim)
    print(f"  测试集 MSE = {test_mse:.4e}, RMSE = {test_rmse:.4e}, R² = {test_r2:.4f}")

    # ---- 3. 初始化光学系统和哈特曼传感器 ----
    optics = Optics(
        wavelength=cfg.wavelength,
        pixel_pitch=cfg.pixel_pitch,
        focal_length=cfg.focal_length,
        n_pixels=IS,
        sub_ap_pixels=cfg.sub_ap_pixels,
    )
    hs = HartmannSensor(subcfg, optics)
    mask = generate_mask(cfg)

    # ---- 4. 哈特曼传感器标定 (平面波) ----
    print("哈特曼传感器标定 (平面波)...")
    hs.calibrate(mask, noise_sigma=cfg.noise_sigma)
    print("  标定完成")

    # ---- 5. 构建斜率响应矩阵 Z2S ----
    n_wf = cfg.n_wf_modes
    print(f"构建斜率响应矩阵 ({n_wf}阶泽尼克)...")
    Z2S = hs.build_response_matrix(modes, mask, n_wf, noise_sigma=cfg.noise_sigma)
    Recon = np.linalg.pinv(Z2S)
    print(f"  Z2S 形状: {Z2S.shape}")

    # ---- 6. 测试: 随机波前 → 对比两种复原方法 ----
    idx = min(cfg.batch_test_index, y_test.shape[0] - 1)
    print(f"\n测试样本 #{idx}:")

    wf_skip = cfg.wf_skip_count

    # 6.1 生成随机波前
    np.random.seed(21)
    coe = cfg.wf_coeff_std * np.random.randn(n_wf)
    # 将跳过的低阶模式系数置零
    coe[:wf_skip] = 0.0
    wf = reconstruct_from_zernike(coe, modes)

    # 6.2 重构真实近场振幅 A0
    np.random.seed(2)
    A0 = reconstruct_from_zernike(y_test[idx, :n_amp], modes) + y_test[idx, -1]

    # 6.3 传统方法: 真实光强 + 波前 → 测量斜率 → 复原
    Input_trad = A0 * np.exp(-1j * wf)
    slopes_int, _ = hs.measure_slopes(Input_trad, noise_sigma=cfg.noise_sigma)
    recoe = hs.reconstruct(slopes_int, Recon)
    recoe[:wf_skip] = 0.0

    # 6.4 本方法: ELM 预测光强 → 测斜率 → 差分
    A1 = reconstruct_from_zernike(T_sim[idx, :n_amp], modes) + T_sim[idx, -1]

    Input_R = A1  # 纯振幅，无波前
    slopes_r, _ = hs.measure_slopes(Input_R, noise_sigma=cfg.noise_sigma)
    recoeR = hs.reconstruct(slopes_int - slopes_r, Recon)
    recoeR[:wf_skip] = 0.0

    # ---- 7. 结果对比 ----
    # 泽尼克系数对比
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    orders = np.arange(wf_skip, n_wf)
    ax.plot(orders, coe[wf_skip:], "ko-", linewidth=1.5, label="真实值")
    ax.plot(orders, recoe[wf_skip:], "r.-", linewidth=1.5, label="传统方法")
    ax.plot(orders, recoeR[wf_skip:], "b.-", linewidth=1.5, label="本方法")
    ax.set_xlabel("泽尼克阶数", fontsize=13)
    ax.set_ylabel("泽尼克系数", fontsize=13)
    ax.set_title("波前泽尼克系数对比 (三种方法)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "coefficients_comparison.png"), dpi=150)
    plt.show()

    # 重构波前分布图
    wf_0 = reconstruct_from_zernike(coe, modes)
    wf_1 = reconstruct_from_zernike(recoe, modes)
    wf_r = reconstruct_from_zernike(recoeR, modes)

    wf_0_m = wf_0 * mask
    wf_1_m = wf_1 * mask
    wf_r_m = wf_r * mask

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    vlim = 2.0
    im0 = axes[0].imshow(wf_0_m, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[0].set_title("入射波前 (真实)", fontsize=13)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(wf_1_m, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[1].set_title("传统方法复原波前", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(wf_r_m, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[2].set_title("本方法复原波前", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle("波前复原对比", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "wavefront_comparison.png"), dpi=150)
    plt.show()

    # 残差
    rms_trad = compute_rms_wavefront(wf_1, wf_0, mask)
    pv_trad = compute_pv_wavefront(wf_1, wf_0, mask)
    rms_prop = compute_rms_wavefront(wf_r, wf_0, mask)
    pv_prop = compute_pv_wavefront(wf_r, wf_0, mask)

    print(f"  传统方法: RMS ≈ {rms_trad:.4f}λ, PV ≈ {pv_trad:.4f}λ")
    print(f"  本方法:   RMS ≈ {rms_prop:.4f}λ, PV ≈ {pv_prop:.4f}λ")

    diff_trad = (wf_1 - wf_0) * mask
    diff_prop = (wf_r - wf_0) * mask

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
    NF0 = A0 * mask
    NF1 = A1 * mask
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

    fig.suptitle(f"近场分布对比 ({n_amp}阶泽尼克)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "nearfield_comparison.png"), dpi=150)
    plt.show()

    print(f"\n仿真完成! 结果图片已保存至 '{cfg.result_dir}/'")


if __name__ == "__main__":
    main()
