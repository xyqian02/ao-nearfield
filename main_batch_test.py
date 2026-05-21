"""
批量波前复原测试主脚本

支持三种测试模式，比较传统哈特曼波前复原方法与
本方法(ELM近场复原 + 差分斜率)的波前残差 RMS。

运行方式:
    python main_batch_test.py rc   # 随机组合测试
    python main_batch_test.py fi   # 固定光强测试
    python main_batch_test.py fw   # 固定波前测试

MATLAB 对应: Main_more.m
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import get_config, SystemConfig
from src.optics import Optics
from src.elm import ELM
from src.hartmann import HartmannSensor
from src.evaluation import compute_rms_wavefront, compute_mse
from src.utils import (
    load_mat,
    set_seed,
    configure_chinese_font,
    get_data_filename,
    reconstruct_from_zernike,
    load_accessories,
    generate_mask,
    get_zernike_decay_weights,
)


# ===========================================================================
# 模式元数据
# ===========================================================================
MODE_META = {
    "rc": {"name": "随机组合", "label": "rc", "desc": "随机波前 + 数据集光强，每组均不同"},
    "fi": {"name": "固定光强", "label": "fi", "desc": "同一光强分布，叠加随机波前"},
    "fw": {"name": "固定波前", "label": "fw", "desc": "同一波前分布，叠加数据集中的不同光强"},
}


# ===========================================================================
# 系统初始化
# ===========================================================================
def setup_system(cfg: SystemConfig) -> dict:
    """初始化仿真系统：加载数据、训练 ELM、标定哈特曼传感器"""
    set_seed(cfg.seed)
    configure_chinese_font()
    os.makedirs(cfg.result_dir, exist_ok=True)
    IS = cfg.image_size

    print("=" * 50)
    print(" 加载数据与初始化")
    print("=" * 50)
    name = get_data_filename(cfg)
    subcfg, modes = load_accessories(cfg)
    InputData = load_mat(os.path.join(cfg.data_dir, f"InputData{name}.mat"))
    OutputData = load_mat(os.path.join(cfg.data_dir, f"OutputData{name}.mat"))

    n_sub = subcfg.shape[1]
    n_amp = OutputData.shape[1] - 1
    print(f"  子孔径数: {n_sub}, 光强阶数: {n_amp}, 图像: {IS}x{IS}")

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

    if cfg.n_hidden is None:
        hidden_list = list(
            range(cfg.hidden_range[0], cfg.hidden_range[1] + 1, cfg.hidden_range[2])
        )
        mse_list = np.zeros(len(hidden_list))
        for idx, n_hid in enumerate(
            tqdm(hidden_list, desc="  搜索最优神经元数", unit="个")
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

    print(f"  训练 ELM (N={best_hidden}, activation={cfg.activation})...")
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.vstack([y_train, y_val])
    X_train_full_norm = scaler_X.transform(X_train_full)
    y_train_full_norm = scaler_y.transform(y_train_full)
    X_test_norm = scaler_X.transform(X_test)

    elm = ELM(n_hidden=best_hidden, activation=cfg.activation, random_state=cfg.seed)
    elm.fit(X_train_full_norm, y_train_full_norm)
    T_sim_norm = elm.predict(X_test_norm)
    T_sim = scaler_y.inverse_transform(T_sim_norm)

    optics = Optics(
        wavelength=cfg.wavelength,
        pixel_pitch=cfg.pixel_pitch,
        focal_length=cfg.focal_length,
        n_pixels=IS,
        sub_ap_pixels=cfg.sub_ap_pixels,
    )
    hs = HartmannSensor(subcfg, optics)
    mask = generate_mask(cfg)

    print("  哈特曼传感器标定...")
    hs.calibrate(mask, noise_sigma=cfg.noise_sigma)

    n_wf = cfg.n_wf_modes
    print(f"  构建斜率响应矩阵 ({n_wf}阶)...")
    Z2S = hs.build_response_matrix(modes, mask, n_wf, noise_sigma=cfg.noise_sigma)
    Recon = np.linalg.pinv(Z2S)

    n_test = X_test.shape[0]
    n_batch = min(cfg.n_batch_samples or n_test, n_test)

    return {
        "subcfg": subcfg, "modes": modes, "n_sub": n_sub,
        "n_amp": n_amp, "y_test": y_test, "T_sim": T_sim,
        "optics": optics, "hs": hs, "mask": mask, "Recon": Recon,
        "n_batch": n_batch,
    }


# ===========================================================================
# 单次测试迭代
# ===========================================================================
def run_single_test(
    hs, mask, Recon, cfg, A0, A1, coe_wf, modes,
) -> tuple[float, float]:
    """单次波前复原测试，返回 (rms_trad, rms_prop)"""
    n_wf = cfg.n_wf_modes
    wf_skip = cfg.wf_skip_count

    coe_wf = coe_wf.copy()
    coe_wf[:wf_skip] = 0.0
    wf_full = reconstruct_from_zernike(coe_wf, modes)

    Input_trad = A0 * np.exp(-1j * wf_full)
    slopes_int, _ = hs.measure_slopes(Input_trad, noise_sigma=cfg.noise_sigma)
    recoe = hs.reconstruct(slopes_int, Recon)
    recoe[:wf_skip] = 0.0

    Input_R = A1
    slopes_r, _ = hs.measure_slopes(Input_R, noise_sigma=cfg.noise_sigma)
    recoeR = hs.reconstruct(slopes_int - slopes_r, Recon)
    recoeR[:wf_skip] = 0.0

    wf_0 = reconstruct_from_zernike(coe_wf, modes)
    wf_1 = reconstruct_from_zernike(recoe, modes)
    wf_r = reconstruct_from_zernike(recoeR, modes)

    rms_trad = compute_rms_wavefront(wf_1, wf_0, mask)
    rms_prop = compute_rms_wavefront(wf_r, wf_0, mask)

    return rms_trad, rms_prop


# ===========================================================================
# 测试模式
# ===========================================================================
def test_random_combination(cfg, ctx):
    modes = ctx["modes"]; y_test = ctx["y_test"]; T_sim = ctx["T_sim"]
    n_amp = ctx["n_amp"]; n_batch = ctx["n_batch"]
    RMS_trad = np.zeros(n_batch); RMS_prop = np.zeros(n_batch)
    wf_weights = get_zernike_decay_weights(cfg.n_wf_modes, cfg)
    for index in tqdm(range(n_batch), desc="随机组合测试", unit="样本"):
        coe = cfg.wf_coeff_std * wf_weights * np.random.randn(cfg.n_wf_modes)
        A0 = reconstruct_from_zernike(y_test[index, :n_amp], modes) + y_test[index, -1]
        A1 = reconstruct_from_zernike(T_sim[index, :n_amp], modes) + T_sim[index, -1]
        RMS_trad[index], RMS_prop[index] = run_single_test(
            ctx["hs"], ctx["mask"], ctx["Recon"], cfg, A0, A1, coe, modes)
    return RMS_trad, RMS_prop


def test_fixed_intensity(cfg, ctx):
    modes = ctx["modes"]; y_test = ctx["y_test"]; T_sim = ctx["T_sim"]
    n_amp = ctx["n_amp"]; n_batch = ctx["n_batch"]
    A0_fixed = reconstruct_from_zernike(y_test[0, :n_amp], modes) + y_test[0, -1]
    A1_fixed = reconstruct_from_zernike(T_sim[0, :n_amp], modes) + T_sim[0, -1]
    print(f"  固定光强取自测试样本 #0")
    RMS_trad = np.zeros(n_batch); RMS_prop = np.zeros(n_batch)
    wf_weights = get_zernike_decay_weights(cfg.n_wf_modes, cfg)
    for index in tqdm(range(n_batch), desc="固定光强测试", unit="样本"):
        coe = cfg.wf_coeff_std * wf_weights * np.random.randn(cfg.n_wf_modes)
        RMS_trad[index], RMS_prop[index] = run_single_test(
            ctx["hs"], ctx["mask"], ctx["Recon"], cfg, A0_fixed, A1_fixed, coe, modes)
    return RMS_trad, RMS_prop


def test_fixed_wavefront(cfg, ctx):
    modes = ctx["modes"]; y_test = ctx["y_test"]; T_sim = ctx["T_sim"]
    n_amp = ctx["n_amp"]; n_batch = ctx["n_batch"]
    wf_weights = get_zernike_decay_weights(cfg.n_wf_modes, cfg)
    coe_fixed = cfg.wf_coeff_std * wf_weights * np.random.randn(cfg.n_wf_modes)
    print(f"  固定波前已生成 (coeff_std={cfg.wf_coeff_std})")
    RMS_trad = np.zeros(n_batch); RMS_prop = np.zeros(n_batch)
    for index in tqdm(range(n_batch), desc="固定波前测试", unit="样本"):
        A0 = reconstruct_from_zernike(y_test[index, :n_amp], modes) + y_test[index, -1]
        A1 = reconstruct_from_zernike(T_sim[index, :n_amp], modes) + T_sim[index, -1]
        RMS_trad[index], RMS_prop[index] = run_single_test(
            ctx["hs"], ctx["mask"], ctx["Recon"], cfg, A0, A1, coe_fixed, modes)
    return RMS_trad, RMS_prop


# ===========================================================================
# 结果可视化
# ===========================================================================
def visualize_and_save(RMS_trad, RMS_prop, cfg, mode_label, mode_name):
    n_batch = len(RMS_trad)
    mean_trad = np.mean(RMS_trad); mean_prop = np.mean(RMS_prop)
    improvement = (mean_trad - mean_prop) / mean_trad * 100 if mean_trad > 0 else 0.0

    print(f"\n{'=' * 50}")
    print(f" 测试结果 [{mode_name}]")
    print(f"{'=' * 50}")
    print(f"  传统方法平均 RMS: {mean_trad:.4f} lambda")
    print(f"  本方法平均 RMS:   {mean_prop:.4f} lambda")
    print(f"  提升幅度:         {improvement:.1f}%")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_range = np.arange(n_batch)
    ax.plot(x_range, RMS_trad, "b-", linewidth=1, alpha=0.7, label="传统方法")
    ax.plot(x_range, RMS_prop, "r-", linewidth=1, alpha=0.7, label="本方法")
    ax.axhline(mean_trad, color="b", linestyle="--", linewidth=1.2,
               label=f"传统方法均值 = {mean_trad:.4f}lambda")
    ax.axhline(mean_prop, color="r", linestyle="--", linewidth=1.2,
               label=f"本方法均值 = {mean_prop:.4f}lambda")
    ax.set_xlabel("测试样本序号", fontsize=13)
    ax.set_ylabel("波前残差 RMS (lambda)", fontsize=13)
    ax.set_title(f"批量波前复原对比 [{mode_name}] (提升 {improvement:.1f}%)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(cfg.result_dir, f"batch_test_{mode_label}.png")
    plt.savefig(fig_path, dpi=150)
    print(f"  图表已保存: {fig_path}")
    plt.show()

    npz_path = os.path.join(cfg.result_dir, f"batch_test_{mode_label}.npz")
    np.savez(npz_path, RMS_trad=RMS_trad, RMS_prop=RMS_prop,
             mean_trad=mean_trad, mean_prop=mean_prop,
             improvement=improvement, mode=mode_label)
    print(f"  数据已保存: {npz_path}")


TEST_DISPATCH = {"rc": test_random_combination, "fi": test_fixed_intensity, "fw": test_fixed_wavefront}


def parse_args():
    parser = argparse.ArgumentParser(description="批量波前复原测试")
    parser.add_argument("mode", choices=["rc", "fi", "fw"], default="rc", nargs="?",
                        help="测试模式: rc/fi/fw")
    return parser.parse_args()


def main():
    args = parse_args()
    mode_label = args.mode
    mode_meta = MODE_META[mode_label]
    print(f"\n{'=' * 60}")
    print(f"  测试模式: {mode_meta['name']} ({mode_label})")
    print(f"  {mode_meta['desc']}")
    print(f"{'=' * 60}")

    cfg = get_config()
    ctx = setup_system(cfg)
    test_func = TEST_DISPATCH[mode_label]
    RMS_trad, RMS_prop = test_func(cfg, ctx)
    visualize_and_save(RMS_trad, RMS_prop, cfg, mode_label, mode_meta["name"])


if __name__ == "__main__":
    main()
