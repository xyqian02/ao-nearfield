"""
实验数据 ELM 预测与波前复原

对实验采集的哈特曼阵列图像，使用 ELM 预测光强泽尼克系数，
进而通过差分斜率法分离光强-波前耦合，实现波前复原对比。

用法:
    python main_experiment.py                                          # 基础运行
    python main_experiment.py --normalize                               # 标定归一化
    python main_experiment.py --enhanced-data                           # 增强训练数据
    python main_experiment.py --ratio-features                          # 比强度特征
    python main_experiment.py --post-calibrate --n-calib 20             # 线性后校准
    python main_experiment.py -n -r -p -n 20                            # 全优化 (简写)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import get_config, SystemConfig
from src.optics import Optics
from src.elm import ELM
from src.hartmann import HartmannSensor
from src.evaluation import (
    compute_mse, compute_rms_wavefront, compute_pv_wavefront,
)
from src.utils import (
    load_mat, save_mat, set_seed, configure_chinese_font, get_data_filename,
    reconstruct_from_zernike, load_accessories, generate_mask,
    preprocess_experimental_image, extract_sub_spot_centroids,
    extract_sub_ap_total_intensity, plot_hartmann_grid,
    generate_simulation_data,
)


# =========================================================================
# 命令行参数
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="实验数据 ELM 预测与波前复原")
    p.add_argument("-n", "--normalize", action="store_true",
                   help="标定图像归一化 (方案1)")
    p.add_argument("-e", "--enhanced-data", action="store_true",
                   help="增强仿真训练数据 (方案2)")
    p.add_argument("-r", "--ratio-features", action="store_true",
                   help="比强度特征 (方案3)")
    p.add_argument("-p", "--post-calibrate", action="store_true",
                   help="波前系数线性后校准 (方案4)")
    p.add_argument("--n-calib", type=int, default=20,
                   help="校准样本数 (配合 -p，默认 20)")
    p.add_argument("--n-test", type=int, default=100,
                   help="测试样本数 (默认 100)")
    p.add_argument("--n-samples", type=int, default=10000,
                   help="仿真训练样本数 (默认 10000)")
    p.add_argument("--model-dir", type=str, default="models",
                   help="模型保存目录 (默认 models/)")
    p.add_argument("--exp-dir", type=str,
                   default="data/experimental_data/250616",
                   help="实验数据目录")
    p.add_argument("--truth-file", type=str,
                   default="data/experimental_data/f1000d_r012.mat",
                   help="波前真值 .mat 文件")
    # 可视化
    p.add_argument("--show", type=int, default=None, metavar="N",
                   help="可视化第 N 帧实验图像 (0=标定帧, 1~N=测试帧)")
    p.add_argument("--animate", type=str, default=None, metavar="M-N",
                   help="生成第 M 到 N 帧的连续动画 (如 --animate 1-50)")
    p.add_argument("--fps", type=int, default=10,
                   help="动画帧率 (默认 10)")
    return p.parse_args()


# =========================================================================
# 4种可选方案实现
# =========================================================================


# --- 方案1: 标定归一化 ---
def apply_calibration_normalization(
    intensities: np.ndarray, calib_intensity: np.ndarray, eps: float = 1e-8,
) -> np.ndarray:
    """子孔径强度除以标定图像对应值"""
    return intensities / (calib_intensity + eps)


# --- 方案3: 比强度特征 ---
def compute_ratio_features(intensity: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """I_i / mean(I)，对全局亮度变化不敏感"""
    return intensity / (np.mean(intensity) + eps)


# --- 方案4: 波前系数后校准 ---
def fit_wavefront_calibration(
    reco_coeffs: np.ndarray, true_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    用 Ridge 回归拟合 reco → true 的线性校正

    参数:
        reco_coeffs: (n_calib, n_wf) 复原波前系数
        true_coeffs: (n_calib, n_wf) 真值波前系数

    返回:
        (A, b): corrected = A @ x + b
    """
    from sklearn.linear_model import Ridge
    n_modes = reco_coeffs.shape[1]
    model = Ridge(alpha=0.1, fit_intercept=True)
    model.fit(reco_coeffs, true_coeffs)
    return model.coef_, model.intercept_


def apply_calibration(
    reco_coeffs: np.ndarray, A: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """应用线性校准"""
    return reco_coeffs @ A.T + b


# =========================================================================
# ELM 训练与模型管理
# =========================================================================
def get_model_paths(model_dir: str, suffix: str) -> tuple[str, str, str]:
    """获取模型文件路径"""
    os.makedirs(model_dir, exist_ok=True)
    elm_path = os.path.join(model_dir, f"elm_{suffix}.npz")
    sx_path = os.path.join(model_dir, f"scaler_X_{suffix}.joblib")
    sy_path = os.path.join(model_dir, f"scaler_y_{suffix}.joblib")
    return elm_path, sx_path, sy_path


def train_or_load_elm(
    X: np.ndarray, y: np.ndarray, cfg: SystemConfig,
    model_dir: str, suffix: str,
) -> tuple[ELM, MinMaxScaler, MinMaxScaler]:
    """训练 ELM 或从缓存加载"""
    elm_path, sx_path, sy_path = get_model_paths(model_dir, suffix)

    if os.path.exists(elm_path):
        print(f"  加载已保存的 ELM 模型: {elm_path}")
        elm = ELM.load(elm_path)
        scaler_X = joblib.load(sx_path)
        scaler_y = joblib.load(sy_path)
        return elm, scaler_X, scaler_y

    print("  训练 ELM 模型...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=cfg.test_ratio, shuffle=False,
    )
    val_size = cfg.val_ratio / (1 - cfg.test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, shuffle=False,
    )

    scaler_X = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(y_train)
    X_train_norm = scaler_X.transform(X_train)
    y_train_norm = scaler_y.transform(y_train)
    X_val_norm = scaler_X.transform(X_val)

    # 搜索最优神经元数
    if cfg.n_hidden is None:
        hidden_list = list(range(
            cfg.hidden_range[0], cfg.hidden_range[1] + 1, cfg.hidden_range[2],
        ))
        mse_list = np.zeros(len(hidden_list))
        for idx, n_hid in enumerate(
            tqdm(hidden_list, desc="  搜索最优神经元数", unit="个")
        ):
            elm_tmp = ELM(n_hidden=n_hid, activation=cfg.activation,
                          random_state=cfg.seed)
            elm_tmp.fit(X_train_norm, y_train_norm)
            y_val_pred = scaler_y.inverse_transform(
                elm_tmp.predict(X_val_norm).reshape(-1, y_train_norm.shape[1])
            )
            mse_list[idx] = compute_mse(y_val, y_val_pred)
        best_hidden = hidden_list[np.argmin(mse_list)]
        print(f"  最优神经元数: {best_hidden}, 验证集 MSE={np.min(mse_list):.4e}")
    else:
        best_hidden = cfg.n_hidden

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.vstack([y_train, y_val])
    X_train_full_norm = scaler_X.transform(X_train_full)
    y_train_full_norm = scaler_y.transform(y_train_full)

    elm = ELM(n_hidden=best_hidden, activation=cfg.activation,
              random_state=cfg.seed)
    elm.fit(X_train_full_norm, y_train_full_norm)

    # 保存模型
    elm.save(elm_path)
    joblib.dump(scaler_X, sx_path)
    joblib.dump(scaler_y, sy_path)
    print(f"  模型已保存: {elm_path}")

    return elm, scaler_X, scaler_y


# =========================================================================
# 单样本波前复原测试
# =========================================================================
def process_single_sample(
    hs: HartmannSensor, mask: np.ndarray, Recon: np.ndarray,
    modes: np.ndarray, cfg: SystemConfig,
    measured_slopes: np.ndarray, A_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    单样本波前复原: 传统方法 + 本方法

    返回:
        recoe_trad: 传统方法复原系数 (n_wf,)
        recoe_prop: 本方法复原系数 (n_wf,)
    """
    n_wf = cfg.n_wf_modes

    # 强度诱导斜率 (仿真)
    intensity_slopes, _ = hs.measure_slopes(A_pred, noise_sigma=0.0)

    # 传统方法: 直接复原
    recoe_trad = hs.reconstruct(measured_slopes, Recon)
    recoe_trad[:cfg.wf_skip_count] = 0.0

    # 本方法: 差分斜率复原
    wf_slopes = measured_slopes - intensity_slopes
    recoe_prop = hs.reconstruct(wf_slopes, Recon)
    recoe_prop[:cfg.wf_skip_count] = 0.0

    return recoe_trad, recoe_prop


# =========================================================================
# 结果可视化
# =========================================================================
def visualize_results(
    rms_trad: np.ndarray, rms_prop: np.ndarray,
    pv_trad: np.ndarray, pv_prop: np.ndarray,
    cfg: SystemConfig, args,
):
    """汇总统计与可视化"""
    out_dir = os.path.join(cfg.result_dir, "experiment")
    os.makedirs(out_dir, exist_ok=True)

    mean_rms_t = np.mean(rms_trad)
    mean_rms_p = np.mean(rms_prop)
    mean_pv_t = np.mean(pv_trad)
    mean_pv_p = np.mean(pv_prop)
    impr_rms = (mean_rms_t - mean_rms_p) / mean_rms_t * 100 if mean_rms_t > 0 else 0
    impr_pv = (mean_pv_t - mean_pv_p) / mean_pv_t * 100 if mean_pv_t > 0 else 0

    print(f"\n{'=' * 55}")
    print(f"  波前复原结果汇总")
    print(f"{'=' * 55}")
    print(f"  传统方法: RMS = {mean_rms_t:.4f}λ, PV = {mean_pv_t:.4f}λ")
    print(f"  本方法:   RMS = {mean_rms_p:.4f}λ, PV = {mean_pv_p:.4f}λ")
    print(f"  RMS 提升: {impr_rms:.1f}%, PV 提升: {impr_pv:.1f}%")

    # 构建策略标签
    strategies = []
    if args.normalize: strategies.append("标定归一化")
    if args.enhanced_data: strategies.append("增强数据")
    if args.ratio_features: strategies.append("比强度特征")
    if args.post_calibrate: strategies.append("后校准")
    strat_label = " + ".join(strategies) if strategies else "无优化"

    # ---- RMS 散点图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_test = len(rms_trad)
    x = np.arange(n_test)
    axes[0].plot(x, rms_trad, "b.-", linewidth=0.8, alpha=0.7, label="传统方法")
    axes[0].plot(x, rms_prop, "r.-", linewidth=0.8, alpha=0.7, label="本方法")
    axes[0].axhline(mean_rms_t, color="b", linestyle="--", linewidth=1.2,
                    label=f"传统均值={mean_rms_t:.4f}λ")
    axes[0].axhline(mean_rms_p, color="r", linestyle="--", linewidth=1.2,
                    label=f"本方法均值={mean_rms_p:.4f}λ")
    axes[0].set_xlabel("测试样本序号")
    axes[0].set_ylabel("波前残差 RMS (λ)")
    axes[0].set_title(f"RMS 对比 [{strat_label}]\n提升 {impr_rms:.1f}%")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, pv_trad, "b.-", linewidth=0.8, alpha=0.7, label="传统方法")
    axes[1].plot(x, pv_prop, "r.-", linewidth=0.8, alpha=0.7, label="本方法")
    axes[1].axhline(mean_pv_t, color="b", linestyle="--", linewidth=1.2,
                    label=f"传统均值={mean_pv_t:.4f}λ")
    axes[1].axhline(mean_pv_p, color="r", linestyle="--", linewidth=1.2,
                    label=f"本方法均值={mean_pv_p:.4f}λ")
    axes[1].set_xlabel("测试样本序号")
    axes[1].set_ylabel("波前残差 PV (λ)")
    axes[1].set_title(f"PV 对比 [{strat_label}]\n提升 {impr_pv:.1f}%")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rms_pv_comparison.png"), dpi=150)
    plt.show()

    # ---- 保存定量结果 ----
    np.savez(
        os.path.join(out_dir, "results.npz"),
        rms_trad=rms_trad, rms_prop=rms_prop,
        pv_trad=pv_trad, pv_prop=pv_prop,
        mean_rms_trad=mean_rms_t, mean_rms_prop=mean_rms_p,
        mean_pv_trad=mean_pv_t, mean_pv_prop=mean_pv_p,
        improvement_rms=impr_rms, improvement_pv=impr_pv,
        strategies=strat_label,
    )
    print(f"  结果已保存至: {out_dir}/")


def visualize_single_sample(
    true_wf: np.ndarray, wf_trad: np.ndarray, wf_prop: np.ndarray,
    mask: np.ndarray, sample_idx: int, cfg: SystemConfig,
    rms_t: float, rms_p: float, pv_t: float, pv_p: float,
):
    """单样本波前图对比"""
    out_dir = os.path.join(cfg.result_dir, "experiment")
    os.makedirs(out_dir, exist_ok=True)

    true_m = true_wf * mask
    trad_m = wf_trad * mask
    prop_m = wf_prop * mask

    vlim = max(np.max(np.abs(true_m)), 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    im0 = axes[0].imshow(true_m, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[0].set_title("真实波前 (真值)", fontsize=12)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(trad_m, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[1].set_title(f"传统方法\nRMS={rms_t:.4f}λ, PV={pv_t:.4f}λ", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(prop_m, cmap="jet", vmin=-vlim, vmax=vlim)
    axes[2].set_title(f"本方法\nRMS={rms_p:.4f}λ, PV={pv_p:.4f}λ", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(f"波前复原对比 (样本 #{sample_idx})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"wavefront_sample_{sample_idx:03d}.png"),
                dpi=150)
    plt.close(fig)


# =========================================================================
# 可视化模式
# =========================================================================
def run_show_mode(args, cfg, subcfg, mask, modes, optics, hs, elm, scaler_X, scaler_y):
    """三图并列对比: 实验 | 仿真全场(振幅+波前) | 仿真纯振幅"""
    configure_chinese_font()
    IS = cfg.image_size
    n_amp = cfg.n_amp_modes
    n_sub = subcfg.shape[1]
    pupil_radius = cfg.beam_size / 2
    out_dir = os.path.join(cfg.result_dir, "experiment")
    os.makedirs(out_dir, exist_ok=True)

    if elm is None:
        print("错误: --show 三图对比需要 ELM 模型，请先运行实验模式训练模型")
        return

    # 加载真值波前
    totalProj = load_mat(args.truth_file, "totalProj")  # (36, 100)
    frame_idx = args.show

    # 加载并预处理实验图像
    if frame_idx == 0:
        img_path = os.path.join(args.exp_dir, "0000000.bmp")
        frame_label = "标定帧"
        true_wf_coeffs = np.zeros(cfg.n_wf_modes)  # 标定帧无波前
    else:
        img_path = os.path.join(args.exp_dir, f"{frame_idx:07d}.bmp")
        frame_label = f"测试帧 #{frame_idx}"
        col = min(frame_idx - 1, totalProj.shape[1] - 1)
        true_wf_coeffs = totalProj[:cfg.n_wf_modes, col].copy()
        true_wf_coeffs[:cfg.wf_skip_count] = 0.0

    print(f"加载实验图像 {frame_label}: {img_path}")
    spot_img_exp = preprocess_experimental_image(img_path)

    # ELM 预测振幅
    intensity = extract_sub_ap_total_intensity(spot_img_exp, subcfg, cfg.sub_ap_pixels, IS)
    X_input = scaler_X.transform(intensity.reshape(1, -1))
    pred_norm = elm.predict(X_input).reshape(1, -1)
    pred_coeffs = scaler_y.inverse_transform(pred_norm).ravel()
    A_pred = reconstruct_from_zernike(pred_coeffs[:n_amp], modes) + pred_coeffs[n_amp]
    A_pred = np.maximum(A_pred, 0.0)

    # 标定哈特曼传感器
    hs.calibrate(mask, noise_sigma=0.0)

    # --- 生成仿真哈特曼图像 ---
    # 仿真全场: 振幅 + 真实波前
    true_wf = reconstruct_from_zernike(true_wf_coeffs, modes)
    field_full = A_pred * np.exp(-1j * true_wf)
    _, sim_full = hs.measure_slopes(field_full, noise_sigma=0.0)

    # 仿真纯振幅: 仅振幅，无波前
    _, sim_amp = hs.measure_slopes(A_pred, noise_sigma=0.0)

    # --- 1×3 画布 ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    plot_hartmann_grid(
        spot_img_exp, subcfg, cfg.sub_ap_pixels, IS,
        title=f"实验图像 — {frame_label}\n({n_sub} 有效子孔径)",
        ax=axes[0], pupil_radius=pupil_radius,
    )

    plot_hartmann_grid(
        sim_full, subcfg, cfg.sub_ap_pixels, IS,
        title="仿真全场 — A_pred × exp(−i·WF_true)\n(振幅 + 波前 → 衍射)",
        ax=axes[1], pupil_radius=pupil_radius,
    )

    plot_hartmann_grid(
        sim_amp, subcfg, cfg.sub_ap_pixels, IS,
        title="仿真纯振幅 — A_pred\n(仅振幅 → 衍射，无波前)",
        ax=axes[2], pupil_radius=pupil_radius,
    )

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"hartmann_compare_{frame_idx:03d}.png")
    plt.savefig(save_path, dpi=150)
    print(f"  已保存: {save_path}")
    if plt.isinteractive():
        plt.show()
    else:
        plt.close(fig)


def run_animate_mode(args, cfg, subcfg, mask, modes):
    """连续帧动画: 逐帧显示实验光斑图案变化"""
    configure_chinese_font()
    import matplotlib.animation as animation

    IS = cfg.image_size
    n_sub = subcfg.shape[1]
    out_dir = os.path.join(cfg.result_dir, "experiment")
    os.makedirs(out_dir, exist_ok=True)

    # 解析帧范围
    parts = args.animate.split("-")
    start_frame, end_frame = int(parts[0]), int(parts[1])
    n_frames = end_frame - start_frame + 1
    print(f"动画范围: 帧 {start_frame} ~ {end_frame} ({n_frames} 帧, {args.fps} fps)")

    # 预加载所有帧
    print("预加载图像...")
    frames = []
    for idx in tqdm(range(start_frame, end_frame + 1), desc="  加载", unit="帧"):
        if idx == 0:
            img_path = os.path.join(args.exp_dir, "0000000.bmp")
        else:
            img_path = os.path.join(args.exp_dir, f"{idx:07d}.bmp")
        frames.append(preprocess_experimental_image(img_path))

    # 创建动画
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(frames[0], cmap="jet")

    # 叠加网格 (静态)
    for i in range(n_sub):
        y1 = subcfg[0, i] + IS / 2
        x1 = subcfg[1, i] + IS / 2
        rect = plt.Rectangle(
            (x1, y1), cfg.sub_ap_pixels, cfg.sub_ap_pixels,
            fill=False, edgecolor="r", linewidth=0.5,
        )
        ax.add_patch(rect)

    ax.set_aspect("equal")
    title = ax.set_title(f"帧 {start_frame} / {end_frame}")
    plt.tight_layout()

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        actual_frame = start_frame + frame_idx
        title.set_text(f"帧 {actual_frame} / {end_frame}")
        return [im, title]

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // args.fps, blit=True,
    )

    # 保存: 优先 MP4，回退 GIF
    mp4_path = os.path.join(out_dir, f"hartmann_anim_{start_frame}-{end_frame}.mp4")
    gif_path = os.path.join(out_dir, f"hartmann_anim_{start_frame}-{end_frame}.gif")

    try:
        print("  尝试保存 MP4 (FFmpeg)...")
        ani.save(mp4_path, writer="ffmpeg", fps=args.fps, dpi=150)
        print(f"  已保存: {mp4_path}")
    except Exception:
        print("  FFmpeg 不可用，保存 GIF...")
        ani.save(gif_path, writer="pillow", fps=args.fps, dpi=150)
        print(f"  已保存: {gif_path}")

    if plt.isinteractive():
        plt.show()
    else:
        plt.close(fig)


# =========================================================================
# 主流程
# =========================================================================
def main():
    args = parse_args()
    cfg = get_config()
    set_seed(cfg.seed)
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ---- 可视化模式分支 ----
    if args.show is not None or args.animate is not None:
        print("=" * 55)
        print("  哈特曼阵列可视化")
        print("=" * 55)
        configure_chinese_font()
        os.makedirs(args.model_dir, exist_ok=True)
        IS = cfg.image_size

        # 加载配件
        print("加载配件...")
        subcfg, modes = load_accessories(cfg)
        mask = generate_mask(cfg)
        print(f"  子孔径: {subcfg.shape[1]}, 图像: {IS}x{IS}")

        optics = Optics(
            wavelength=cfg.wavelength, pixel_pitch=cfg.pixel_pitch,
            focal_length=cfg.focal_length, n_pixels=IS,
            sub_ap_pixels=cfg.sub_ap_pixels,
        )
        hs = HartmannSensor(subcfg, optics)

        # 尝试加载 ELM (仿真对比用)
        elm, scaler_X, scaler_y = None, None, None
        elm_path = os.path.join(args.model_dir, "elm_std.npz")
        if os.path.exists(elm_path):
            print(f"  加载 ELM: {elm_path}")
            elm = ELM.load(elm_path)
            scaler_X = joblib.load(os.path.join(args.model_dir, "scaler_X_std.joblib"))
            scaler_y = joblib.load(os.path.join(args.model_dir, "scaler_y_std.joblib"))
        else:
            print("  未找到 ELM 模型，跳过仿真对比")

        if args.show is not None:
            run_show_mode(args, cfg, subcfg, mask, modes, optics, hs,
                         elm, scaler_X, scaler_y)
        elif args.animate is not None:
            run_animate_mode(args, cfg, subcfg, mask, modes)

        print("\n完成!")
        return

    # ---- 正常实验模式 ----
    # 构建模型后缀 (区分不同策略)
    suffixes = []
    if args.enhanced_data: suffixes.append("enh")
    else: suffixes.append("std")
    if args.ratio_features: suffixes.append("ratio")
    model_suffix = "_".join(suffixes)

    # 打印策略信息
    print("=" * 55)
    print("  实验数据 ELM 预测与波前复原")
    print("=" * 55)
    print(f"  优化策略:")
    print(f"    标定归一化:    {'[ON]' if args.normalize else '[OFF]'}")
    print(f"    增强训练数据:  {'[ON]' if args.enhanced_data else '[OFF]'}")
    print(f"    比强度特征:    {'[ON]' if args.ratio_features else '[OFF]'}")
    print(f"    波前后校准:    {'[ON]' if args.post_calibrate else '[OFF]'}"
          + (f" (n={args.n_calib})" if args.post_calibrate else ""))
    print(f"    测试样本数:    {args.n_test}")
    print(f"    训练样本数:    {args.n_samples}")
    print(f"    模型后缀:      {model_suffix}")

    configure_chinese_font()
    os.makedirs(args.model_dir, exist_ok=True)

    IS = cfg.image_size

    # ---- 1. 加载配件 ----
    print("\n[1/7] 加载配件...")
    subcfg, modes = load_accessories(cfg)
    mask = generate_mask(cfg)
    n_sub = subcfg.shape[1]
    n_amp = cfg.n_amp_modes
    n_wf = cfg.n_wf_modes
    print(f"  子孔径: {n_sub}, 光强阶数: {n_amp}, 波前阶数: {n_wf}")

    # ---- 2. 训练或加载 ELM ----
    print("\n[2/7] 准备 ELM 模型...")
    optics = Optics(
        wavelength=cfg.wavelength, pixel_pitch=cfg.pixel_pitch,
        focal_length=cfg.focal_length, n_pixels=IS,
        sub_ap_pixels=cfg.sub_ap_pixels,
    )
    hs_tmp = HartmannSensor(subcfg, optics)

    elm_path = os.path.join(args.model_dir, f"elm_{model_suffix}.npz")

    if os.path.exists(elm_path):
        print("  加载已保存的 ELM 模型...")
        elm = ELM.load(elm_path)
        scaler_X = joblib.load(
            os.path.join(args.model_dir, f"scaler_X_{model_suffix}.joblib"))
        scaler_y = joblib.load(
            os.path.join(args.model_dir, f"scaler_y_{model_suffix}.joblib"))
    else:
        # 尝试加载已保存的实验仿真数据
        data_tag = f"_s{cfg.image_size}_a{cfg.n_amp_modes}"
        if args.enhanced_data:
            data_tag += "_enh"
        data_input_path = os.path.join(cfg.data_dir, f"InputData{data_tag}.mat")
        data_output_path = os.path.join(cfg.data_dir, f"OutputData{data_tag}.mat")

        if os.path.exists(data_input_path) and os.path.exists(data_output_path):
            print(f"  加载已保存的仿真数据: {data_input_path}")
            InputTrain = load_mat(data_input_path)
            OutputTrain = load_mat(data_output_path)
        else:
            print(f"  生成{'增强' if args.enhanced_data else '标准'}仿真训练数据...")
            InputTrain, OutputTrain = generate_simulation_data(
                cfg, modes, subcfg, optics, hs_tmp,
                n_samples=args.n_samples,
                enhanced=args.enhanced_data,
                seed=cfg.seed, show_progress=True,
            )
            save_mat(data_input_path, InputData=InputTrain)
            save_mat(data_output_path, OutputData=OutputTrain)
            print(f"  仿真数据已保存: {data_input_path}")

        # 对训练数据也应用比强度特征
        if args.ratio_features:
            InputTrain_ratio = np.zeros_like(InputTrain)
            for i in range(InputTrain.shape[0]):
                InputTrain_ratio[i] = compute_ratio_features(InputTrain[i])
            InputTrain = InputTrain_ratio

        elm, scaler_X, scaler_y = train_or_load_elm(
            InputTrain, OutputTrain, cfg, args.model_dir, model_suffix,
        )

    # ---- 3. 加载实验数据 ----
    print("\n[3/7] 加载实验数据...")
    # 标定图像
    calib_path = os.path.join(args.exp_dir, "0000000.bmp")
    print(f"  标定图像: {calib_path}")
    calib_img = preprocess_experimental_image(calib_path)

    # 提取标定质心和光强 (作为参考)
    calib_centroids = extract_sub_spot_centroids(
        calib_img, subcfg, cfg.sub_ap_pixels, IS)
    calib_intensity = extract_sub_ap_total_intensity(
        calib_img, subcfg, cfg.sub_ap_pixels, IS)

    # 波前真值
    print(f"  真值文件: {args.truth_file}")
    totalProj = load_mat(args.truth_file, "totalProj")  # (36, 100)
    print(f"  totalProj 形状: {totalProj.shape}")

    # ---- 4. 哈特曼传感器标定 + Z2S ----
    print("\n[4/7] 哈特曼传感器仿真标定...")
    hs = HartmannSensor(subcfg, optics)
    hs.calibrate(mask, noise_sigma=cfg.noise_sigma)

    print("  构建 Z2S 斜率响应矩阵...")
    Z2S = hs.build_response_matrix(modes, mask, n_wf, noise_sigma=cfg.noise_sigma)
    Recon = np.linalg.pinv(Z2S)
    print(f"  Z2S: {Z2S.shape}, Recon: {Recon.shape}")

    # ---- 5. 处理样本与波前复原 ----
    n_test = min(args.n_test, totalProj.shape[1])
    n_calib = min(args.n_calib, n_test // 2) if args.post_calibrate else 0

    print(f"\n[5/7] 处理测试样本 ({n_test} 个)...")
    if args.post_calibrate:
        print(f"  前 {n_calib} 个用于校准，剩余 {n_test - n_calib} 个评估")

    # 辅助函数: 处理单张图像，返回 (slopes, pred_coeffs, A_pred)
    def process_one_image(img_path):
        spot_img = preprocess_experimental_image(img_path)
        centroids = extract_sub_spot_centroids(spot_img, subcfg, cfg.sub_ap_pixels, IS)
        slopes = np.zeros(2 * n_sub)
        for i_sub in range(n_sub):
            slopes[2 * i_sub] = centroids[0, i_sub] - calib_centroids[0, i_sub]
            slopes[2 * i_sub + 1] = centroids[1, i_sub] - calib_centroids[1, i_sub]

        intensity = extract_sub_ap_total_intensity(spot_img, subcfg, cfg.sub_ap_pixels, IS)
        if args.normalize:
            intensity = apply_calibration_normalization(intensity, calib_intensity)
        if args.ratio_features:
            intensity = compute_ratio_features(intensity)

        X_input = scaler_X.transform(intensity.reshape(1, -1))
        pred_norm = elm.predict(X_input).reshape(1, -1)
        pred_coeffs = scaler_y.inverse_transform(pred_norm).ravel()

        A_pred = reconstruct_from_zernike(pred_coeffs[:n_amp], modes) + pred_coeffs[n_amp]
        A_pred = np.maximum(A_pred, 0.0)
        return slopes, pred_coeffs, A_pred

    # 5a. 处理校准样本
    reco_prop_calib_list = []
    true_coeffs_calib_list = []
    if args.post_calibrate:
        for idx in tqdm(range(n_calib), desc="  校准样本", unit="样本"):
            img_path = os.path.join(args.exp_dir, f"{idx + 1:07d}.bmp")
            measured_slopes, pred_coeffs, A_pred = process_one_image(img_path)
            recoe_trad, recoe_prop = process_single_sample(
                hs, mask, Recon, modes, cfg, measured_slopes, A_pred)
            true_wf_coeffs = totalProj[:n_wf, idx].copy()
            true_wf_coeffs[:cfg.wf_skip_count] = 0.0
            reco_prop_calib_list.append(recoe_prop)
            true_coeffs_calib_list.append(true_wf_coeffs)

    # 5b. 拟合后校准
    A_mat = None
    b_vec = None
    if args.post_calibrate:
        print("\n[6/7] 拟合波前系数后校准...")
        A_mat, b_vec = fit_wavefront_calibration(
            np.array(reco_prop_calib_list),
            np.array(true_coeffs_calib_list),
        )
        print(f"  校准矩阵 A: {A_mat.shape}, 偏置 b: {b_vec.shape}")
    else:
        print("\n[6/7] 跳过后校准")

    # 5c. 处理评估样本
    start_eval = n_calib if args.post_calibrate else 0
    n_eval = n_test - start_eval
    rms_trad_all = np.zeros(n_eval)
    rms_prop_all = np.zeros(n_eval)
    pv_trad_all = np.zeros(n_eval)
    pv_prop_all = np.zeros(n_eval)

    for eval_i, idx in enumerate(
        tqdm(range(start_eval, n_test), desc="  评估样本", unit="样本")
    ):
        img_path = os.path.join(args.exp_dir, f"{idx + 1:07d}.bmp")
        measured_slopes, pred_coeffs, A_pred = process_one_image(img_path)

        recoe_trad, recoe_prop = process_single_sample(
            hs, mask, Recon, modes, cfg, measured_slopes, A_pred)

        true_wf_coeffs = totalProj[:n_wf, idx].copy()
        true_wf_coeffs[:cfg.wf_skip_count] = 0.0
        true_wf = reconstruct_from_zernike(true_wf_coeffs, modes)

        if args.post_calibrate:
            recoe_prop = apply_calibration(recoe_prop, A_mat, b_vec)
            recoe_prop[:cfg.wf_skip_count] = 0.0
            # 也对传统方法应用相同校准
            recoe_trad = apply_calibration(recoe_trad, A_mat, b_vec)
            recoe_trad[:cfg.wf_skip_count] = 0.0

        wf_trad = reconstruct_from_zernike(recoe_trad, modes)
        wf_prop = reconstruct_from_zernike(recoe_prop, modes)

        rms_trad_all[eval_i] = compute_rms_wavefront(wf_trad, true_wf, mask)
        rms_prop_all[eval_i] = compute_rms_wavefront(wf_prop, true_wf, mask)
        pv_trad_all[eval_i] = compute_pv_wavefront(wf_trad, true_wf, mask)
        pv_prop_all[eval_i] = compute_pv_wavefront(wf_prop, true_wf, mask)

        if eval_i < 3:
            visualize_single_sample(
                true_wf, wf_trad, wf_prop, mask, eval_i, cfg,
                rms_trad_all[eval_i], rms_prop_all[eval_i],
                pv_trad_all[eval_i], pv_prop_all[eval_i],
            )

    # ---- 7. 可视化与保存 ----
    print("\n[7/7] 结果可视化...")
    visualize_results(
        rms_trad_all, rms_prop_all,
        pv_trad_all, pv_prop_all,
        cfg, args,
    )

    print("\n完成!")


if __name__ == "__main__":
    main()
