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
from dataclasses import dataclass
from tqdm import tqdm
from typing import Callable

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
    configure_chinese_font,
    get_data_filename,
    reconstruct_from_zernike,
    create_embedded_mask,
)


# ===========================================================================
# 参数配置
# ===========================================================================
@dataclass
class Config:
    """批量测试配置，所有可调参数集中在此"""

    # ---- 光学系统参数 ----
    wavelength: float = 1.064e-3
    focal_length: float = 12.0
    pixel_pitch: float = 14e-3
    sub_ap_pixels: int = 20
    image_size: int = 256

    # ---- 数据参数 ----
    n_zernike_amp: int = 25            # 光强泽尼克阶数
    flag_noise: bool = True
    flag_wf: bool = True

    # ---- 波前参数 ----
    n_wf_modes: int = 15
    wf_coeff_std: float = 0.2

    # ---- ELM 参数 ----
    activation: str = "softplus"
    n_hidden: int = 500

    # ---- 噪声 ----
    noise_sigma: float = 0.5

    # ---- 批量测试样本数 ----
    n_batch_samples: int | None = None  # None=全部测试集

    # ---- 随机种子 ----
    seed: int = 10

    # ---- 路径 ----
    accessories_dir: str = "accessories"
    data_dir: str = "data"
    result_dir: str = "result"


# ===========================================================================
# 模式元数据
# ===========================================================================
MODE_META = {
    "rc": {
        "name": "随机组合",
        "label": "rc",
        "desc": "随机波前 + 数据集光强，每组均不同",
    },
    "fi": {
        "name": "固定光强",
        "label": "fi",
        "desc": "同一光强分布，叠加随机波前",
    },
    "fw": {
        "name": "固定波前",
        "label": "fw",
        "desc": "同一波前分布，叠加数据集中的不同光强",
    },
}


# ===========================================================================
# 辅助函数
# ===========================================================================
def _rms_wavefront_lambda(
    wf_pred: np.ndarray,
    wf_true: np.ndarray,
    wavelength: float,
) -> float:
    """计算波前残差的 RMS 值 (λ 单位)"""
    residual = wf_pred - wf_true
    return float(np.std(residual) / (2 * np.pi) * wavelength * 1e3)


# ===========================================================================
# 系统初始化 (所有模式共享)
# ===========================================================================
def setup_system(cfg: Config) -> dict:
    """
    初始化仿真系统：加载数据、训练 ELM、标定哈特曼传感器

    返回包含所有共享对象的字典，各测试模式直接使用，
    避免重复加载和训练。
    """
    set_seed(cfg.seed)
    configure_chinese_font()
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ---- 加载数据 ----
    print("=" * 50)
    print(" 加载数据与初始化")
    print("=" * 50)
    name = get_data_filename(cfg.n_zernike_amp, cfg.flag_noise, cfg.flag_wf)
    subcfg = load_mat(os.path.join(cfg.accessories_dir, "Subcfg.mat"), "Subcfg")
    modes = load_mat(os.path.join(cfg.accessories_dir, "modes250.mat"), "modes")
    InputData = load_mat(os.path.join(cfg.data_dir, f"InputData{name}.mat"))
    OutputData = load_mat(os.path.join(cfg.data_dir, f"OutputData{name}.mat"))

    n_sub = subcfg.shape[1]
    nZer = OutputData.shape[0] - 1
    print(f"  子孔径数: {n_sub}, 泽尼克阶数: {nZer}")
    print(f"  输入数据: {InputData.shape}, 输出数据: {OutputData.shape}")

    # ---- 数据划分与 ELM 训练 ----
    zer_indices = list(range(nZer)) + [OutputData.shape[0] - 1]
    X = InputData
    y = OutputData[zer_indices, :]

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_ratio=0.1, shuffle=False
    )
    X_norm, y_norm, scaler_X, scaler_y = normalize_data(X_train, y_train)
    X_test_norm = apply_normalize(X_test, scaler_X)

    print(f"  训练 ELM (N={cfg.n_hidden}, activation={cfg.activation})...")
    elm = ELM(n_hidden=cfg.n_hidden, activation=cfg.activation)
    elm.fit(X_norm, y_norm)
    pred_norm = elm.predict(X_test_norm)
    T_sim = reverse_normalize(pred_norm, scaler_y)  # ELM预测系数

    # ---- 哈特曼传感器初始化与标定 ----
    optics = Optics(
        wavelength=cfg.wavelength,
        pixel_pitch=cfg.pixel_pitch,
        focal_length=cfg.focal_length,
        n_pixels=cfg.image_size,
        sub_ap_pixels=cfg.sub_ap_pixels,
    )
    hs = HartmannSensor(subcfg, optics)

    mask = create_embedded_mask(cfg.image_size, 240)

    print("  哈特曼传感器标定...")
    hs.calibrate(mask, noise_sigma=cfg.noise_sigma)

    print(f"  构建斜率响应矩阵 (前{cfg.n_wf_modes}阶)...")
    Z2S = hs.build_response_matrix(
        modes, mask, cfg.n_wf_modes, noise_sigma=cfg.noise_sigma
    )
    Recon = np.linalg.pinv(Z2S)

    # 计算测试样本数
    n_test = X_test.shape[1]
    n_batch = min(cfg.n_batch_samples or n_test, n_test)

    return {
        "subcfg": subcfg,
        "modes": modes,
        "n_sub": n_sub,
        "nZer": nZer,
        "y_test": y_test,
        "T_sim": T_sim,
        "optics": optics,
        "hs": hs,
        "mask": mask,
        "Recon": Recon,
        "n_batch": n_batch,
    }


# ===========================================================================
# 单次测试迭代 (核心逻辑复用)
# ===========================================================================
def run_single_test(
    hs: HartmannSensor,
    mask: np.ndarray,
    Recon: np.ndarray,
    cfg: Config,
    A0_240: np.ndarray,          # 真实光强分布 (240×240)
    A1_240: np.ndarray,          # ELM预测光强分布 (240×240)
    coe_wf: np.ndarray,          # 波前泽尼克系数
    modes: np.ndarray,
) -> tuple[float, float]:
    """
    执行单次波前复原测试

    对指定光强和波前组合，分别用传统方法和本方法复原波前，
    计算各自的 RMS 残差。

    返回:
        (rms_trad, rms_prop): 传统方法和本方法的 RMS 残差 (λ)
    """
    # 构建波前全场 (256×256)
    wf_full = np.zeros((cfg.image_size, cfg.image_size))
    wf_full[8:248, 8:248] = reconstruct_from_zernike(
        coe_wf, modes, cfg.n_wf_modes - 2, 2
    )

    # 传统方法: 真实光强 + 波前 → 斜率 → 波前复原
    IntensityMat = np.zeros_like(mask)
    IntensityMat[8:248, 8:248] = A0_240
    Input_trad = IntensityMat * np.exp(-1j * wf_full)
    slopes_int, _ = hs.measure_slopes(Input_trad, noise_sigma=cfg.noise_sigma)
    recoe = hs.reconstruct(slopes_int, Recon)

    # 本方法: ELM光强单独入射 → 差分斜率
    IntensityMat_R = np.zeros_like(mask)
    IntensityMat_R[8:248, 8:248] = A1_240
    Input_R = IntensityMat_R
    slopes_r, _ = hs.measure_slopes(Input_R, noise_sigma=cfg.noise_sigma)
    recoeR = hs.reconstruct(slopes_int - slopes_r, Recon)

    # 重构波前并计算残差
    wf_0 = reconstruct_from_zernike(coe_wf, modes, cfg.n_wf_modes - 2, 2)
    wf_1 = reconstruct_from_zernike(recoe, modes, cfg.n_wf_modes - 2, 2)
    wf_r = reconstruct_from_zernike(recoeR, modes, cfg.n_wf_modes - 2, 2)

    rms_trad = _rms_wavefront_lambda(wf_1, wf_0, cfg.wavelength)
    rms_prop = _rms_wavefront_lambda(wf_r, wf_0, cfg.wavelength)

    return rms_trad, rms_prop


# ===========================================================================
# 测试模式: 随机组合 (rc) — 原 MATLAB 行为
# ===========================================================================
def test_random_combination(cfg: Config, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    随机组合测试模式

    每次迭代随机生成波前 + 从数据集取不同光强分布。
    每组测试数据的光强和波前均不同。
    """
    modes = ctx["modes"]
    y_test = ctx["y_test"]
    T_sim = ctx["T_sim"]
    nZer = ctx["nZer"]
    n_batch = ctx["n_batch"]

    RMS_trad = np.zeros(n_batch)
    RMS_prop = np.zeros(n_batch)

    for index in tqdm(range(n_batch), desc="随机组合测试", unit="样本"):
        # 随机波前
        coe = cfg.wf_coeff_std * np.random.randn(cfg.n_wf_modes)

        # 数据集中的光强 (每个样本不同)
        A0_240 = reconstruct_from_zernike(y_test[:, index], modes, nZer, 0) + y_test[-1, index]
        A1_240 = reconstruct_from_zernike(T_sim[:, index], modes, nZer, 0) + T_sim[-1, index]

        RMS_trad[index], RMS_prop[index] = run_single_test(
            ctx["hs"], ctx["mask"], ctx["Recon"], cfg,
            A0_240, A1_240, coe, modes,
        )

    return RMS_trad, RMS_prop


# ===========================================================================
# 测试模式: 固定光强 (fi)
# ===========================================================================
def test_fixed_intensity(cfg: Config, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    固定光强测试模式

    从测试集第一个样本提取光强分布，在所有迭代中保持不变。
    每次迭代生成不同随机波前，用于测试波前复原算法对
    波前变化的敏感性，排除光强变化的干扰。
    """
    modes = ctx["modes"]
    y_test = ctx["y_test"]
    T_sim = ctx["T_sim"]
    nZer = ctx["nZer"]
    n_batch = ctx["n_batch"]

    # 提取固定光强分布 (使用测试集第一个样本)
    fixed_idx = 0
    A0_fixed = reconstruct_from_zernike(y_test[:, fixed_idx], modes, nZer, 0) + y_test[-1, fixed_idx]
    A1_fixed = reconstruct_from_zernike(T_sim[:, fixed_idx], modes, nZer, 0) + T_sim[-1, fixed_idx]
    print(f"  固定光强取自测试样本 #{fixed_idx}")

    RMS_trad = np.zeros(n_batch)
    RMS_prop = np.zeros(n_batch)

    for index in tqdm(range(n_batch), desc="固定光强测试", unit="样本"):
        # 随机波前
        coe = cfg.wf_coeff_std * np.random.randn(cfg.n_wf_modes)

        # 始终使用相同光强
        RMS_trad[index], RMS_prop[index] = run_single_test(
            ctx["hs"], ctx["mask"], ctx["Recon"], cfg,
            A0_fixed, A1_fixed, coe, modes,
        )

    return RMS_trad, RMS_prop


# ===========================================================================
# 测试模式: 固定波前 (fw)
# ===========================================================================
def test_fixed_wavefront(cfg: Config, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    固定波前测试模式

    在开始时生成一个随机波前，在所有迭代中保持不变。
    每次迭代使用数据集中不同的光强分布，
    用于测试光强分布变化对波前复原精度的影响。
    """
    modes = ctx["modes"]
    y_test = ctx["y_test"]
    T_sim = ctx["T_sim"]
    nZer = ctx["nZer"]
    n_batch = ctx["n_batch"]

    # 生成固定波前 (一次生成，全程复用)
    coe_fixed = cfg.wf_coeff_std * np.random.randn(cfg.n_wf_modes)
    print(f"  固定波前已生成 (coeff_std={cfg.wf_coeff_std})")

    RMS_trad = np.zeros(n_batch)
    RMS_prop = np.zeros(n_batch)

    for index in tqdm(range(n_batch), desc="固定波前测试", unit="样本"):
        # 数据集中的不同光强
        A0_240 = reconstruct_from_zernike(y_test[:, index], modes, nZer, 0) + y_test[-1, index]
        A1_240 = reconstruct_from_zernike(T_sim[:, index], modes, nZer, 0) + T_sim[-1, index]

        # 始终使用相同波前
        RMS_trad[index], RMS_prop[index] = run_single_test(
            ctx["hs"], ctx["mask"], ctx["Recon"], cfg,
            A0_240, A1_240, coe_fixed, modes,
        )

    return RMS_trad, RMS_prop


# ===========================================================================
# 结果可视化 (所有模式共享)
# ===========================================================================
def visualize_and_save(
    RMS_trad: np.ndarray,
    RMS_prop: np.ndarray,
    cfg: Config,
    mode_label: str,
    mode_name: str,
):
    """
    绘制并保存 RMS 残差对比图

    生成双曲线图，带均值参考线，文件名和标题包含模式标识。
    """
    n_batch = len(RMS_trad)
    mean_trad = np.mean(RMS_trad)
    mean_prop = np.mean(RMS_prop)
    improvement = (mean_trad - mean_prop) / mean_trad * 100 if mean_trad > 0 else 0.0

    print(f"\n{'=' * 50}")
    print(f" 测试结果 [{mode_name}]")
    print(f"{'=' * 50}")
    print(f"  传统方法平均 RMS: {mean_trad:.4f} λ")
    print(f"  本方法平均 RMS:   {mean_prop:.4f} λ")
    print(f"  提升幅度:         {improvement:.1f}%")

    # ---- 绘图 ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_range = np.arange(n_batch)

    ax.plot(x_range, RMS_trad, "b-", linewidth=1, alpha=0.7, label="传统方法")
    ax.plot(x_range, RMS_prop, "r-", linewidth=1, alpha=0.7, label="本方法")
    ax.axhline(
        mean_trad, color="b", linestyle="--", linewidth=1.2,
        label=f"传统方法均值 = {mean_trad:.4f}λ",
    )
    ax.axhline(
        mean_prop, color="r", linestyle="--", linewidth=1.2,
        label=f"本方法均值 = {mean_prop:.4f}λ",
    )
    ax.set_xlabel("测试样本序号", fontsize=13)
    ax.set_ylabel("波前残差 RMS (λ)", fontsize=13)
    ax.set_title(
        f"批量波前复原对比 [{mode_name}] (提升 {improvement:.1f}%)", fontsize=13
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存带模式标识的图片
    fig_path = os.path.join(cfg.result_dir, f"batch_test_{mode_label}.png")
    plt.savefig(fig_path, dpi=150)
    print(f"  图表已保存: {fig_path}")
    plt.show()

    # ---- 保存数值结果 ----
    npz_path = os.path.join(cfg.result_dir, f"batch_test_{mode_label}.npz")
    np.savez(
        npz_path,
        RMS_trad=RMS_trad,
        RMS_prop=RMS_prop,
        mean_trad=mean_trad,
        mean_prop=mean_prop,
        improvement=improvement,
        mode=mode_label,
    )
    print(f"  数据已保存: {npz_path}")


# ===========================================================================
# 测试模式调度表
# ===========================================================================
# 将模式标签映射到对应的测试函数，方便扩展新模式
TEST_DISPATCH: dict[str, Callable[[Config, dict], tuple[np.ndarray, np.ndarray]]] = {
    "rc": test_random_combination,
    "fi": test_fixed_intensity,
    "fw": test_fixed_wavefront,
}


# ===========================================================================
# 命令行参数解析
# ===========================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量波前复原测试 — 比较传统方法与ELM方法的波前复原精度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "模式说明:\n"
            "  rc  随机组合 - 波前和光强每组均不同 (原MATLAB行为)\n"
            "  fi  固定光强 - 同一光强 + 不同随机波前\n"
            "  fw  固定波前 - 同一波前 + 不同数据集光强\n\n"
            "示例:\n"
            "  python main_batch_test.py rc\n"
            "  python main_batch_test.py fi\n"
        ),
    )
    parser.add_argument(
        "mode",
        choices=["rc", "fi", "fw"],
        default="fw",
        nargs="?",
        help="测试模式: rc=随机组合, fi=固定光强, fw=固定波前 (默认: rc)",
    )
    return parser.parse_args()


# ===========================================================================
# 主函数
# ===========================================================================
def main():
    args = parse_args()
    mode_label = args.mode
    mode_meta = MODE_META[mode_label]

    print(f"\n{'=' * 60}")
    print(f"  测试模式: {mode_meta['name']} ({mode_label})")
    print(f"  {mode_meta['desc']}")
    print(f"{'=' * 60}")

    # 初始化系统 (所有模式共享)
    cfg = Config()
    ctx = setup_system(cfg)

    # 根据模式调度测试函数
    test_func = TEST_DISPATCH[mode_label]
    RMS_trad, RMS_prop = test_func(cfg, ctx)

    # 结果可视化与保存
    visualize_and_save(RMS_trad, RMS_prop, cfg, mode_label, mode_meta["name"])


if __name__ == "__main__":
    main()
