"""
超参数优化主脚本: 不同激活函数 × 神经元数的 MSE 对比

对比 ELM 在不同激活函数 (ReLU, Sigmoid, Tanh, Softplus) 和
隐藏层神经元数下的性能，帮助选择最优超参数组合。

支持多次运行取平均 ± 标准差，生成带有置信区间的平滑曲线。

MATLAB 对应: Main_optimization.m
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.elm import ELM
from src.utils import (
    load_mat,
    set_seed,
    normalize_data,
    apply_normalize,
    reverse_normalize,
    compute_mse,
    split_data,
    configure_chinese_font,
    get_data_filename,
)
from scipy.ndimage import gaussian_filter1d


# ===========================================================================
# 参数配置
# ===========================================================================
@dataclass
class Config:
    """超参数优化配置"""

    # ---- 实验方法 ----
    methods: tuple = ("Relu", "Sigmoid", "Tanh", "Softplus")

    # ---- 数据参数 ----
    n_zernike: int = 25
    flag_noise: bool = True
    flag_wf: bool = True
    test_ratio: float = 0.1

    # ---- 搜索参数 ----
    hidden_range: tuple = (50, 1000, 25)     # (起始, 结束, 步长)
    n_runs: int = 5                          # 每个配置的重复次数

    # ---- 随机种子 ----
    base_seed: int = 42

    # ---- 路径 ----
    accessories_dir: str = "accessories"
    data_dir: str = "data"
    figure_data_dir: str = "figure_data"
    result_dir: str = "result"


def main():
    cfg = Config()
    set_seed(cfg.base_seed)
    configure_chinese_font()
    os.makedirs(cfg.figure_data_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ---- 1. 加载数据 ----
    print("加载数据...")
    name = get_data_filename(cfg.n_zernike, cfg.flag_noise, cfg.flag_wf)
    InputData = load_mat(os.path.join(cfg.data_dir, f"InputData{name}.mat"))
    OutputData = load_mat(os.path.join(cfg.data_dir, f"OutputData{name}.mat"))
    modes = load_mat(os.path.join(cfg.accessories_dir, "modes250.mat"), "modes")

    nZer = OutputData.shape[0] - 1
    zer_indices = list(range(nZer)) + [OutputData.shape[0] - 1]
    X = InputData
    y = OutputData[zer_indices, :]

    # 数据划分与归一化
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_ratio=cfg.test_ratio, shuffle=False
    )
    X_norm, y_norm, scaler_X, scaler_y = normalize_data(X_train, y_train)
    X_test_norm = apply_normalize(X_test, scaler_X)

    # ---- 2. 主搜索循环 ----
    hidden_list = list(
        range(cfg.hidden_range[0], cfg.hidden_range[1] + 1, cfg.hidden_range[2])
    )
    n_methods = len(cfg.methods)
    n_hidden_vals = len(hidden_list)

    MSE_mean = np.zeros((n_methods, n_hidden_vals))
    MSE_std = np.zeros((n_methods, n_hidden_vals))

    print(f"搜索 {n_methods} 种激活函数 × {n_hidden_vals} 个神经元数 × {cfg.n_runs} 次重复...")
    pbar_methods = tqdm(cfg.methods, desc="优化进度", unit="方法")
    for m_idx, method in enumerate(pbar_methods):
        pbar_methods.set_postfix({"当前方法": method})
        for h_idx, n_hid in enumerate(hidden_list):
            mse_runs = np.zeros(cfg.n_runs)
            for r in range(cfg.n_runs):
                set_seed(cfg.base_seed + r * 1000)
                elm = ELM(n_hidden=n_hid, activation=method.lower())
                elm.fit(X_norm, y_norm)
                pred_norm = elm.predict(X_test_norm)
                pred = reverse_normalize(pred_norm, scaler_y)
                mse_runs[r] = compute_mse(y_test, pred)
            MSE_mean[m_idx, h_idx] = np.mean(mse_runs)
            MSE_std[m_idx, h_idx] = np.std(mse_runs)

    # ---- 3. 绘图 (投稿级) ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    best_vals = np.zeros(n_methods)
    best_neus = np.zeros(n_methods)
    legend_handles = []

    for m_idx, method in enumerate(cfg.methods):
        # 高斯平滑
        mean_curve = gaussian_filter1d(MSE_mean[m_idx], sigma=2.0)
        std_curve = gaussian_filter1d(MSE_std[m_idx], sigma=2.0)

        upper = mean_curve + std_curve
        lower = mean_curve - std_curve
        lower = np.maximum(lower, np.min(mean_curve) * 0.1)

        # 置信区间填充
        ax.fill_between(
            hidden_list, upper, lower,
            color=colors[m_idx], alpha=0.15, edgecolor="none",
        )

        # 主曲线
        (line,) = ax.plot(
            hidden_list, mean_curve, "-",
            color=colors[m_idx], linewidth=1.6, label=method,
        )

        # 找最优点
        best_idx = np.argmin(MSE_mean[m_idx])
        best_vals[m_idx] = MSE_mean[m_idx, best_idx]
        best_neus[m_idx] = hidden_list[best_idx]

        ax.plot(
            best_neus[m_idx], mean_curve[best_idx], "p",
            color=colors[m_idx], markersize=9, markeredgecolor=colors[m_idx],
        )
        ax.text(
            best_neus[m_idx] - 60, mean_curve[best_idx] * 1.15,
            f"N={int(best_neus[m_idx])}\nMSE={best_vals[m_idx]:.2e}",
            fontsize=9, color=colors[m_idx],
        )
        legend_handles.append(line)

    ax.legend(legend_handles, cfg.methods, loc="best", frameon=False)
    ax.set_xlabel("隐藏层神经元数", fontsize=13)
    ax.set_ylabel("MSE", fontsize=13)
    ax.set_title("不同激活函数性能对比 (均值±标准差)", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(hidden_list[0] - 30, hidden_list[-1] + 30)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(cfg.figure_data_dir, "optimization_comparison.png"), dpi=300
    )
    plt.savefig(
        os.path.join(cfg.figure_data_dir, "optimization_comparison.pdf")
    )
    plt.show()

    # ---- 4. 输出最优结果 ----
    print("\n===== 最优结果 =====")
    for m_idx, method in enumerate(cfg.methods):
        print(
            f"  {method}: 最优神经元数 = {int(best_neus[m_idx])}, "
            f"MSE = {best_vals[m_idx]:.4e}"
        )

    # ---- 5. 保存数据 ----
    result_data = {
        "hidden_list": hidden_list,
        "MSE_mean": MSE_mean,
        "MSE_std": MSE_std,
        "best_neus": best_neus,
        "best_vals": best_vals,
        "methods": list(cfg.methods),
    }
    np.savez(
        os.path.join(cfg.figure_data_dir, "optimization_data.npz"), **result_data
    )
    print(f"\n数据已保存至 '{cfg.figure_data_dir}/optimization_data.npz'")


if __name__ == "__main__":
    main()
