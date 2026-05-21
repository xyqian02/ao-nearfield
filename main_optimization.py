"""
超参数优化主脚本: 不同激活函数 × 神经元数的 MSE 对比

对比 ELM 在不同激活函数 (ReLU, Sigmoid, Tanh, Softplus) 和
隐藏层神经元数下的性能，帮助选择最优超参数组合。

支持多次运行取平均 ± 标准差，生成带有置信区间的平滑曲线。
使用验证集做超参选择，测试集做最终评估。

MATLAB 对应: Main_optimization.m
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
from src.elm import ELM
from src.evaluation import compute_mse, compute_rmse, compute_r2
from src.utils import (
    load_mat,
    set_seed,
    configure_chinese_font,
    get_data_filename,
)
from scipy.ndimage import gaussian_filter1d


def main():
    cfg = get_config()
    set_seed(cfg.seed)
    configure_chinese_font()
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ---- 1. 加载数据 ----
    print("加载数据...")
    name = get_data_filename(cfg)
    InputData = load_mat(os.path.join(cfg.data_dir, f"InputData{name}.mat"))
    OutputData = load_mat(os.path.join(cfg.data_dir, f"OutputData{name}.mat"))

    nZer = OutputData.shape[1] - 1
    X = InputData                       # (n_samples, n_features)
    y = OutputData                      # (n_samples, n_outputs)

    # ---- 2. 划分训练/验证/测试集 ----
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=cfg.test_ratio, shuffle=False
    )
    val_size = cfg.val_ratio / (1 - cfg.test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, shuffle=False
    )

    n_train = X_train.shape[0]
    print(f"  训练: {n_train}, 验证: {X_val.shape[0]}, 测试: {X_test.shape[0]}")

    # 归一化 (仅在训练集上拟合)
    scaler_X = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(y_train)
    X_train_norm = scaler_X.transform(X_train)
    y_train_norm = scaler_y.transform(y_train)
    X_val_norm = scaler_X.transform(X_val)

    # ---- 3. 主搜索循环 (验证集评估) ----
    hidden_list = list(
        range(cfg.opt_hidden_range[0], cfg.opt_hidden_range[1] + 1, cfg.opt_hidden_range[2])
    )
    n_methods = len(cfg.opt_methods)
    n_hidden_vals = len(hidden_list)

    MSE_mean = np.zeros((n_methods, n_hidden_vals))
    MSE_std = np.zeros((n_methods, n_hidden_vals))

    print(f"搜索 {n_methods} 种激活函数 × {n_hidden_vals} 个神经元数 × {cfg.opt_n_runs} 次重复 (验证集)...")
    pbar_methods = tqdm(cfg.opt_methods, desc="优化进度", unit="方法")
    for m_idx, method in enumerate(pbar_methods):
        pbar_methods.set_postfix({"当前方法": method})
        for h_idx, n_hid in enumerate(hidden_list):
            mse_runs = np.zeros(cfg.opt_n_runs)
            for r in range(cfg.opt_n_runs):
                elm = ELM(
                    n_hidden=n_hid,
                    activation=method.lower(),
                    random_state=cfg.seed + r * 1000,
                )
                elm.fit(X_train_norm, y_train_norm)
                y_val_pred_norm = elm.predict(X_val_norm)
                y_val_pred = scaler_y.inverse_transform(y_val_pred_norm)
                mse_runs[r] = compute_mse(y_val, y_val_pred)
            MSE_mean[m_idx, h_idx] = np.mean(mse_runs)
            MSE_std[m_idx, h_idx] = np.std(mse_runs)

    # ---- 4. 各方法最优神经元 → 测试集评估 ----
    best_neus = np.zeros(n_methods, dtype=int)
    best_val_mses = np.zeros(n_methods)
    test_results = {}

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.vstack([y_train, y_val])
    X_train_full_norm = scaler_X.transform(X_train_full)
    y_train_full_norm = scaler_y.transform(y_train_full)
    X_test_norm = scaler_X.transform(X_test)

    for m_idx, method in enumerate(cfg.opt_methods):
        best_idx = np.argmin(MSE_mean[m_idx])
        best_neus[m_idx] = hidden_list[best_idx]
        best_val_mses[m_idx] = MSE_mean[m_idx, best_idx]

        elm = ELM(
            n_hidden=int(best_neus[m_idx]),
            activation=method.lower(),
            random_state=cfg.seed,
        )
        elm.fit(X_train_full_norm, y_train_full_norm)
        y_test_pred_norm = elm.predict(X_test_norm)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_norm)

        test_results[method] = {
            "mse": compute_mse(y_test, y_test_pred),
            "rmse": compute_rmse(y_test, y_test_pred),
            "r2": compute_r2(y_test, y_test_pred),
            "best_n": int(best_neus[m_idx]),
        }

    # ---- 5. 绘图 (投稿级) ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    legend_handles = []
    for m_idx, method in enumerate(cfg.opt_methods):
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

        # 标注最优点 (验证集最优)
        best_idx = np.argmin(MSE_mean[m_idx])
        ax.plot(
            best_neus[m_idx], mean_curve[best_idx], "p",
            color=colors[m_idx], markersize=9, markeredgecolor=colors[m_idx],
        )
        ax.text(
            best_neus[m_idx] - 60, mean_curve[best_idx] * 1.15,
            f"N={int(best_neus[m_idx])}\nMSE={best_val_mses[m_idx]:.2e}",
            fontsize=9, color=colors[m_idx],
        )
        legend_handles.append(line)

    ax.legend(legend_handles, cfg.opt_methods, loc="best", frameon=False)
    ax.set_xlabel("隐藏层神经元数", fontsize=13)
    ax.set_ylabel("MSE (验证集)", fontsize=13)
    ax.set_title("不同激活函数性能对比 (均值±标准差)", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(hidden_list[0] - 30, hidden_list[-1] + 30)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(cfg.result_dir, "optimization_comparison.png"), dpi=300
    )
    plt.savefig(
        os.path.join(cfg.result_dir, "optimization_comparison.pdf")
    )
    plt.show()

    # ---- 6. 输出结果 ----
    print("\n===== 测试集最终评估 =====")
    for method in cfg.opt_methods:
        r = test_results[method]
        print(
            f"  {method}: N={r['best_n']}, "
            f"MSE={r['mse']:.4e}, RMSE={r['rmse']:.4e}, R²={r['r2']:.4f}"
        )

    global_best = min(test_results, key=lambda m: test_results[m]["mse"])
    print(f"\n  全局最优: {global_best} (测试集 MSE = {test_results[global_best]['mse']:.4e})")

    # ---- 7. 保存数据 ----
    result_data = {
        "hidden_list": hidden_list,
        "MSE_mean": MSE_mean,
        "MSE_std": MSE_std,
        "best_neus": best_neus,
        "best_val_mses": best_val_mses,
        "methods": np.array(cfg.opt_methods),
        "test_mse": np.array([test_results[m]["mse"] for m in cfg.opt_methods]),
        "test_rmse": np.array([test_results[m]["rmse"] for m in cfg.opt_methods]),
        "test_r2": np.array([test_results[m]["r2"] for m in cfg.opt_methods]),
    }
    np.savez(
        os.path.join(cfg.result_dir, "optimization_data.npz"), **result_data
    )
    print(f"\n数据已保存至 '{cfg.result_dir}/optimization_data.npz'")


if __name__ == "__main__":
    main()
