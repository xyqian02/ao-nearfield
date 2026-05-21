"""
ELM 训练与近场复原主脚本

使用极限学习机(ELM)从哈特曼子孔径光强预测光强泽尼克系数，
进而重构近场振幅分布。包含隐藏层神经元数优化和结果可视化。

核心流程:
1. 加载训练数据 (InputData / OutputData)
2. 划分训练集(80%)、验证集(10%)、测试集(10%)
3. 数据归一化 (MinMaxScaler → [-1, 1])
4. 在验证集上网格搜索最优隐藏层神经元数
5. 在训练+验证集上用最优神经元数训练 ELM
6. 测试集评估: 预测系数 → 重构近场 → 可视化对比

MATLAB 对应: Main_A_ELM.m
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
    circ_mask,
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

    # ---- 1. 加载数据 ----
    print("加载数据...")
    name = get_data_filename(cfg)
    InputData = load_mat(os.path.join(cfg.data_dir, f"InputData{name}.mat"))
    OutputData = load_mat(os.path.join(cfg.data_dir, f"OutputData{name}.mat"))
    _, modes = load_accessories(cfg)

    n_amp = OutputData.shape[1] - 1  # 光强泽尼克阶数
    IS = cfg.image_size
    print(f"  输入维度: {InputData.shape}, 输出维度: {OutputData.shape}")
    print(f"  光强泽尼克阶数: {n_amp}, 图像尺寸: {IS}×{IS}")

    # ---- 2. 准备数据 ----
    X = InputData                       # (n_samples, n_features)
    y = OutputData                      # (n_samples, n_outputs)

    # ---- 3. 划分训练/验证/测试集 ----
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=cfg.test_ratio, shuffle=False
    )
    val_size = cfg.val_ratio / (1 - cfg.test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, shuffle=False
    )

    n_train, n_val, n_test = X_train.shape[0], X_val.shape[0], X_test.shape[0]
    print(f"  训练样本: {n_train}, 验证样本: {n_val}, 测试样本: {n_test}")

    # ---- 4. 数据归一化 ----
    scaler_X = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(y_train)

    X_train_norm = scaler_X.transform(X_train)
    y_train_norm = scaler_y.transform(y_train)
    X_val_norm = scaler_X.transform(X_val)

    # ---- 5. 网格搜索最优隐藏层神经元数 ----
    if cfg.n_hidden is None:
        print("搜索最优隐藏层神经元数 (验证集评估)...")
        hidden_list = list(
            range(cfg.hidden_range[0], cfg.hidden_range[1] + 1, cfg.hidden_range[2])
        )
        mse_list = np.zeros(len(hidden_list))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plt.ion()
        pbar = tqdm(hidden_list, desc="搜索最优神经元数", unit="个")
        for idx, n_hid in enumerate(pbar):
            elm = ELM(n_hidden=n_hid, activation=cfg.activation, random_state=cfg.seed)
            elm.fit(X_train_norm, y_train_norm)
            y_val_pred_norm = elm.predict(X_val_norm)
            y_val_pred = scaler_y.inverse_transform(y_val_pred_norm)
            mse_list[idx] = compute_mse(y_val, y_val_pred)
            pbar.set_postfix({"val_MSE": f"{mse_list[idx]:.4e}"})

            ax.clear()
            ax.plot(hidden_list[: idx + 1], mse_list[: idx + 1], "b-", linewidth=1.5)
            ax.set_xlabel("隐藏层神经元数", fontsize=13)
            ax.set_ylabel("MSE (验证集)", fontsize=13)
            ax.set_title(f"激活函数: {cfg.activation}", fontsize=13)
            ax.grid(True, alpha=0.3)
            plt.pause(0.01)

        min_idx = np.argmin(mse_list)
        best_hidden = hidden_list[min_idx]
        best_val_mse = mse_list[min_idx]

        ax.clear()
        from scipy.interpolate import make_interp_spline
        x_smooth = np.linspace(hidden_list[0], hidden_list[-1], 300)
        spl = make_interp_spline(hidden_list, mse_list, k=3)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, "b-", linewidth=1.5)
        ax.plot(best_hidden, best_val_mse, "ro", markersize=8)
        ax.axvline(best_hidden, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(
            best_hidden, best_val_mse * 1.1,
            f"  N={best_hidden}\n  MSE={best_val_mse:.2e}",
            color="red", fontsize=10,
        )
        ax.set_xlabel("隐藏层神经元数", fontsize=13)
        ax.set_ylabel("MSE (验证集)", fontsize=13)
        ax.set_title(f"最优神经元搜索 ({cfg.activation})", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.result_dir, "neuron_search.png"), dpi=150)
        plt.ioff()
        plt.show()

        print(f"  最优隐藏层神经元数: {best_hidden}, 验证集 MSE = {best_val_mse:.4e}")
    else:
        best_hidden = cfg.n_hidden

    # ---- 6. 训练最终模型，测试集评估 ----
    print(f"训练最终 ELM (N={best_hidden}, activation={cfg.activation})...")
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.vstack([y_train, y_val])
    X_train_full_norm = scaler_X.transform(X_train_full)
    y_train_full_norm = scaler_y.transform(y_train_full)

    elm = ELM(n_hidden=best_hidden, activation=cfg.activation, random_state=cfg.seed)
    elm.fit(X_train_full_norm, y_train_full_norm)

    X_test_norm = scaler_X.transform(X_test)
    y_test_pred_norm = elm.predict(X_test_norm)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_norm)

    test_mse = compute_mse(y_test, y_test_pred)
    test_rmse = compute_rmse(y_test, y_test_pred)
    test_r2 = compute_r2(y_test, y_test_pred)
    print(f"  测试集 MSE = {test_mse:.4e}, RMSE = {test_rmse:.4e}, R² = {test_r2:.4f}")

    # ---- 7. 可视化: 泽尼克系数对比 ----
    idx = min(cfg.batch_test_index, n_test - 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x_orders = np.arange(1, n_amp + 1)
    ax1.plot(x_orders, y_test[idx, :n_amp], "o-", linewidth=1, label="真实值")
    ax1.plot(x_orders, y_test_pred[idx, :n_amp], "s-", linewidth=1, label="预测值")
    ax1.set_xlabel("泽尼克阶数", fontsize=13)
    ax1.set_ylabel("泽尼克系数", fontsize=13)
    ax1.set_title("光强泽尼克系数对比", fontsize=13)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    bar_width = 0.35
    ax2.bar(x_orders - bar_width / 2, y_test[idx, :n_amp], bar_width, label="真实值")
    ax2.bar(x_orders + bar_width / 2, y_test_pred[idx, :n_amp], bar_width, label="预测值")
    ax2.set_xlabel("泽尼克阶数", fontsize=13)
    ax2.set_ylabel("泽尼克系数", fontsize=13)
    ax2.set_title("光强泽尼克系数对比 (柱状图)", fontsize=13)
    ax2.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "coefficient_comparison.png"), dpi=150)
    plt.show()

    # ---- 8. 重构近场分布并对比 ----
    A0 = reconstruct_from_zernike(y_test[idx, :n_amp], modes)
    A1 = reconstruct_from_zernike(y_test_pred[idx, :n_amp], modes)

    A0 = A0 + y_test[idx, -1]
    A1 = A1 + y_test_pred[idx, -1]

    mask_circ = generate_mask(cfg)
    NF0 = A0 * mask_circ
    NF1 = A1 * mask_circ

    cMax = max(np.max(NF0), np.max(NF1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    im0 = axes[0].imshow(NF0, cmap="jet", vmin=0, vmax=cMax)
    axes[0].set_title("原始近场分布", fontsize=13)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(NF1, cmap="jet", vmin=0, vmax=cMax)
    axes[1].set_title("ELM 复原近场分布", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    diff = NF1 - NF0
    im2 = axes[2].imshow(diff, cmap="jet", vmin=-cMax, vmax=cMax)
    axes[2].set_title("残差 (复原 - 原始)", fontsize=13)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(f"{n_amp}阶泽尼克近场复原 (ELM, {cfg.activation})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "nearfield_2d_comparison.png"), dpi=150)
    plt.show()

    # 三维分布对比
    fig = plt.figure(figsize=(12, 5))
    X_m, Y_m = np.meshgrid(np.arange(IS), np.arange(IS))

    ax3d_0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d_0.plot_surface(X_m, Y_m, NF0, cmap="jet", vmin=0, vmax=cMax,
                         linewidth=0, antialiased=True)
    ax3d_0.set_title("原始近场 (3D)", fontsize=13)

    ax3d_1 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3d_1.plot_surface(X_m, Y_m, NF1, cmap="jet", vmin=0, vmax=cMax,
                         linewidth=0, antialiased=True)
    ax3d_1.set_title("ELM 复原近场 (3D)", fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "nearfield_3d_comparison.png"), dpi=150)
    plt.show()

    print(f"\n完成! 结果图片已保存至 '{cfg.result_dir}/'")


if __name__ == "__main__":
    main()
