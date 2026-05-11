"""
ELM 训练与近场复原主脚本

使用极限学习机(ELM)从哈特曼子孔径光强预测泽尼克系数，
进而重构近场振幅分布。包含隐藏层神经元数优化和结果可视化。

核心流程:
1. 加载训练数据 (InputData / OutputData)
2. 划分训练集(90%)和测试集(10%)
3. 数据归一化 (MinMaxScaler → [-1, 1])
4. 网格搜索最优隐藏层神经元数
5. 使用最优神经元数训练 ELM
6. 测试集评估: 预测泽尼克系数 → 重构近场 → 可视化对比

MATLAB 对应: Main_A_ELM.m
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
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
    circ_mask,
    configure_chinese_font,
    get_data_filename,
    reconstruct_from_zernike,
)


# ===========================================================================
# 参数配置
# ===========================================================================
@dataclass
class Config:
    """ELM 训练参数"""

    # ---- 数据参数 ----
    n_zernike: int = 25                      # 泽尼克阶数 (对应数据文件名)
    flag_noise: bool = True                   # 是否含噪声
    flag_wf: bool = True                      # 是否含波前扰动
    test_ratio: float = 0.1                   # 测试集比例

    # ---- ELM 参数 ----
    activation: str = "softplus"              # 激活函数: sigmoid/relu/softplus/tanh/sin/rbf
    hidden_range: tuple = (100, 1000, 25)     # 隐藏层神经元搜索范围 (起始, 结束, 步长)
    n_hidden: int | None = None               # 指定神经元数 (None=自动搜索最优)

    # ---- 可视化参数 ----
    test_index: int = 10                      # 展示测试集中第几个样本

    # ---- 随机种子 ----
    seed: int = 23

    # ---- 路径 ----
    accessories_dir: str = "accessories"
    data_dir: str = "data"
    result_dir: str = "result"


def main():
    cfg = Config()
    set_seed(cfg.seed)
    configure_chinese_font()
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ---- 1. 加载数据 ----
    print("加载数据...")
    name = get_data_filename(cfg.n_zernike, cfg.flag_noise, cfg.flag_wf)
    InputData = load_mat(os.path.join(cfg.data_dir, f"InputData{name}.mat"))
    OutputData = load_mat(os.path.join(cfg.data_dir, f"OutputData{name}.mat"))
    modes = load_mat(os.path.join(cfg.accessories_dir, "modes250.mat"), "modes")

    nZer = OutputData.shape[0] - 1  # 泽尼克阶数
    print(f"  输入维度: {InputData.shape}, 输出维度: {OutputData.shape}")
    print(f"  泽尼克阶数: {nZer}")

    # ---- 2. 划分数据 ----
    zer_indices = list(range(nZer)) + [OutputData.shape[0] - 1]
    X = InputData                     # (n_sub, n_samples)
    y = OutputData[zer_indices, :]    # (nZer+1, n_samples)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_ratio=cfg.test_ratio, shuffle=False
    )
    n_test = X_test.shape[1]
    print(f"  训练样本: {X_train.shape[1]}, 测试样本: {n_test}")

    # ---- 3. 数据归一化 ----
    X_norm, y_norm, scaler_X, scaler_y = normalize_data(X_train, y_train)
    X_test_norm = apply_normalize(X_test, scaler_X)

    # ---- 4. 网格搜索最优隐藏层神经元数 ----
    if cfg.n_hidden is None:
        print("搜索最优隐藏层神经元数...")
        hidden_list = list(
            range(cfg.hidden_range[0], cfg.hidden_range[1] + 1, cfg.hidden_range[2])
        )
        mse_list = np.zeros(len(hidden_list))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plt.ion()  # 交互模式，允许实时更新
        pbar = tqdm(hidden_list, desc="搜索最优神经元数", unit="个")
        for idx, n_hid in enumerate(pbar):
            elm = ELM(n_hidden=n_hid, activation=cfg.activation)
            elm.fit(X_norm, y_norm)
            pred_norm = elm.predict(X_test_norm)
            pred = reverse_normalize(pred_norm, scaler_y)
            mse_list[idx] = compute_mse(y_test, pred)
            pbar.set_postfix({"MSE": f"{mse_list[idx]:.4e}"})

            # 实时更新曲线
            ax.clear()
            ax.plot(hidden_list[: idx + 1], mse_list[: idx + 1], "b-", linewidth=1.5)
            ax.set_xlabel("隐藏层神经元数", fontsize=13)
            ax.set_ylabel("MSE", fontsize=13)
            ax.set_title(f"激活函数: {cfg.activation}", fontsize=13)
            ax.grid(True, alpha=0.3)
            plt.pause(0.01)

        min_idx = np.argmin(mse_list)
        best_hidden = hidden_list[min_idx]
        best_mse = mse_list[min_idx]

        # 平滑曲线并标注最优点
        ax.clear()
        from scipy.interpolate import make_interp_spline
        x_smooth = np.linspace(hidden_list[0], hidden_list[-1], 300)
        spl = make_interp_spline(hidden_list, mse_list, k=3)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, "b-", linewidth=1.5)
        ax.plot(best_hidden, best_mse, "ro", markersize=8)
        ax.axvline(best_hidden, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(
            best_hidden,
            best_mse * 1.1,
            f"  N={best_hidden}\n  MSE={best_mse:.2e}",
            color="red",
            fontsize=10,
        )
        ax.set_xlabel("隐藏层神经元数", fontsize=13)
        ax.set_ylabel("MSE", fontsize=13)
        ax.set_title(f"最优神经元搜索 ({cfg.activation})", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.result_dir, "neuron_search.png"), dpi=150)
        plt.ioff()  # 恢复阻塞模式
        plt.show()

        print(f"  最优隐藏层神经元数: {best_hidden}, MSE = {best_mse:.4e}")
    else:
        best_hidden = cfg.n_hidden

    # ---- 5. 训练最终模型 ----
    print(f"训练 ELM (N={best_hidden}, activation={cfg.activation})...")
    elm = ELM(n_hidden=best_hidden, activation=cfg.activation)
    elm.fit(X_norm, y_norm)
    pred_norm = elm.predict(X_test_norm)
    pred = reverse_normalize(pred_norm, scaler_y)

    # 评估
    final_mse = compute_mse(y_test, pred)
    print(f"  测试集最终 MSE = {final_mse:.4e}")

    # ---- 6. 可视化: 泽尼克系数对比 ----
    idx = min(cfg.test_index, n_test - 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 折线图
    x_orders = np.arange(1, nZer + 1)
    ax1.plot(x_orders, y_test[:nZer, idx], "o-", linewidth=1, label="真实值")
    ax1.plot(x_orders, pred[:nZer, idx], "s-", linewidth=1, label="预测值")
    ax1.set_xlabel("泽尼克阶数", fontsize=13)
    ax1.set_ylabel("泽尼克系数", fontsize=13)
    ax1.set_title("泽尼克系数对比", fontsize=13)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 柱状图
    bar_width = 0.35
    ax2.bar(x_orders - bar_width / 2, y_test[:nZer, idx], bar_width, label="真实值")
    ax2.bar(x_orders + bar_width / 2, pred[:nZer, idx], bar_width, label="预测值")
    ax2.set_xlabel("泽尼克阶数", fontsize=13)
    ax2.set_ylabel("泽尼克系数", fontsize=13)
    ax2.set_title("泽尼克系数对比 (柱状图)", fontsize=13)
    ax2.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "coefficient_comparison.png"), dpi=150)
    plt.show()

    # ---- 7. 重构近场分布并对比 ----
    A0 = reconstruct_from_zernike(y_test[:, idx], modes, nZer)
    A1 = reconstruct_from_zernike(pred[:, idx], modes, nZer)

    # 加上偏移量确保非负
    A0 = A0 + y_test[-1, idx]
    A1 = A1 + pred[-1, idx]

    # 圆形掩模截断
    mask_240 = circ_mask(240)
    NF0 = A0 * mask_240
    NF1 = A1 * mask_240

    cMax = max(np.max(NF0), np.max(NF1))

    # 二维分布对比
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

    fig.suptitle(f"{cfg.n_zernike}阶泽尼克近场复原 (ELM, {cfg.activation})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "nearfield_2d_comparison.png"), dpi=150)
    plt.show()

    # 三维分布对比
    fig = plt.figure(figsize=(12, 5))
    X_m, Y_m = np.meshgrid(np.arange(240), np.arange(240))

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
