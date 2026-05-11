"""
泽尼克阶数影响分析主脚本

分析不同泽尼克阶数 (15~250) 对 ELM 预测 MSE 的影响。
对于每个阶数，加载对应数据，训练 ELM，搜索最优神经元数，
记录最小 MSE，绘制 MSE 随泽尼克阶数的变化曲线。

MATLAB 对应: Main_SureMax.m
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
    configure_chinese_font,
    get_data_filename,
)


# ===========================================================================
# 参数配置
# ===========================================================================
@dataclass
class Config:
    """泽尼克阶数影响分析配置"""

    # ---- 搜索范围 ----
    zernike_range: tuple = (15, 250, 5)       # (起始阶数, 结束阶数, 步长)

    # ---- ELM 参数 ----
    activation: str = "softplus"
    hidden_range: tuple = (100, 1500, 50)     # 每个阶数的搜索范围

    # ---- 数据标记 ----
    flag_noise: bool = True
    flag_wf: bool = True

    # ---- 随机种子 ----
    seed: int = 42

    # ---- 路径 ----
    accessories_dir: str = "accessories"
    data_dir: str = "data"
    result_dir: str = "result"


def main():
    cfg = Config()
    set_seed(cfg.seed)
    configure_chinese_font()
    os.makedirs(cfg.result_dir, exist_ok=True)

    zernike_orders = list(
        range(cfg.zernike_range[0], cfg.zernike_range[1] + 1, cfg.zernike_range[2])
    )
    n_orders = len(zernike_orders)
    MSE_results = np.zeros(n_orders)

    hidden_list = list(
        range(cfg.hidden_range[0], cfg.hidden_range[1] + 1, cfg.hidden_range[2])
    )

    print(f"分析泽尼克阶数 {zernike_orders[0]}~{zernike_orders[-1]} 对 MSE 的影响...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.ion()  # 交互模式，实时更新曲线

    pbar = tqdm(zernike_orders, desc="泽尼克阶数分析", unit="阶")
    for idx, NN in enumerate(pbar):
        pbar.set_postfix({"当前阶数": f"{NN}"})
        name = get_data_filename(NN, cfg.flag_noise, cfg.flag_wf)

        input_path = os.path.join(cfg.data_dir, f"InputData{name}.mat")
        output_path = os.path.join(cfg.data_dir, f"OutputData{name}.mat")

        if not os.path.exists(input_path):
            MSE_results[idx] = np.nan
            continue

        # 加载数据
        InputData = load_mat(input_path)
        OutputData = load_mat(output_path)
        nZer = OutputData.shape[0] - 1

        zer_indices = list(range(nZer))
        X = InputData
        y = OutputData[zer_indices, :]

        X_train, X_test, y_train, y_test = split_data(
            X, y, test_ratio=0.1, shuffle=False
        )
        X_norm, y_norm, scaler_X, scaler_y = normalize_data(X_train, y_train)
        X_test_norm = apply_normalize(X_test, scaler_X)

        # 搜索最优神经元数
        best_mse = np.inf
        for n_hid in hidden_list:
            elm = ELM(n_hidden=n_hid, activation=cfg.activation)
            elm.fit(X_norm, y_norm)
            pred_norm = elm.predict(X_test_norm)
            pred = reverse_normalize(pred_norm, scaler_y)
            mse_val = compute_mse(y_test, pred)
            if mse_val < best_mse:
                best_mse = mse_val

        MSE_results[idx] = best_mse

        # 实时绘图
        valid_idx = ~np.isnan(MSE_results)
        ax.clear()
        ax.plot(
            np.array(zernike_orders)[valid_idx],
            MSE_results[valid_idx],
            "b-", linewidth=1.5,
        )
        ax.set_xlabel("泽尼克阶数", fontsize=13)
        ax.set_ylabel("MSE", fontsize=13)
        ax.set_title("MSE 随泽尼克阶数的变化", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.pause(0.01)

    # 最终图
    valid_idx = ~np.isnan(MSE_results)
    ax.clear()
    ax.plot(
        np.array(zernike_orders)[valid_idx],
        MSE_results[valid_idx],
        "b-", linewidth=1.5,
    )
    ax.set_xlabel("泽尼克阶数", fontsize=13)
    ax.set_ylabel("MSE", fontsize=13)
    ax.set_title("MSE 随泽尼克阶数的变化", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.ioff()  # 恢复阻塞模式
    plt.savefig(os.path.join(cfg.result_dir, "mse_vs_zernike_order.png"), dpi=150)
    plt.show()

    print(f"\n完成! 结果图片已保存至 '{cfg.result_dir}/'")


if __name__ == "__main__":
    main()
