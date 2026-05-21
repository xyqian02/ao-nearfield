"""
泽尼克阶数影响分析主脚本

分析不同光强泽尼克阶数对 ELM 预测 MSE 的影响。
对于每个阶数，加载对应数据，在验证集上搜索最优神经元数，
记录验证集最优 MSE，绘制 MSE 随泽尼克阶数的变化曲线。

注意: 此脚本需要对应阶数的数据文件已存在 (由 main_data_generation.py 生成)。

MATLAB 对应: Main_SureMax.m
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
from src.evaluation import compute_mse
from src.utils import (
    load_mat,
    set_seed,
    configure_chinese_font,
)


def main():
    cfg = get_config()
    set_seed(cfg.seed)
    configure_chinese_font()
    os.makedirs(cfg.result_dir, exist_ok=True)

    zernike_orders = list(
        range(cfg.suremax_zernike_range[0],
              cfg.suremax_zernike_range[1] + 1,
              cfg.suremax_zernike_range[2])
    )
    n_orders = len(zernike_orders)
    MSE_results = np.zeros(n_orders)

    hidden_list = list(
        range(cfg.opt_hidden_range[0], cfg.opt_hidden_range[1] + 1, cfg.opt_hidden_range[2])
    )

    print(f"分析光强泽尼克阶数 {zernike_orders[0]}~{zernike_orders[-1]} 对 MSE 的影响...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.ion()

    pbar = tqdm(zernike_orders, desc="泽尼克阶数分析", unit="阶")
    for idx, NN in enumerate(pbar):
        pbar.set_postfix({"当前阶数": f"{NN}"})

        # 构建该阶数的文件名
        suffix = f"_s{cfg.image_size}_a{NN}"
        if cfg.flag_noise:
            suffix += "_noise"
        if cfg.flag_wf:
            suffix += "_wf"

        input_path = os.path.join(cfg.data_dir, f"InputData{suffix}.mat")
        output_path = os.path.join(cfg.data_dir, f"OutputData{suffix}.mat")

        if not os.path.exists(input_path):
            MSE_results[idx] = np.nan
            continue

        InputData = load_mat(input_path)
        OutputData = load_mat(output_path)
        n_amp = OutputData.shape[1] - 1

        X = InputData
        y = OutputData[:, :n_amp]  # 不含 offset 列

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        scaler_X = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
        scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(y_train)
        X_train_norm = scaler_X.transform(X_train)
        y_train_norm = scaler_y.transform(y_train)
        X_val_norm = scaler_X.transform(X_val)

        best_mse = np.inf
        for n_hid in hidden_list:
            elm = ELM(n_hidden=n_hid, activation=cfg.activation, random_state=cfg.seed)
            elm.fit(X_train_norm, y_train_norm)
            y_val_pred_norm = elm.predict(X_val_norm)
            y_val_pred = scaler_y.inverse_transform(y_val_pred_norm)
            mse_val = compute_mse(y_val, y_val_pred)
            if mse_val < best_mse:
                best_mse = mse_val

        MSE_results[idx] = best_mse

        valid_idx = ~np.isnan(MSE_results)
        ax.clear()
        ax.plot(
            np.array(zernike_orders)[valid_idx],
            MSE_results[valid_idx],
            "b-", linewidth=1.5,
        )
        ax.set_xlabel("光强泽尼克阶数", fontsize=13)
        ax.set_ylabel("MSE (验证集最优)", fontsize=13)
        ax.set_title("MSE 随光强泽尼克阶数的变化", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.pause(0.01)

    valid_idx = ~np.isnan(MSE_results)
    ax.clear()
    ax.plot(
        np.array(zernike_orders)[valid_idx],
        MSE_results[valid_idx],
        "b-", linewidth=1.5,
    )
    ax.set_xlabel("光强泽尼克阶数", fontsize=13)
    ax.set_ylabel("MSE (验证集最优)", fontsize=13)
    ax.set_title("MSE 随光强泽尼克阶数的变化", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(cfg.result_dir, "mse_vs_zernike_order.png"), dpi=150)
    plt.show()

    print(f"\n完成! 结果图片已保存至 '{cfg.result_dir}/'")


if __name__ == "__main__":
    main()
