"""
数据生成主脚本

生成仿真训练数据集：随机生成光强泽尼克系数作为标签(OutputData)，
通过哈特曼子孔径衍射传播计算各子孔径总光强作为特征(InputData)。

核心流程:
1. 按配置生成子孔径坐标(subcfg)和泽尼克模式矩阵(modes)
2. 对指定阶数 n_amp_modes，生成 n_samples 组随机光强泽尼克系数
3. 重构近场振幅分布 (modes 线性组合，全图 image_size×image_size)
4. 可选叠加随机波前 (flag_wf)
5. 对每个有效子孔径进行衍射传播(DL) → 焦斑光强 → 加噪 → 求和
6. 保存 InputData(n_samples × n_sub) 和 OutputData(n_samples × (n_amp_modes+1))

MATLAB 对应: Main_Data_Generation.m
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import get_config
from src.optics import Optics
from src.hartmann import HartmannSensor
from src.utils import (
    set_seed,
    configure_chinese_font,
    get_data_filename,
    generate_modes,
    generate_mask,
    generate_subcfg,
    save_mat,
    generate_simulation_data,
)


def main():
    cfg = get_config()
    set_seed(cfg.seed)
    configure_chinese_font()

    # ---- 生成配件 ----
    print("生成配件数据...")
    mask = generate_mask(cfg)                         # (image_size, image_size) 圆形光瞳
    modes = generate_modes(cfg)                       # (n_modes_total, image_size, image_size)
    subcfg = generate_subcfg(cfg)                     # (2, n_valid_sub)
    n_sub = subcfg.shape[1]
    print(f"  图像尺寸: {cfg.image_size}×{cfg.image_size}")
    print(f"  子孔径数: {n_sub} ({cfg.n_sub_dim}×{cfg.n_sub_dim} 阵列, 有效 {n_sub})")
    print(f"  modes 形状: {modes.shape}")

    # ---- 初始化光学系统 ----
    optics = Optics(
        wavelength=cfg.wavelength,
        pixel_pitch=cfg.pixel_pitch,
        focal_length=cfg.focal_length,
        n_pixels=cfg.image_size,
        sub_ap_pixels=cfg.sub_ap_pixels,
    )

    hs = HartmannSensor(subcfg, optics)  # 复用其子孔径提取逻辑

    # ---- 显示子孔径布局 ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    for i in range(n_sub):
        y1 = subcfg[0, i] + cfg.image_size / 2
        x1 = subcfg[1, i] + cfg.image_size / 2
        rect = plt.Rectangle(
            (x1, y1), cfg.sub_ap_pixels, cfg.sub_ap_pixels,
            fill=False, edgecolor="r", linewidth=0.5,
        )
        ax.add_patch(rect)
    ax.set_aspect("equal")
    ax.set_title(f"子孔径布局 ({cfg.n_sub_dim}×{cfg.n_sub_dim}, 有效{n_sub})")
    plt.tight_layout()
    plt.show()

    # ---- 数据生成主循环 ----
    os.makedirs(cfg.data_dir, exist_ok=True)
    IS = cfg.image_size  # 简写

    for n_amp in range(
        cfg.n_amp_modes_range[0],
        cfg.n_amp_modes_range[1] + 1,
        cfg.n_amp_modes_range[2],
    ):
        print(f"\n生成光强 {n_amp} 阶泽尼克数据...")

        InputData, OutputData = generate_simulation_data(
            cfg, modes, subcfg, optics, hs,
            n_samples=cfg.n_samples, n_amp=n_amp,
            enhanced=False, seed=cfg.seed, show_progress=True,
        )

        # ---- 保存数据 ----
        suffix = get_data_filename(cfg)
        input_path = os.path.join(cfg.data_dir, f"InputData{suffix}.mat")
        output_path = os.path.join(cfg.data_dir, f"OutputData{suffix}.mat")
        save_mat(input_path, InputData=InputData)
        save_mat(output_path, OutputData=OutputData)
        print(f"  已保存: {input_path}")
        print(f"  已保存: {output_path}")

    print("\n数据生成完成!")


if __name__ == "__main__":
    main()
