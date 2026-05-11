"""
实验数据处理主脚本

从实际采集的近场图像数据生成哈特曼光斑阵列。
支持两种模式:
- 动态子孔径: 根据每帧近场分布动态判断有效子孔径 (对应 pro1.m)
- 固定网格: 紧密排列的固定子孔径网格 (对应 pro2.m)

核心流程:
1. 加载近场图像序列 (.mat 格式)
2. 对每一帧:
   a. 归一化近场强度
   b. 动态/固定判定有效子孔径
   c. 衍射传播 → 焦斑强度 → 加噪
   d. 合成哈特曼光斑阵列
3. 可视化结果

MATLAB 对应: pro1.m, pro2.m
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optics import Optics
from src.hartmann import create_subcfg, create_sub_valid
from src.utils import load_mat, set_seed, configure_chinese_font


# ===========================================================================
# 参数配置
# ===========================================================================
@dataclass
class Config:
    """实验数据处理配置"""

    # ---- 模式选择 ----
    mode: str = "dynamic"            # "dynamic" = 动态子孔径, "grid" = 固定网格

    # ---- 光学系统参数 ----
    wavelength: float = 1.064e-3
    focal_length: float = 21.0       # 微透镜焦距 (mm)
    pixel_pitch: float = 14e-3
    sub_ap_pixels: int = 20
    image_size: int = 1024           # 实验图像通常 1024×1024

    # ---- 子孔径参数 ----
    sub_grid_dim: int = 50           # 子孔径阵列维度 (50×50)
    energy_ratio: float = 0.1        # 有效子孔径能量阈值

    # ---- 噪声 ----
    noise_sigma: float = 0.5
    flag_noise: bool = True

    # ---- 实验数据路径 ----
    experiment_data_path: str = ""   # 需用户指定 .mat 文件路径

    # ---- 结果 ----
    result_dir: str = "result"

    # ---- 随机种子 ----
    seed: int = 42


def main():
    cfg = Config()
    set_seed(cfg.seed)
    configure_chinese_font()
    os.makedirs(cfg.result_dir, exist_ok=True)

    # ---- 1. 加载实验数据 ----
    if not cfg.experiment_data_path:
        print("=" * 60)
        print("  实验数据处理脚本")
        print("  请指定 experiment_data_path 参数为你的 .mat 文件路径")
        print("  数据中需包含近场图像变量 (默认变量名: imgN)")
        print("=" * 60)
        print("\n示例用法:")
        print("  cfg.experiment_data_path = 'your_nearfield_data.mat'")
        print("  然后运行: python main_experiment.py")
        return

    print(f"加载实验数据: {cfg.experiment_data_path}")
    data = load_mat(cfg.experiment_data_path)
    NF_data = data  # 假设直接是图像数据，或使用 key 参数指定变量名

    if NF_data.ndim == 3:
        n_row, n_col, n_frames = NF_data.shape
    else:
        # 单帧数据，扩展为三维
        NF_data = NF_data[:, :, np.newaxis]
        n_row, n_col, n_frames = NF_data.shape

    print(f"  数据尺寸: {n_row} × {n_col} × {n_frames}")

    # 自动匹配图像尺寸
    if n_row != cfg.image_size:
        print(f"  自动调整 image_size: {cfg.image_size} → {n_row}")
        cfg.image_size = n_row

    # ---- 2. 初始化光学系统 ----
    optics = Optics(
        wavelength=cfg.wavelength,
        pixel_pitch=cfg.pixel_pitch,
        focal_length=cfg.focal_length,
        n_pixels=cfg.image_size,
        sub_ap_pixels=cfg.sub_ap_pixels,
    )

    # ---- 3. 逐帧处理 ----
    Hartmann_Images = np.zeros((n_row, n_col, n_frames))
    SubIntensity = []  # 每帧的子孔径强度列表

    print(f"处理 {n_frames} 帧...")
    for k in tqdm(range(n_frames), desc="逐帧处理", unit="帧"):
        # 读取当前帧并归一化
        I_frame = NF_data[:, :, k].astype(np.float64)
        I_max = np.max(I_frame)
        if I_max > 0:
            I_norm = I_frame / I_max
        else:
            I_norm = I_frame

        if cfg.mode == "dynamic":
            # 动态子孔径: 根据近场能量判断有效性
            SubValid = create_sub_valid(
                cfg.sub_grid_dim, I_norm, cfg.sub_ap_pixels, cfg.energy_ratio
            )
            subcfg_frame = create_subcfg(SubValid, cfg.image_size, cfg.sub_ap_pixels)
        else:
            # 固定网格: 紧密排列
            n_sub_y = int(np.floor(n_row / cfg.sub_ap_pixels))
            n_sub_x = int(np.floor(n_col / cfg.sub_ap_pixels))
            y_starts = np.arange(0, n_row, cfg.sub_ap_pixels)[:n_sub_y]
            x_starts = np.arange(0, n_col, cfg.sub_ap_pixels)[:n_sub_x]
            Y_grid, X_grid = np.meshgrid(y_starts, x_starts)
            y_all = Y_grid.flatten()
            x_all = X_grid.flatten()
            n_sub_frame = len(y_all)
            subcfg_frame = np.zeros((2, n_sub_frame))
            # 转为以图像中心为原点的坐标
            subcfg_frame[0, :] = y_all - cfg.image_size / 2  # y 坐标
            subcfg_frame[1, :] = x_all - cfg.image_size / 2  # x 坐标

        n_sub_frame = subcfg_frame.shape[1]

        # 振幅计算 (假设相位为平面波)
        Ampl = np.sqrt(np.maximum(I_frame, 0))
        InputField = Ampl

        Hartmann_Frame = np.zeros((n_row, n_col))
        matI = np.zeros(n_sub_frame)

        for iSub in range(n_sub_frame):
            # 提取子孔径区域
            y1_mat = int(np.round(subcfg_frame[0, iSub] + cfg.image_size / 2))
            x1_mat = int(np.round(subcfg_frame[1, iSub] + cfg.image_size / 2))
            y1_py = max(0, min(y1_mat - 1, cfg.image_size - cfg.sub_ap_pixels))
            x1_py = max(0, min(x1_mat - 1, cfg.image_size - cfg.sub_ap_pixels))

            sub_field = InputField[
                y1_py : y1_py + cfg.sub_ap_pixels,
                x1_py : x1_py + cfg.sub_ap_pixels,
            ]

            # 衍射计算
            result = optics.dl_propagate(sub_field)
            spot = np.abs(result) ** 2

            if cfg.flag_noise:
                spot = spot + 1.0 + cfg.noise_sigma * np.random.randn(
                    cfg.sub_ap_pixels, cfg.sub_ap_pixels
                )
                spot = np.maximum(spot, 0.0)

            Hartmann_Frame[
                y1_py : y1_py + cfg.sub_ap_pixels,
                x1_py : x1_py + cfg.sub_ap_pixels,
            ] = spot
            matI[iSub] = np.sum(spot)

        Hartmann_Images[:, :, k] = Hartmann_Frame
        SubIntensity.append(matI)

    # ---- 4. 可视化 ----
    mid_frame = n_frames // 2

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    im0 = axes[0, 0].imshow(
        Hartmann_Images[:, :, 0], cmap="jet",
    )
    axes[0, 0].set_title(f"第一帧哈特曼光斑 (子孔径: {len(SubIntensity[0])})", fontsize=12)
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(
        Hartmann_Images[:, :, mid_frame], cmap="jet",
    )
    axes[0, 1].set_title(
        f"第{mid_frame}帧哈特曼光斑 (子孔径: {len(SubIntensity[mid_frame])})", fontsize=12
    )
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[1, 0].imshow(
        Hartmann_Images[:, :, -1], cmap="jet",
    )
    axes[1, 0].set_title(
        f"最后一帧哈特曼光斑 (子孔径: {len(SubIntensity[-1])})", fontsize=12
    )
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)

    # 子孔径数量变化
    n_sub_array = np.array([len(s) for s in SubIntensity])
    axes[1, 1].plot(n_sub_array, "b-", linewidth=1.5)
    axes[1, 1].set_xlabel("帧序号", fontsize=12)
    axes[1, 1].set_ylabel("有效子孔径数量", fontsize=12)
    axes[1, 1].set_title("有效子孔径数量变化", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "hartmann_experiment.png"), dpi=150)
    plt.show()

    print(f"\n处理完成! 共处理 {n_frames} 帧")
    print(f"结果图片已保存至 '{cfg.result_dir}/hartmann_experiment.png'")


if __name__ == "__main__":
    main()
