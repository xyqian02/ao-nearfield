"""
数据生成主脚本

生成仿真训练数据集：随机生成泽尼克系数作为标签(OutputData)，
通过哈特曼子孔径衍射计算得到各子孔径总光强作为特征(InputData)。

核心流程:
1. 加载子孔径配置(Subcfg)和泽尼克模式矩阵(modes)
2. 对指定阶数 nZer，生成 nSignal 组随机泽尼克系数
3. 重构近场振幅分布(泽尼克模式线性组合)
4. 可选叠加随机波前(相位扰动)
5. 对每个子孔径进行衍射传播(DL) → 焦斑光强 → 加噪 → 求和
6. 保存 InputData(n_sub × nSignal) 和 OutputData((nZer+1) × nSignal)

MATLAB 对应: Main_Data_Generation.m
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.optics import Optics
from src.utils import load_mat, save_mat, set_seed, configure_chinese_font


# ===========================================================================
# 参数配置
# ===========================================================================
@dataclass
class Config:
    """仿真参数配置，所有可调参数集中在此，方便替换"""

    # ---- 光学系统参数 ----
    wavelength: float = 1.064e-3          # 波长 (mm)
    focal_length: float = 21.0            # 微透镜焦距 (mm)
    pixel_pitch: float = 14e-3            # 像元尺寸 (mm/pixel)
    sub_ap_pixels: int = 20               # 子孔径边长 (像素)
    image_size: int = 256                 # 图像尺寸 (像素)

    # ---- 数据生成参数 ----
    n_zernike: int = 25                   # 泽尼克模式阶数 (可循环多阶)
    n_zernike_range: tuple = (25, 25, 5)  # (起始, 结束, 步长)
    n_samples: int = 10000                # 样本总数

    # ---- 噪声与扰动 ----
    noise_sigma: float = 0.5              # 相机本底噪声 RMS
    flag_noise: bool = True               # 是否添加噪声
    flag_wf: bool = True                  # 是否添加波前扰动
    wf_coeff_std: float = 0.2             # 波前系数标准差
    n_wf_modes: int = 15                  # 波前扰动阶数

    # ---- 随机种子 ----
    seed: int = 42

    # ---- 路径 ----
    accessories_dir: str = "accessories"
    data_dir: str = "data"


def main():
    cfg = Config()
    set_seed(cfg.seed)
    configure_chinese_font()  # 初始化中文字体支持

    # ---- 加载辅助数据 ----
    print("加载辅助数据...")
    subcfg = load_mat(os.path.join(cfg.accessories_dir, "Subcfg.mat"), "Subcfg")
    modes = load_mat(
        os.path.join(cfg.accessories_dir, "modes250.mat"), "modes"
    )
    # modes 形状 (240, 240, 250)，modes[:,:,k] 为第 k+1 阶泽尼克模式
    n_sub = subcfg.shape[1]
    print(f"  子孔径数量: {n_sub}")
    print(f"  泽尼克模式矩阵尺寸: {modes.shape}")

    # ---- 初始化光学系统 ----
    optics = Optics(
        wavelength=cfg.wavelength,
        pixel_pitch=cfg.pixel_pitch,
        focal_length=cfg.focal_length,
        n_pixels=cfg.image_size,
        sub_ap_pixels=cfg.sub_ap_pixels,
    )

    # ---- 显示子孔径布局 ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.zeros((cfg.image_size, cfg.image_size)), cmap="jet", vmin=0, vmax=1)
    for i in range(n_sub):
        y1 = subcfg[0, i] + cfg.image_size / 2
        x1 = subcfg[1, i] + cfg.image_size / 2
        rect = plt.Rectangle(
            (x1, y1), cfg.sub_ap_pixels, cfg.sub_ap_pixels,
            fill=False, edgecolor="r", linewidth=0.5,
        )
        ax.add_patch(rect)
    ax.set_aspect("equal")
    ax.set_title("子孔径布局 (红色矩形)")
    plt.tight_layout()
    plt.show()

    # ---- 生成标准圆域掩模 ----
    mask = np.zeros((cfg.image_size, cfg.image_size))
    temp = Optics.std_beam(240, 100, 100, 1e99)
    mask[8:248, 8:248] = temp  # MATLAB: Mask(9:248,9:248) → Python 0-based

    # ---- 数据生成主循环 ----
    os.makedirs(cfg.data_dir, exist_ok=True)

    for nZer in range(
        cfg.n_zernike_range[0],
        cfg.n_zernike_range[1] + 1,
        cfg.n_zernike_range[2],
    ):
        print(f"\n生成 {nZer} 阶泽尼克数据...")

        # 预分配数组: OutputData = (nZer+1) × n_samples, InputData = n_sub × n_samples
        OutputData = np.zeros((nZer + 1, cfg.n_samples))
        InputData = np.zeros((n_sub, cfg.n_samples))

        # 随机生成泽尼克系数（高斯分布 N(0,1)）
        temp_coeffs = np.random.randn(nZer, cfg.n_samples)
        # 可选的指数衰减: for i in range(nZer): ratio=10*exp(-i/30); temp_coeffs[i]*=ratio
        OutputData[:nZer, :] = temp_coeffs

        for i in tqdm(range(cfg.n_samples), desc=f"  {nZer}阶泽尼克数据", unit="样本"):

            # ---- 重构近场振幅分布 ----
            tempA = np.zeros(240)
            for j in range(nZer):
                # modes[:,:,j] 是第 j+1 阶泽尼克模式 (240×240)
                tempA = tempA + OutputData[j, i] * modes[:, :, j]

            # 保存最小值用于后续偏移
            min_val = np.min(tempA)
            OutputData[nZer, i] = -min_val

            # 振幅分布 (256×256)，非负
            Ampl = np.zeros((cfg.image_size, cfg.image_size))
            Ampl[8:248, 8:248] = tempA - min_val

            # ---- 可选: 叠加随机波前 ----
            wf = np.zeros((cfg.image_size, cfg.image_size))
            if cfg.flag_wf:
                coe = cfg.wf_coeff_std * np.random.randn(cfg.n_wf_modes)
                tempW = np.zeros(240)
                for jj in range(cfg.n_wf_modes):
                    tempW = tempW + coe[jj] * modes[:, :, jj]
                wf[8:248, 8:248] = tempW

            # 入射复振幅场
            InputField = Ampl * np.exp(-1j * wf)

            # ---- 对每个子孔径进行衍射计算 ----
            for iSub in range(n_sub):
                # 计算子孔径区域在图像中的位置 (MATLAB 1-based → Python 0-based)
                y1_mat = int(np.round(subcfg[0, iSub] + cfg.image_size / 2))
                x1_mat = int(np.round(subcfg[1, iSub] + cfg.image_size / 2))
                y1_py = max(0, min(y1_mat - 1, cfg.image_size - cfg.sub_ap_pixels))
                x1_py = max(0, min(x1_mat - 1, cfg.image_size - cfg.sub_ap_pixels))

                sub_field = InputField[
                    y1_py : y1_py + cfg.sub_ap_pixels,
                    x1_py : x1_py + cfg.sub_ap_pixels,
                ]

                # 衍射传播
                result = optics.dl_propagate(sub_field)
                spot = np.abs(result) ** 2

                # 添加噪声
                if cfg.flag_noise:
                    spot = spot + 1.0 + cfg.noise_sigma * np.random.randn(
                        cfg.sub_ap_pixels, cfg.sub_ap_pixels
                    )

                # 记录该子孔径总光强
                InputData[iSub, i] = np.sum(spot)

        # ---- 保存数据 ----
        if not cfg.flag_noise and not cfg.flag_wf:
            suffix = f"_{nZer}"
        elif cfg.flag_noise and not cfg.flag_wf:
            suffix = f"_{nZer}_noise"
        elif cfg.flag_noise and cfg.flag_wf:
            suffix = f"_{nZer}_noise_wf"
        else:
            suffix = f"_{nZer}_wf"

        input_path = os.path.join(cfg.data_dir, f"InputData{suffix}.mat")
        output_path = os.path.join(cfg.data_dir, f"OutputData{suffix}.mat")
        save_mat(input_path, InputData=InputData)
        save_mat(output_path, OutputData=OutputData)
        print(f"  已保存: {input_path}")
        print(f"  已保存: {output_path}")

    print("\n数据生成完成!")


if __name__ == "__main__":
    main()
