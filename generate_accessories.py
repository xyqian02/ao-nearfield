"""
配件预生成工具

按当前配置预生成 subcfg 和 modes 并缓存为 .mat 文件，
后续主脚本运行时自动从缓存加载，避免重复计算。

用法:
    python generate_accessories.py                # 使用默认配置
    python generate_accessories.py --show         # 生成并显示子孔径布局
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import get_config
from src.utils import (
    set_seed,
    configure_chinese_font,
    generate_modes,
    generate_mask,
    generate_subcfg,
    save_accessories,
)


def main():
    parser = argparse.ArgumentParser(description="配件预生成工具")
    parser.add_argument("--show", action="store_true", help="显示子孔径布局")
    args = parser.parse_args()

    cfg = get_config()
    set_seed(cfg.seed)
    configure_chinese_font()

    print(f"配置: image_size={cfg.image_size}, n_sub_dim={cfg.n_sub_dim}, "
          f"sub_ap_pixels={cfg.sub_ap_pixels}")
    print(f"  n_modes_total={cfg.n_modes_total}, "
          f"n_amp_modes={cfg.n_amp_modes}, n_wf_modes={cfg.n_wf_modes}")

    # 生成配件
    mask = generate_mask(cfg)
    modes = generate_modes(cfg)
    subcfg = generate_subcfg(cfg)

    print(f"\n生成结果:")
    print(f"  mask:  {mask.shape}")
    print(f"  modes: {modes.shape}  (H={modes.shape[0]}, W={modes.shape[1]}, N={modes.shape[2]})")
    print(f"  subcfg: {subcfg.shape}  ({subcfg.shape[1]} 个有效子孔径)")

    # 保存为 .mat 文件
    save_accessories(cfg, modes=modes, subcfg=subcfg)

    # 显示子孔径布局
    if args.show:
        n_sub = subcfg.shape[1]
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
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
        ax.set_title(
            f"子孔径布局 {cfg.n_sub_dim}×{cfg.n_sub_dim}  "
            f"(有效 {n_sub}/{cfg.n_sub_dim * cfg.n_sub_dim})"
        )
        plt.tight_layout()
        plt.show()

        # 显示前几个泽尼克模式
        n_show = min(9, cfg.n_modes_total)
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for k in range(n_show):
            ax = axes[k // 3, k % 3]
            im = ax.imshow(modes[:, :, k], cmap="jet")
            ax.set_title(f"Noll {k + 1}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle("泽尼克模式示例", fontsize=14)
        plt.tight_layout()
        plt.show()

    print("\n完成! 配件已生成，可直接运行各 main_*.py 脚本。")


if __name__ == "__main__":
    main()
