"""
统一系统配置中心

所有主脚本通过 get_config() 获取配置，读取项目根目录的 config.json。
切换配置只需替换 config.json，或直接编辑其中参数。

    cp config_simulation.json config.json   # 切回原仿真参数
    python configs.py                       # 用当前 SystemConfig 默认值覆盖 config.json

两类泽尼克系数（独立配置，不可混淆）:
- n_amp_modes: 光强(振幅)泽尼克阶数 — 重构近场振幅分布，ELM 预测目标
- n_wf_modes:  波前(相位)泽尼克阶数 — 波前传感，Z2S 响应矩阵行数
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass
class SystemConfig:
    """自适应光学仿真系统统一配置"""

    # =========================================================================
    # 光学系统参数
    # =========================================================================
    wavelength: float = 635e-6          # 波长 (mm)，635nm
    focal_length: float = 20.0          # 微透镜焦距 (mm)
    pixel_pitch: float = 12.8e-3        # 像元尺寸 (mm/pixel)，12.8μm

    # =========================================================================
    # 探测器靶面参数
    # =========================================================================
    image_size: int = 400               # 全图尺寸 (像素)
    beam_size: int = 400                # 光束直径 (像素)，≤ image_size，modes 网格尺寸
    sub_ap_pixels: int = 25             # 子孔径边长 (像素)
    n_sub_dim: int = 16                 # 子孔径阵列维度 (n_sub_dim × n_sub_dim)
    sub_valid_ratio: float = 0.6        # 有效子孔径能量阈值 (MATLAB 默认 0.6)

    # =========================================================================
    # 泽尼克模式参数
    # =========================================================================
    n_modes_total: int = 36             # 生成的总模式数 (Noll 1 ~ n_modes_total)
    n_amp_modes: int = 35               # 光强泽尼克阶数 (ELM 预测目标维度)
    n_amp_modes_range: Tuple[int, int, int] = (35, 35, 5)  # data_generation 循环
    n_wf_modes: int = 35                # 波前泽尼克阶数 (Z2S 响应矩阵行数)
    wf_skip_count: int = 3              # 波前跳过前 N 个模式 (0=全用, 1=去piston, 2=去piston+tip, 3=去piston+tip+tilt)
    wf_coeff_std: float = 0.2           # 波前系数标准差 (仅 flag_wf=True 时使用)

    # =========================================================================
    # 数据生成参数
    # =========================================================================
    n_samples: int = 1000               # 样本总数
    flag_noise: bool = True             # 是否添加噪声
    noise_sigma: float = 0.5            # 相机本底噪声 RMS
    flag_wf: bool = False               # 是否叠加随机波前扰动

    # =========================================================================
    # ELM 参数
    # =========================================================================
    activation: str = "softplus"        # 激活函数
    n_hidden: int | None = None          # 隐藏层神经元数 (None=自动搜索)
    hidden_range: Tuple[int, int, int] = (100, 1000, 25)  # 自动搜索时
    test_ratio: float = 0.1             # 测试集比例
    val_ratio: float = 0.1              # 验证集比例

    # =========================================================================
    # 超参数优化专用
    # =========================================================================
    opt_methods: Tuple[str, ...] = ("Relu", "Sigmoid", "Tanh", "Softplus")
    opt_hidden_range: Tuple[int, int, int] = (50, 1000, 10)
    opt_n_runs: int = 5

    # =========================================================================
    # suremax 专用
    # =========================================================================
    suremax_zernike_range: Tuple[int, int, int] = (15, 250, 5)

    # =========================================================================
    # 批量测试专用
    # =========================================================================
    n_batch_samples: int | None = None  # None=使用全部测试集
    batch_test_index: int = 10

    # =========================================================================
    # 路径与随机种子
    # =========================================================================
    accessories_dir: str = "accessories"
    data_dir: str = "data"
    result_dir: str = "result"
    seed: int = 42


    # =========================================================================
    # 派生属性
    # =========================================================================
    @property
    def beam_offset(self) -> int:
        """光束在靶面中的偏移量，(image_size - beam_size) // 2"""
        return (self.image_size - self.beam_size) // 2

    @property
    def sub_ap_size(self) -> float:
        """子孔径物理尺寸 (mm)"""
        return self.sub_ap_pixels * self.pixel_pitch

    @property
    def wf_effective_modes(self) -> int:
        """波前实际使用的模式数 (跳过前 wf_skip_count 个)"""
        return self.n_wf_modes - self.wf_skip_count

    @property
    def wf_start_mode(self) -> int:
        """波前起始模式索引 (0-based, 在 modes 数组中的偏移)"""
        return self.wf_skip_count


def _dict_to_config(d: dict) -> SystemConfig:
    """将字典转为 SystemConfig，处理 tuple 类型还原"""
    for key in list(d.keys()):
        # 还原 tuple 字段 (JSON 存为 list)
        if key in ("n_amp_modes_range", "hidden_range", "opt_methods",
                    "opt_hidden_range", "suremax_zernike_range"):
            d[key] = tuple(d[key])
    return SystemConfig(**d)


def get_config(path: str = "config_exp.json") -> SystemConfig:
    """
    从 JSON 文件加载配置，文件不存在时返回 SystemConfig 默认值。

    用法:
        from configs import get_config
        cfg = get_config()               # 默认读取 ./config.json
        cfg = get_config("my_config.json")  # 指定文件
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _dict_to_config(data)
    return SystemConfig()


def save_config(cfg: SystemConfig, path: str) -> None:
    """将配置保存为 JSON 文件"""
    d = asdict(cfg)
    # tuple → list for JSON
    for key in list(d.keys()):
        if isinstance(d[key], tuple):
            d[key] = list(d[key])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    print(f"配置已保存: {path}")


if __name__ == "__main__":
    # 将当前默认参数保存为 config.json
    save_config(SystemConfig(), "config.json")
