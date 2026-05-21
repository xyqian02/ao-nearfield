"""
自适应光学(AO)近场分布复原仿真系统

基于哈特曼-夏克波前传感器和极限学习机(ELM)的
近场振幅分布与波前复原仿真平台。
"""

from .optics import Optics
from .elm import ELM
from .hartmann import HartmannSensor, create_subcfg, create_sub_valid
from .evaluation import compute_mse, compute_rmse, compute_r2, compute_rms_wavefront, compute_pv_wavefront
from .utils import (
    load_mat,
    save_mat,
    set_seed,
    configure_chinese_font,
    get_data_filename,
    reconstruct_from_zernike,
    create_embedded_mask,
    circ_mask,
    generate_modes,
    generate_mask,
    generate_subcfg,
    load_accessories,
    save_accessories,
)

__all__ = [
    "Optics",
    "ELM",
    "HartmannSensor",
    "create_subcfg",
    "create_sub_valid",
    "load_mat",
    "save_mat",
    "compute_mse",
    "compute_rmse",
    "compute_r2",
    "compute_rms_wavefront",
    "compute_pv_wavefront",
    "set_seed",
    "configure_chinese_font",
    "get_data_filename",
    "reconstruct_from_zernike",
    "create_embedded_mask",
    "circ_mask",
    "generate_modes",
    "generate_mask",
    "generate_subcfg",
    "load_accessories",
    "save_accessories",
]
