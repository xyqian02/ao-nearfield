"""
光学计算核心模块

包含衍射传播(DL)、泽尼克多项式生成、光瞳函数、标准光束函数等
自适应光学仿真的光学计算基础功能。

MATLAB 对应: DL.m, Zernike.m, nmZern.m, Pupil.m, StdBeamFunc.m
"""

import numpy as np
from scipy.special import factorial


class Optics:
    """光学系统计算核心，统一管理光学参数"""

    def __init__(
        self,
        wavelength: float = 1.064e-3,   # 波长，单位 mm
        pixel_pitch: float = 14e-3,     # 像元尺寸，单位 mm
        focal_length: float = 21.0,     # 微透镜焦距，单位 mm
        n_pixels: int = 256,            # 图像尺寸 (像素)
        sub_ap_pixels: int = 20,        # 子孔径边长 (像素)
    ):
        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch
        self.focal_length = focal_length
        self.n_pixels = n_pixels
        self.sub_ap_pixels = sub_ap_pixels

    @property
    def k(self) -> float:
        """波数 k = 2π/λ"""
        return 2.0 * np.pi / self.wavelength

    @property
    def sub_ap_size(self) -> float:
        """子孔径物理尺寸 (mm)"""
        return self.sub_ap_pixels * self.pixel_pitch

    def dl_propagate(
        self,
        field: np.ndarray,
        distance: float | None = None,
    ) -> np.ndarray:
        """
        角谱法衍射传播计算

        模拟光场通过微透镜后在焦面/传输距离处的复振幅分布。
        算法: 输入场 × 透镜相位因子 → FFT → 频域传输函数 → IFFT → 输出场

        参数:
            field: 输入复振幅场，形状 (N, N)
            distance: 传输距离 (mm)，默认为焦距 f

        返回:
            输出复振幅场，形状 (N, N)

        MATLAB 对应: DL.m
        """
        if distance is None:
            distance = self.focal_length

        grid_num = field.shape[0]
        # 输入面孔径尺寸 = 子孔径像素数 × 像元尺寸
        d = grid_num * self.pixel_pitch
        grid_step = d / grid_num

        # 空间域坐标
        x = np.linspace(-d / 2, d / 2, grid_num)
        X, Y = np.meshgrid(x, x)

        # 频域坐标 (MATLAB: axisFFT = -1/(2*GridStep) + (0:GridNum-1)/D)
        axis_fft = -1.0 / (2.0 * grid_step) + np.arange(grid_num) / d
        FFT_X, FFT_Y = np.meshgrid(axis_fft, axis_fft)

        # 透镜相位因子: exp(i * k / (2f) * (x² + y²))
        field = field * np.exp(1j * self.k / (2.0 * self.focal_length) * (X**2 + Y**2))

        # 频域传输函数: exp(-i * 2π * z * sqrt(1/λ² - (f_x² + f_y²)))
        # sqrt_arg = 1/lambda² - (fft_x² + fft_y²)
        sqrt_arg = 1.0 / self.wavelength**2 - (FFT_X**2 + FFT_Y**2)
        # 处理倏逝波分量 (sqrt_arg < 0)，设为0
        sqrt_arg = np.maximum(sqrt_arg, 0.0)
        mat = np.exp(-1j * 2.0 * np.pi * distance * np.sqrt(sqrt_arg))

        # FFT → 频域相乘 → IFFT
        temp = mat * np.fft.fftshift(np.fft.fft2(field))
        result = np.fft.ifft2(np.fft.ifftshift(temp))

        return result

    @staticmethod
    def _nm_zern(mode: int) -> tuple[int, int]:
        """
        将泽尼克模式号转换为径向阶数 n 和角向频率 m (Noll 索引方案)

        参数:
            mode: 泽尼克模式号 (从1开始，1=piston)

        返回:
            (n, m): 径向阶数和角向频率

        MATLAB 对应: nmZern.m
        """
        csum = np.cumsum(np.arange(1, mode + 1))
        n = int(np.sum(csum < mode))

        if n == 0:
            m = 0
        elif n % 2 == 0:
            m = int(np.fix((mode - csum[n]) / 2)) * 2
        else:
            m = int(np.round((mode - csum[n]) / 2)) * 2 - 1

        return n, m

    @staticmethod
    def zernike(mode: int, Na: int, pupil: np.ndarray | None = None) -> np.ndarray:
        """
        生成单阶泽尼克多项式

        在单位圆上计算指定模式的泽尼克多项式。
        使用 Noll 索引方案：mode=1 为 piston, mode=2,3 为 tip/tilt, mode=4 为 defocus。

        参数:
            mode: 泽尼克模式号 (从1开始)
            Na: 输出矩阵边长 (Na × Na)
            pupil: 光瞳掩模，若为 None 则默认使用圆形光瞳

        返回:
            Na × Na 的泽尼克多项式矩阵

        MATLAB 对应: Zernike.m
        """
        n, m = Optics._nm_zern(mode)

        # 单位圆坐标网格
        x = np.linspace(-1, 1, Na)
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2)
        th = np.arctan2(Y, X)

        # 径向多项式 R_n^m(r)
        R = np.zeros_like(r)
        s = 0
        while s <= (n - abs(m)) // 2:
            a = ((-1) ** s) * factorial(n - s)
            b = (
                factorial(s)
                * factorial((n + abs(m)) // 2 - s)
                * factorial((n - abs(m)) // 2 - s)
            )
            coeff = a / b
            R = R + coeff * r ** (n - 2 * s)
            s = s + 1

        # 角向部分 + 归一化因子
        if m == 0:
            z = np.sqrt(n + 1) * R
        elif mode % 2 == 0:  # 偶模式 → cos(mθ)
            z = np.sqrt(2 * (n + 1)) * R * np.cos(abs(m) * th)
        else:  # 奇模式 → sin(mθ)
            z = np.sqrt(2 * (n + 1)) * R * np.sin(abs(m) * th)

        # 光瞳截断
        if pupil is None:
            z = z * Optics.pupil_circle(Na)
        else:
            z = z * pupil

        return z

    @staticmethod
    def zernike_modes(n_modes: int, Na: int) -> np.ndarray:
        """
        批量生成前 n_modes 阶泽尼克模式

        参数:
            n_modes: 泽尼克模式阶数 (≥1)
            Na: 输出矩阵边长

        返回:
            三维数组，形状 (n_modes, Na, Na)，
            其中 modes[k-1] 为第 k 阶泽尼克多项式 (k=1,2,...,n_modes)

        注: 此函数使用圆形光瞳 Pupil_circle 进行截断。
        如需自定义光瞳，请使用 zernike() 逐阶生成。
        """
        pupil = Optics.pupil_circle(Na)
        modes = np.zeros((n_modes, Na, Na))
        for i in range(n_modes):
            modes[i] = Optics.zernike(i + 1, Na, pupil)
        return modes

    @staticmethod
    def pupil_circle(N: int) -> np.ndarray:
        """
        创建圆形光瞳函数

        在 N×N 区域中心生成单位圆，圆内为1，圆外为0。

        参数:
            N: 输出矩阵边长

        返回:
            N × N 的二值光瞳矩阵

        MATLAB 对应: Pupil.m
        """
        # 向量化实现，比逐像素循环快 50-100 倍
        Y, X = np.ogrid[:N, :N]
        center = N / 2.0
        radius = N / 2.0
        distance = np.hypot(X + 0.5 - center, Y + 0.5 - center)
        return (distance <= radius).astype(np.float64)

    @staticmethod
    def std_beam(
        N: int,
        rec: float | tuple,
        cir: float | tuple,
        m: float | tuple = 1e99,
    ) -> np.ndarray:
        """
        生成标准近场光束分布（矩形域内含圆形/环形遮挡）

        创建一个理想的近场强度分布，可用于模拟平面波
        或带有中心遮拦的光束。

        参数:
            N: 坐标网格维度
            rec: 矩形区域尺寸 (Lx, Ly) 或单值 L
            cir: 圆形区域直径 (Dx, Dy) 或单值 D
            m: 中心遮拦比例 (Mx, My) 或单值 M (默认 1e99 表示无遮拦)

        返回:
            N × N 的近场分布矩阵

        MATLAB 对应: StdBeamFunc.m
        """
        # 统一处理标量/向量参数
        if np.isscalar(rec):
            Lx = Ly = abs(rec)
        else:
            Lx, Ly = abs(rec[0]), abs(rec[1])

        if np.isscalar(cir):
            Dx = Dy = abs(cir)
        else:
            Dx, Dy = abs(cir[0]), abs(cir[1])

        if np.isscalar(m):
            Mx = My = abs(m)
        else:
            Mx, My = abs(m[0]), abs(m[1])

        sizeX = min(Lx, Dx)
        sizeY = min(Ly, Dy)
        Dim = max(sizeX, sizeY)

        X, Y = np.meshgrid(
            np.linspace(-Dim / 2, Dim / 2, N),
            np.linspace(-Dim / 2, Dim / 2, N),
        )

        # 主圆形区域 + 矩形截断
        near_field = np.zeros((N, N))
        condition_outer = (
            (np.hypot(2 * X / Dx, 2 * Y / Dy) <= 1)
            & (np.abs(X) <= sizeX / 2)
            & (np.abs(Y) <= sizeY / 2)
        )
        near_field[condition_outer] = 1.0

        # 中心遮拦
        condition_inner = (
            (np.hypot(2 * X / (Dx / Mx), 2 * Y / (Dy / My)) <= 1)
            & (np.abs(X) <= sizeX / 2 / Mx)
            & (np.abs(Y) <= sizeY / 2 / My)
        )
        near_field[condition_inner] = 0.0

        return near_field
