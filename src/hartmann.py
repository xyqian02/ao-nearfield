"""
哈特曼-夏克波前传感器模块

实现哈特曼传感器的核心功能:
- 子孔径配置（坐标计算、有效性判定）
- 平面波标定（质心原点计算）
- 斜率测量（各子孔径x/y方向波前斜率）
- 斜率响应矩阵构建（Z2S矩阵）
- 波前复原（从斜率反演泽尼克系数）

MATLAB 对应: CrtSubcfg.m, CrtSubValid.m
以及 Main 脚本中散落的波前传感逻辑。
"""

import numpy as np
from .optics import Optics


def create_sub_valid(
    n_dim: int,
    near_field: np.ndarray,
    n_sub_pix: int,
    ratio: float = 0.5,
) -> np.ndarray:
    """
    计算有效子孔径掩模矩阵

    判断每个子孔径位置处近场能量是否足够，只有能量超过阈值的
    子孔径才被视为"有效"，参与后续计算。

    参数:
        n_dim: 子孔径阵列维度 (如 10 表示 10×10 阵列)
        near_field: 近场分布矩阵 (已归一化)
        n_sub_pix: 每个子孔径的像素数 (边长)
        ratio: 有效子孔径能量阈值 (0~1)，默认 0.5

    返回:
        n_dim × n_dim 的二值矩阵，1 表示有效，0 表示无效

    MATLAB 对应: CrtSubValid.m
    """
    sub_valid = np.zeros((n_dim, n_dim))
    n_pixel = near_field.shape[0]

    # 如果子孔径阵列超出图像，创建更大的掩模
    if n_dim * n_sub_pix > n_pixel:
        mask = np.zeros((n_dim * n_sub_pix, n_dim * n_sub_pix))
        st = int(np.round((n_dim * n_sub_pix - n_pixel) / 2))
        mask[st : st + n_pixel, st : st + n_pixel] = near_field
    else:
        mask = near_field

    for i in range(n_dim):
        for j in range(n_dim):
            x = i * n_sub_pix
            y = j * n_sub_pix
            temp = mask[x : x + n_sub_pix, y : y + n_sub_pix]
            if np.sum(temp) >= n_sub_pix * n_sub_pix * ratio:
                sub_valid[i, j] = 1

    return sub_valid


def create_subcfg(
    sub_valid_mat: np.ndarray,
    n_pixel: int,
    n_sub_pix: int,
) -> np.ndarray:
    """
    根据有效子孔径掩模生成子孔径坐标矩阵

    坐标以图像中心为原点，每个子孔径的坐标为其左下角像素位置。

    参数:
        sub_valid_mat: 有效子孔径掩模矩阵 (由 create_sub_valid 生成)
        n_pixel: 图像尺寸 (像素)
        n_sub_pix: 每个子孔径的像素数 (边长)

    返回:
        形状 (2, n_sub) 的坐标矩阵:
        - 第0行: y 坐标 (行方向)
        - 第1行: x 坐标 (列方向)

    MATLAB 对应: CrtSubcfg.m
    """
    W, H = sub_valid_mat.shape
    n_sub = int(np.sum(sub_valid_mat))
    subcfg = np.zeros((2, n_sub))

    num = 0
    for i in range(W):
        for j in range(H):
            if sub_valid_mat[i, j]:
                # (i-1)*nSubPix - nPixel/2  (MATLAB 1-based → Python 0-based: i*nSubPix - nPixel/2)
                subcfg[0, num] = i * n_sub_pix - n_pixel / 2  # y 坐标
                subcfg[1, num] = j * n_sub_pix - n_pixel / 2  # x 坐标
                num += 1

    return subcfg


class HartmannSensor:
    """
    哈特曼-夏克波前传感器

    模拟哈特曼传感器的完整工作流程:
    标定(平面波) → 测量斜率 → 构建响应矩阵 → 复原波前

    参数:
        subcfg: 子孔径坐标矩阵，形状 (2, n_sub)
        optics: Optics 光学系统对象

    使用示例:
        hs = HartmannSensor(subcfg, optics)
        hs.calibrate(mask, noise_sigma=0.5)
        Z2S = hs.build_response_matrix(modes, mask, n_wf_modes=15, noise_sigma=0.5)
        Recon = np.linalg.pinv(Z2S)
        slopes = hs.measure_slopes(field, noise_sigma=0.5)
        coeffs = hs.reconstruct(slopes, Recon)
    """

    def __init__(self, subcfg: np.ndarray, optics: Optics):
        self.subcfg = subcfg             # (2, n_sub)
        self.optics = optics
        self.n_sub = subcfg.shape[1]
        self.origin_hs: np.ndarray | None = None  # (2, n_sub) 标定质心位置

        # 子孔径内像素坐标网格 (0~19)，用于质心计算
        n = optics.sub_ap_pixels
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        self._X_grid = X  # (n, n)
        self._Y_grid = Y  # (n, n)

    def _extract_sub_ap_field(
        self, field: np.ndarray, i: int
    ) -> np.ndarray:
        """
        提取第 i 个子孔径对应的场区域

        将 MATLAB 的 1-based 索引转换为 Python 的 0-based 索引。
        MATLAB: y1 = Subcfg(1,i) + 128 → InputField(y1:y1+19, x1:x1+19)
        Python: y1_py = int(y1) - 1 → field[y1_py:y1_py+20, x1_py:x1_py+20]
        """
        n_pixels = self.optics.n_pixels
        n_sub_pix = self.optics.sub_ap_pixels

        # 计算 MATLAB 1-based 坐标
        y1_mat = int(np.round(self.subcfg[0, i] + n_pixels / 2))  # 1-based
        x1_mat = int(np.round(self.subcfg[1, i] + n_pixels / 2))  # 1-based

        # 转换为 Python 0-based 索引
        y1_py = y1_mat - 1
        x1_py = x1_mat - 1

        # 边界保护
        y1_py = max(0, min(y1_py, n_pixels - n_sub_pix))
        x1_py = max(0, min(x1_py, n_pixels - n_sub_pix))

        return field[y1_py : y1_py + n_sub_pix, x1_py : x1_py + n_sub_pix]

    def _compute_spot_and_centroid(
        self,
        sub_field: np.ndarray,
        noise_sigma: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        """
        对子孔径区域进行衍射计算，返回焦斑图像和质心坐标

        处理流程:
        1. DL 衍射传播得到复振幅
        2. 计算焦斑强度 |result|²
        3. 添加噪声: result + 1 + sigma*randn(20)
        4. 去基底和负值: result - 2 - 0.3*maxP, result[result<0]=0
        5. 计算质心: sumX/sum0, sumY/sum0

        返回:
            spot: 焦斑强度图 (20×20)
            cx, cy: 质心 x, y 坐标
        """
        result = self.optics.dl_propagate(sub_field)
        spot = np.abs(result) ** 2

        # 添加噪声
        if noise_sigma > 0:
            spot = spot + 1.0 + noise_sigma * np.random.randn(*spot.shape)

        # 去基底（与 MATLAB 原始处理一致）
        max_p = np.max(spot)
        spot = spot - 2.0 - 0.3 * max_p
        spot = np.maximum(spot, 0.0)

        # 质心计算
        sum0 = np.sum(spot)
        sumX = np.sum(spot * self._X_grid)
        sumY = np.sum(spot * self._Y_grid)

        cx = sumX / sum0 if sum0 != 0 else 0.0
        cy = sumY / sum0 if sum0 != 0 else 0.0

        return spot, cx, cy

    def calibrate(
        self,
        mask: np.ndarray,
        noise_sigma: float = 0.0,
    ) -> np.ndarray:
        """
        平面波标定

        以平面波（均匀近场 mask）入射，计算每个子孔径的焦斑质心
        作为参考原点 Origin_HS，用于后续斜率测量。

        参数:
            mask: 近场掩模矩阵 (平面波分布)，形状 (n_pixels, n_pixels)
            noise_sigma: 噪声标准差

        返回:
            origin_hs: 形状 (2, n_sub)，每列为一个子孔径的 (cx, cy) 质心坐标
        """
        n_pixels = self.optics.n_pixels
        n_sub_pix = self.optics.sub_ap_pixels
        self.origin_hs = np.zeros((2, self.n_sub))

        # 用于构建哈特曼合成图像的画布（可选，此处暂不返回图像）
        for i in range(self.n_sub):
            sub_field = self._extract_sub_ap_field(mask, i)
            _, cx, cy = self._compute_spot_and_centroid(sub_field, noise_sigma)
            self.origin_hs[0, i] = cx
            self.origin_hs[1, i] = cy

        return self.origin_hs

    def measure_slopes(
        self,
        field: np.ndarray,
        noise_sigma: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        测量入射场在各子孔径处的波前斜率

        参数:
            field: 入射复振幅场，形状 (n_pixels, n_pixels)
            noise_sigma: 噪声标准差

        返回:
            slopes: 形状 (2*n_sub,) 的斜率向量，
                    奇数位为 x 方向斜率，偶数位为 y 方向斜率
            img_mat: 哈特曼合成图像 (可选，用于可视化)
        """
        if self.origin_hs is None:
            raise RuntimeError("请先调用 calibrate() 进行标定")

        n_pixels = self.optics.n_pixels
        n_sub_pix = self.optics.sub_ap_pixels
        slopes = np.zeros(2 * self.n_sub)
        img_mat = np.zeros((n_pixels, n_pixels))

        for i in range(self.n_sub):
            sub_field = self._extract_sub_ap_field(field, i)
            spot, cx, cy = self._compute_spot_and_centroid(
                sub_field, noise_sigma
            )

            # 斜率 = 当前质心 - 标定质心
            slopes[2 * i] = cx - self.origin_hs[0, i]      # x 方向斜率
            slopes[2 * i + 1] = cy - self.origin_hs[1, i]  # y 方向斜率

            # 填入哈特曼图像
            y1_mat = int(np.round(self.subcfg[0, i] + n_pixels / 2))
            x1_mat = int(np.round(self.subcfg[1, i] + n_pixels / 2))
            y1_py = max(0, min(y1_mat - 1, n_pixels - n_sub_pix))
            x1_py = max(0, min(x1_mat - 1, n_pixels - n_sub_pix))
            img_mat[y1_py : y1_py + n_sub_pix, x1_py : x1_py + n_sub_pix] = spot

        return slopes, img_mat

    def build_response_matrix(
        self,
        modes: np.ndarray,
        mask: np.ndarray,
        n_wf_modes: int,
        noise_sigma: float = 0.0,
    ) -> np.ndarray:
        """
        构建斜率响应矩阵 Z2S

        对每一阶泽尼克模式（波前），计算其在各子孔径处产生的斜率。
        Z2S 矩阵的第 k 行对应第 k 阶泽尼克模式在各子孔径的斜率响应。

        参数:
            modes: 泽尼克模式矩阵，形状 (n_modes, Na, Na)
            mask: 近场掩模矩阵
            n_wf_modes: 用于波前传感的泽尼克阶数
            noise_sigma: 噪声标准差

        返回:
            Z2S: 斜率响应矩阵，形状 (n_wf_modes, 2*n_sub)
        """
        if self.origin_hs is None:
            raise RuntimeError("请先调用 calibrate() 进行标定")

        n_pixels = self.optics.n_pixels
        Z2S = np.zeros((n_wf_modes, 2 * self.n_sub))

        for k in range(n_wf_modes):
            # 构建该阶泽尼克模式对应的波前
            wf = np.zeros((n_pixels, n_pixels))
            # 泽尼克模式放置在 mask 的区域内
            offset = (n_pixels - modes.shape[1]) // 2
            na = modes.shape[1]
            wf[offset : offset + na, offset : offset + na] = modes[:, :, k]

            # 入射场 = 平面波 × exp(-i × 波前)
            input_field = mask * np.exp(-1j * wf)

            for i in range(self.n_sub):
                sub_field = self._extract_sub_ap_field(input_field, i)
                _, cx, cy = self._compute_spot_and_centroid(
                    sub_field, noise_sigma
                )

                # 斜率减去标定原点
                Z2S[k, 2 * i] = cx - self.origin_hs[0, i]      # x 斜率
                Z2S[k, 2 * i + 1] = cy - self.origin_hs[1, i]  # y 斜率

        return Z2S

    @staticmethod
    def reconstruct(slopes: np.ndarray, recon_matrix: np.ndarray) -> np.ndarray:
        """
        从斜率向量复原泽尼克波前系数

        参数:
            slopes: 斜率向量，形状 (2*n_sub,) 或 (n_samples, 2*n_sub)
            recon_matrix: 波前复原矩阵 = pinv(Z2S)，形状 (2*n_sub, n_modes)

        返回:
            复原的泽尼克系数，形状 (n_modes,) 或 (n_samples, n_modes)
        """
        return slopes @ recon_matrix
