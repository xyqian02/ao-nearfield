"""
极限学习机(ELM) 与 神经网络基类模块

提供:
- BaseModel: 神经网络模型抽象基类，定义统一的 fit/predict 接口，
  方便后续替换为 U-Net、Transformer 等其他网络架构
- ELM: 极限学习机实现，支持多种激活函数

MATLAB 对应: elmtrain.m, elmpredict.m
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    神经网络模型抽象基类

    所有神经网络模型（ELM、U-Net、Transformer 等）均需继承此类，
    实现 fit() 和 predict() 方法，确保接口统一，方便在主函数中替换。
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练模型

        参数:
            X: 输入数据，形状 (n_features, n_samples)
            y: 输出标签，形状 (n_outputs, n_samples)
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        模型预测

        参数:
            X: 输入数据，形状 (n_features, n_samples)

        返回:
            预测值，形状 (n_outputs, n_samples)
        """
        ...


class ELM(BaseModel):
    """
    极限学习机 (Extreme Learning Machine)

    单隐藏层前馈神经网络，输入权重和偏置随机生成，
    输出权重通过伪逆（最小二乘）解析求解，无需反向传播。

    参数:
        n_hidden: 隐藏层神经元数量
        activation: 激活函数类型，可选:
            'sigmoid'  - Sigmoid: 1/(1+e^{-x})
            'relu'     - ReLU: max(0, x)
            'softplus' - Softplus: ln(1+e^x)
            'tanh'     - Tanh: tanh(x)
            'sin'      - 正弦: sin(x)
            'rbf'      - 径向基: exp(-x²)

    使用示例:
        elm = ELM(n_hidden=850, activation='softplus')
        elm.fit(X_train, y_train)
        y_pred = elm.predict(X_test)

    MATLAB 对应: elmtrain.m, elmpredict.m
    """

    # 支持的激活函数映射表
    _ACTIVATIONS = {
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "relu": lambda x: np.maximum(x, 0.0),
        "softplus": lambda x: np.log(1.0 + np.exp(x)),
        "tanh": lambda x: np.tanh(x),
        "sin": lambda x: np.sin(x),
        "rbf": lambda x: np.exp(-(x**2)),
    }

    def __init__(self, n_hidden: int = 100, activation: str = "softplus"):
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"不支持的激活函数 '{activation}'，"
                f"可选: {list(self._ACTIVATIONS.keys())}"
            )

        self.n_hidden = n_hidden
        self.activation = activation
        self._activate = self._ACTIVATIONS[activation]

        # 模型参数（训练后填充）
        self.IW: np.ndarray | None = None  # 输入权重矩阵 (N, R)
        self.B: np.ndarray | None = None   # 偏置向量 (N, 1)
        self.LW: np.ndarray | None = None  # 输出权重矩阵 (N, S)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练 ELM 模型

        1. 随机生成输入权重 IW ∈ [-1, 1] 和偏置 B
        2. 计算隐藏层输出 H = activation(IW @ X + B)
        3. 通过伪逆求解输出权重 LW = pinv(H^T) @ y^T

        参数:
            X: 输入数据，形状 (R, Q) = (n_features, n_samples)
            y: 输出标签，形状 (S, Q) = (n_outputs, n_samples)
        """
        R, Q = X.shape  # R=输入特征数, Q=样本数
        N = self.n_hidden

        # 随机生成输入权重和偏置 (范围 [-1, 1])
        self.IW = np.random.rand(N, R) * 2.0 - 1.0  # (N, R)
        self.B = np.random.rand(N, 1)                # (N, 1)

        # 计算隐藏层输出: H = activation(IW @ X + B)
        tempH = self.IW @ X + self.B  # (N, Q)
        H = self._activate(tempH)     # (N, Q)

        # 通过伪逆求解输出权重: LW = pinv(H^T) @ y^T
        # H^T 形状 (Q, N), y^T 形状 (Q, S), LW 形状 (N, S)
        # 容差匹配 MATLAB pinv: tol = max(size(A)) * norm(A) * eps
        rcond = max(H.T.shape) * np.finfo(np.float64).eps
        self.LW = np.linalg.pinv(H.T, rcond=rcond) @ y.T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的 ELM 进行预测

        参数:
            X: 输入数据，形状 (R, Q) = (n_features, n_samples)

        返回:
            预测值，形状 (S, Q) = (n_outputs, n_samples)
        """
        if self.IW is None or self.B is None or self.LW is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        Q = X.shape[1]

        # 计算隐藏层输出
        tempH = self.IW @ X + self.B  # (N, Q)
        H = self._activate(tempH)     # (N, Q)

        # 计算输出: Y = (H^T @ LW)^T
        # H^T 形状 (Q, N), LW 形状 (N, S), 结果 (Q, S) → 转置为 (S, Q)
        y_pred = (H.T @ self.LW).T

        return y_pred

    def __repr__(self) -> str:
        return f"ELM(n_hidden={self.n_hidden}, activation='{self.activation}')"
