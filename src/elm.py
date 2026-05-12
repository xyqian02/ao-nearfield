"""
极限学习机 (ELM) 模块 — sklearn 兼容实现

单隐藏层前馈神经网络，输入权重随机生成，输出权重通过解析求解。
继承 sklearn BaseEstimator / RegressorMixin，兼容 Pipeline 和 GridSearchCV。

MATLAB 对应: elmtrain.m, elmpredict.m
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_random_state


class ELM(BaseEstimator, RegressorMixin):
    """
    极限学习机 (Extreme Learning Machine)

    单隐藏层前馈神经网络：输入权重和偏置随机生成，
    输出权重通过伪逆（最小二乘）解析求解，无需反向传播。

    参数:
        n_hidden: 隐藏层神经元数量
        activation: 激活函数，可选 'sigmoid', 'relu', 'softplus', 'tanh', 'sin', 'rbf'
        alpha: L2 正则化系数，0=无正则化（默认），>0 时使用岭回归求解
        random_state: 随机种子，控制输入权重和偏置的生成

    sklearn 数据约定:
        X: (n_samples, n_features)
        y: (n_samples,) 或 (n_samples, n_outputs)

    使用示例:
        elm = ELM(n_hidden=850, activation='softplus', random_state=42)
        elm.fit(X_train, y_train)
        y_pred = elm.predict(X_test)
    """

    _ACTIVATIONS = {
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "relu": lambda x: np.maximum(x, 0.0),
        "softplus": lambda x: np.log(1.0 + np.exp(x)),
        "tanh": lambda x: np.tanh(x),
        "sin": lambda x: np.sin(x),
        "rbf": lambda x: np.exp(-(x**2)),
    }

    def __init__(
        self,
        n_hidden: int = 100,
        activation: str = "softplus",
        alpha: float = 0.0,
        random_state: int | np.random.RandomState | None = None,
    ):
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"不支持的激活函数 '{activation}'，"
                f"可选: {list(self._ACTIVATIONS.keys())}"
            )
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ELM":
        """
        训练 ELM 模型

        参数:
            X: (n_samples, n_features)
            y: (n_samples,) 或 (n_samples, n_outputs)

        返回:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        Q, R = X.shape   # Q=样本数, R=特征数
        S = y.shape[1]    # 输出维度
        N = self.n_hidden

        # 内部转置为 (n_features, n_samples) 进行矩阵运算
        X_t = X.T  # (R, Q)
        y_t = y.T  # (S, Q)

        # 随机生成输入权重和偏置 [-1, 1]
        rng = check_random_state(self.random_state)
        self.IW_ = rng.rand(N, R) * 2.0 - 1.0  # (N, R)
        self.B_ = rng.rand(N, 1)                # (N, 1)

        # 隐藏层输出: H = activation(IW @ X_t + B)
        tempH = self.IW_ @ X_t + self.B_  # (N, Q)
        H = self._ACTIVATIONS[self.activation](tempH)

        # 求解输出权重
        if self.alpha > 0:
            # L2 正则化 (岭回归): (H@H^T + alpha*I) @ LW = H @ y^T
            A = H @ H.T + self.alpha * np.eye(N)
            self.LW_ = np.linalg.solve(A, H @ y_t.T)  # (N, S)
        else:
            # 标准 ELM: LW = pinv(H^T) @ y^T，容差匹配 MATLAB pinv 行为
            rcond = max(H.T.shape) * np.finfo(np.float64).eps
            self.LW_ = np.linalg.pinv(H.T, rcond=rcond) @ y_t.T  # (N, S)

        self._n_features_in_ = R
        self._n_outputs_ = S
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        模型预测

        参数:
            X: (n_samples, n_features)

        返回:
            (n_samples,) 或 (n_samples, n_outputs)
        """
        X = np.asarray(X, dtype=np.float64)

        X_t = X.T  # (R, Q)
        tempH = self.IW_ @ X_t + self.B_
        H = self._ACTIVATIONS[self.activation](tempH)

        y_pred = H.T @ self.LW_  # (Q, S)

        if y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        return y_pred

    def __repr__(self) -> str:
        return (
            f"ELM(n_hidden={self.n_hidden}, activation='{self.activation}', "
            f"alpha={self.alpha}, random_state={self.random_state})"
        )
