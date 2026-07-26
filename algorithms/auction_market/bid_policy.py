"""A shared, state-conditioned bidding policy for the emitter agents.

面向排放源智能体的、共享且基于状态的竞价策略。

Every emitter applies the *same* parameters to its *own* local features
(parameter sharing = decentralized execution): each agent bids from local
information only, but all agents share one learned valuation. The multiplier is
``exp(clip(w . features))`` scaling the emitter's true value, so the zero policy
(``w = 0``) reproduces the myopic truthful auction exactly -- a clean baseline
and a natural learning start point.

每个排放源把*相同*的参数作用于其*自身*的局部特征(参数共享=去中心化执行):每个智能体
仅依据本地信息出价,但所有智能体共享同一学习到的估值。乘子为 ``exp(clip(w . features))``,
用于缩放该排放源的真实价值;因此零策略(``w = 0``)精确复现近视如实拍卖——既是干净的
基线,也是自然的学习起点。
"""

from __future__ import annotations

from math import exp

import numpy as np

from Simulation.environment import CCSEnv

from .features import N_FEATURES, emitter_features


_CLIP = 4.0


class SharedLinearBidPolicy:
    """Map local emitter features to a bid multiplier with shared weights.

    用共享权重把排放源局部特征映射为出价乘子。
    """

    def __init__(self, params: np.ndarray | None = None) -> None:
        """Initialise from a flat parameter vector (defaults to the myopic bid).

        用扁平参数向量初始化(默认为近视出价)。
        """
        if params is None:
            self.params = np.zeros(N_FEATURES, dtype=float)
        else:
            params = np.asarray(params, dtype=float)
            if params.shape != (N_FEATURES,):
                raise ValueError(
                    f"Expected {N_FEATURES} parameters, got {params.shape}."
                )
            self.params = params

    @property
    def n_params(self) -> int:
        """Return the number of policy parameters. / 返回策略参数数量。"""
        return N_FEATURES

    def multiplier(self, features: np.ndarray) -> float:
        """Return the strictly positive value multiplier for one emitter.

        返回单个排放源的严格正值乘子。
        """
        z = float(np.dot(self.params, features))
        return exp(max(-_CLIP, min(_CLIP, z)))

    def submit(self, env: CCSEnv, emitter_id: str, true_value: float) -> float:
        """Return the submitted bid = multiplier(local features) x true value.

        返回提交出价 = 乘子(局部特征) x 真实价值。
        """
        features = emitter_features(env, emitter_id)
        return self.multiplier(features) * float(true_value)

    def get_params(self) -> np.ndarray:
        """Return a copy of the flat parameter vector. / 返回参数向量副本。"""
        return self.params.copy()

    def set_params(self, params: np.ndarray) -> None:
        """Overwrite the flat parameter vector. / 覆盖参数向量。"""
        params = np.asarray(params, dtype=float)
        if params.shape != (N_FEATURES,):
            raise ValueError(
                f"Expected {N_FEATURES} parameters, got {params.shape}."
            )
        self.params = params
