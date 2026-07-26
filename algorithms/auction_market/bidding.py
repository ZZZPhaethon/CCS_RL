"""Physically grounded emitter bids for the shuttle auction.

为运力竞价提供物理层根据的排放源出价。

An emitter's willingness to pay for the next vessel is the carbon value of the
CO2 it would be forced to vent before it is next served. This mirrors the
environment's own overflow-risk measure (``CCSEnv._overflow_risk_t``): a bid is
never an arbitrary number, it is the projected vent loss over a service horizon.

一个排放源为下一艘船愿意支付的价格,等于它在下次被服务前会被迫放空的 CO2 的碳价值。
这与环境自身的溢出风险度量(``CCSEnv._overflow_risk_t``)一致:出价不是任意数字,
而是一个服务时域内的预计放空损失。
"""

from __future__ import annotations

from dataclasses import dataclass

from Simulation.entities.emitter import Emitter
from Simulation.environment import CCSEnv


_EPS = 1e-9


@dataclass(frozen=True)
class AuctionConfig:
    """Configure the greedy shuttle auction and its bid valuation.

    配置贪心运力拍卖及其出价估值。
    """

    # Carbon value used to price the projected vent loss (EUR/t). Kept aligned
    # with EconomicParameters.carbon_price_eur_per_t by default.
    # 用于给预计放空损失定价的碳价(欧元/吨),默认与经济参数中的碳价一致。
    carbon_price_eur_per_t: float = 80.0

    # Service horizon over which the projected vent (and thus the bid) is
    # evaluated. 48 h matches the v3 risk gate's hours-to-overflow threshold.
    # 评估预计放空(即出价)的服务时域。48 小时与 v3 风险门控的溢出时限一致。
    bid_horizon_h: float = 48.0

    # An emitter below this bid does not compete for a vessel this step.
    # 出价低于该值的排放源在本步不参与竞争。
    reserve_price_eur: float = 0.0

    def __post_init__(self) -> None:
        """Validate auction settings.

        校验拍卖设置。
        """
        if self.carbon_price_eur_per_t < 0.0:
            raise ValueError("carbon_price_eur_per_t must be non-negative.")
        if self.bid_horizon_h <= 0.0:
            raise ValueError("bid_horizon_h must be positive.")
        if self.reserve_price_eur < 0.0:
            raise ValueError("reserve_price_eur must be non-negative.")


def projected_vent_bid(
    env: CCSEnv,
    emitter_id: str,
    config: AuctionConfig,
) -> float:
    """Return one emitter's bid = carbon value of its projected vent loss.

    返回单个排放源的出价 = 其预计放空损失的碳价值。
    """
    if env.simulator is None:
        raise RuntimeError("Call env.reset() before computing bids.")
    emitter = env.network.entities[emitter_id]
    if not isinstance(emitter, Emitter):
        raise TypeError(f"{emitter_id} is not an Emitter.")
    state = env.simulator.state
    inventory_t = float(state.entity_inventory_t.get(emitter_id, 0.0))
    headroom_t = max(0.0, float(emitter.buffer_capacity_t) - inventory_t)
    availability = float(
        state.emitter_availability.get(emitter_id, emitter.availability)
    )
    capture_tph = float(emitter.nominal_capture_tph) * max(0.0, availability)
    projected_vent_t = max(0.0, capture_tph * config.bid_horizon_h - headroom_t)
    return config.carbon_price_eur_per_t * projected_vent_t


def emitter_bids(env: CCSEnv, config: AuctionConfig) -> dict[str, float]:
    """Return every emitter's current bid.

    返回所有排放源的当前出价。
    """
    return {
        emitter_id: projected_vent_bid(env, emitter_id, config)
        for emitter_id in env.emitter_ids
    }
