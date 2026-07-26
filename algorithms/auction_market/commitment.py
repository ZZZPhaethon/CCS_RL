"""Adaptive commitment: how hard the market should defend the rule's plan.

自适应承诺:市场应以多大力度守住规则的计划。

A fixed commitment strength forces one compromise across every regime. The
regime map shows why that is wrong: where the rule's partition already matches
demand (balanced scenarios) any deviation is noise, while where demand is
heterogeneous or stressed the same deviation is worth tens of thousands of
tonnes. This module derives the commitment strength from a runtime signal
instead of a constant.

固定的承诺强度会在所有工况上强加同一个折中。体制图说明了这样为何不对:规则分区已经匹配
需求时(均衡场景),任何偏离都是噪声;而需求异质或受压时,同样的偏离价值数万吨。本模块用
运行时信号导出承诺强度,而非使用常数。

Signal / 信号: the **spread of emitter buffer fill ratios**. A partition that is
serving demand keeps buffers even, so a small spread means the rule is coping
and the market should stand down. A large spread means one source is backing up
while another is not -- exactly the mismatch a market can arbitrage.

信号为**各排放源缓冲填充率的极差**。分区若能匹配需求,各缓冲会保持均衡,因此极差小说明
规则应付得来、市场应当让位;极差大说明一个源在积压而另一个没有——这正是市场可以套利的
错配。
"""

from __future__ import annotations

from dataclasses import dataclass

from Simulation.entities.emitter import Emitter
from Simulation.environment import CCSEnv


_EPS = 1e-9


@dataclass(frozen=True)
class AdaptiveCommitmentConfig:
    """Map the observed fill spread to a defend floor in tonnes.

    将观测到的填充极差映射为以吨计的防守下限。
    """

    # Commitment when the rule is coping: a high floor makes the market defer.
    # 规则应付得来时的承诺:高下限使市场让位。
    floor_high_t: float = 2000.0
    # Commitment when the partition is clearly mismatched: free the market.
    # 分区明显错配时的承诺:释放市场。
    floor_low_t: float = 50.0
    # Fill-spread band over which commitment is relaxed.
    # 承诺从紧到松所跨越的填充极差区间。
    spread_low: float = 0.10
    spread_high: float = 0.40
    # Absolute-urgency band. Spread alone is blind to a uniformly stressed
    # system: when every buffer is filling, the spread is small yet the market
    # matters most. Peak fill therefore relaxes commitment independently.
    # 绝对紧迫度区间。仅看极差会漏判"整体受压":所有缓冲都在积压时极差很小,但恰是市场
    # 最有价值的时候。因此峰值填充率独立地放松承诺。
    fill_low: float = 0.50
    fill_high: float = 0.85

    def __post_init__(self) -> None:
        """Validate the adaptive bands. / 校验自适应区间。"""
        if self.floor_high_t < 0.0 or self.floor_low_t < 0.0:
            raise ValueError("Defend floors must be non-negative.")
        if not 0.0 <= self.spread_low < self.spread_high <= 1.0:
            raise ValueError("Require 0 <= spread_low < spread_high <= 1.")
        if not 0.0 <= self.fill_low < self.fill_high <= 1.0:
            raise ValueError("Require 0 <= fill_low < fill_high <= 1.")


def hours_to_overflow(env: CCSEnv) -> dict[str, float]:
    """Return each emitter's hours of headroom at its current capture rate.

    返回每个排放源按当前捕集率还能支撑的小时数。
    """
    if env.simulator is None:
        raise RuntimeError("Call env.reset() before measuring headroom.")
    state = env.simulator.state
    result: dict[str, float] = {}
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        if not isinstance(emitter, Emitter):
            raise TypeError(f"{emitter_id} is not an Emitter.")
        inventory_t = float(state.entity_inventory_t.get(emitter_id, 0.0))
        headroom_t = max(0.0, float(emitter.buffer_capacity_t) - inventory_t)
        availability = float(
            state.emitter_availability.get(emitter_id, emitter.availability)
        )
        capture_tph = float(emitter.nominal_capture_tph) * max(0.0, availability)
        result[emitter_id] = (
            float("inf") if capture_tph <= _EPS else headroom_t / capture_tph
        )
    return result


def fill_statistics(env: CCSEnv) -> tuple[float, float]:
    """Return the spread and the peak of emitter buffer fill ratios.

    返回排放源缓冲填充率的极差与峰值。
    """
    if env.simulator is None:
        raise RuntimeError("Call env.reset() before measuring fill statistics.")
    state = env.simulator.state
    fills: list[float] = []
    for emitter_id in env.emitter_ids:
        emitter = env.network.entities[emitter_id]
        if not isinstance(emitter, Emitter):
            raise TypeError(f"{emitter_id} is not an Emitter.")
        capacity_t = max(_EPS, float(emitter.buffer_capacity_t))
        inventory_t = float(state.entity_inventory_t.get(emitter_id, 0.0))
        fills.append(inventory_t / capacity_t)
    if not fills:
        return 0.0, 0.0
    return max(fills) - min(fills), max(fills)


def adaptive_defend_floor_t(
    env: CCSEnv,
    config: AdaptiveCommitmentConfig,
) -> float:
    """Return the defend floor implied by the current fill spread.

    返回当前填充极差所对应的防守下限。

    Commitment is relaxed by whichever signal is stronger: an uneven partition
    (spread) or a uniformly stressed system (peak fill). Taking the maximum
    means either condition alone is enough to open the market.

    承诺按两个信号中较强者放松:分区不均(极差)或系统整体受压(峰值填充)。取最大值意味着
    任一条件单独成立即可开放市场。

    Interpolates from ``floor_high_t`` (defer to the rule) down to
    ``floor_low_t`` (let the market reallocate).

    从 ``floor_high_t``(让位于规则)线性插值到 ``floor_low_t``(让市场重分配)。
    """
    spread, peak_fill = fill_statistics(env)
    spread_weight = (spread - config.spread_low) / (
        config.spread_high - config.spread_low
    )
    fill_weight = (peak_fill - config.fill_low) / (
        config.fill_high - config.fill_low
    )
    weight = max(spread_weight, fill_weight)
    weight = min(1.0, max(0.0, weight))
    return config.floor_high_t + weight * (config.floor_low_t - config.floor_high_t)
