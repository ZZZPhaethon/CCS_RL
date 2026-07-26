"""Decompose realised cost into emitter / shipping / storage parties.

将实际成本分解为排放方 / 船运方 / 封存方三方。

The decomposition is exact and introduces no invented numbers: it reuses the
fields already tracked by ``Simulation.economics.EconomicLedger``.

该分解是精确的,不引入任何虚构数字:它复用 ``Simulation.economics.EconomicLedger``
已经记录的字段。

    emitter   = conditioning (source-side prep) + vent_penalty (carbon value)
    shipping  = vessel_fuel + loading + unloading (hoteling fuel)
    storage   = reconditioning (terminal-side prep)

An optional tariff overlay converts the shared chain into a per-party profit and
loss, so that each party has its own interest. Tariffs are transfers between the
parties and are clearly labelled as modelling assumptions, not physical costs.

可选的资费叠加把共享链条转换为每一方的盈亏,从而让每一方拥有自己的利益。资费是各方
之间的转移支付,已明确标注为建模假设,而非物理成本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from Simulation.economics import EconomicLedger


@dataclass(frozen=True)
class PartyCosts:
    """Exact per-party operating and environmental cost (EUR, costs positive).

    精确的每一方运营与环境成本(欧元,成本为正)。
    """

    emitter_conditioning: float
    emitter_vent_penalty: float
    shipping_fuel: float
    shipping_loading: float
    shipping_unloading: float
    storage_reconditioning: float

    @property
    def emitter_total(self) -> float:
        """Total cost borne by emitters. / 排放方承担的总成本。"""
        return self.emitter_conditioning + self.emitter_vent_penalty

    @property
    def shipping_total(self) -> float:
        """Total cost borne by the shipping operator. / 船运方承担的总成本。"""
        return self.shipping_fuel + self.shipping_loading + self.shipping_unloading

    @property
    def storage_total(self) -> float:
        """Total cost borne by the storage operator. / 封存方承担的总成本。"""
        return self.storage_reconditioning

    @property
    def chain_total(self) -> float:
        """Sum across parties (excludes storage-shortfall diagnostics).

        三方之和(不含储存缺额诊断项)。
        """
        return self.emitter_total + self.shipping_total + self.storage_total

    def as_dict(self) -> dict[str, float]:
        """Return a flat dictionary for CSV export. / 返回用于 CSV 导出的扁平字典。"""
        return {
            "emitter_conditioning_eur": self.emitter_conditioning,
            "emitter_vent_penalty_eur": self.emitter_vent_penalty,
            "emitter_total_eur": self.emitter_total,
            "shipping_fuel_eur": self.shipping_fuel,
            "shipping_loading_eur": self.shipping_loading,
            "shipping_unloading_eur": self.shipping_unloading,
            "shipping_total_eur": self.shipping_total,
            "storage_reconditioning_eur": self.storage_reconditioning,
            "storage_total_eur": self.storage_total,
            "chain_total_eur": self.chain_total,
        }


def decompose_costs(ledger: EconomicLedger) -> PartyCosts:
    """Split a realised economic ledger into the three parties' costs.

    将实际经济账本分解为三方成本。
    """
    return PartyCosts(
        emitter_conditioning=float(ledger.conditioning),
        emitter_vent_penalty=float(ledger.vent_penalty),
        shipping_fuel=float(ledger.vessel_fuel),
        shipping_loading=float(ledger.loading),
        shipping_unloading=float(ledger.unloading),
        storage_reconditioning=float(ledger.reconditioning),
    )


def per_emitter_costs(
    per_emitter_captured_t: Mapping[str, float],
    per_emitter_vented_t: Mapping[str, float],
    carbon_price_eur_per_t: float,
) -> dict[str, dict[str, float]]:
    """Return each emitter's vent cost and service rate (winners vs losers).

    返回每个排放源的放空成本与服务率(赢家 vs 输家)。
    """
    result: dict[str, dict[str, float]] = {}
    for emitter_id, captured_t in per_emitter_captured_t.items():
        vented_t = float(per_emitter_vented_t.get(emitter_id, 0.0))
        captured = float(captured_t)
        service_rate = 1.0 - vented_t / captured if captured > 1e-9 else 1.0
        result[emitter_id] = {
            "captured_t": captured,
            "vented_t": vented_t,
            "vent_cost_eur": carbon_price_eur_per_t * vented_t,
            "service_rate": service_rate,
        }
    return result


@dataclass(frozen=True)
class TariffPnL:
    """Chain-level three-party profit/loss under a tariff overlay (EUR).

    资费叠加下的链条级三方盈亏(欧元)。
    """

    transport_tariff_eur_per_t: float
    injection_tariff_eur_per_t: float
    shipping_revenue: float
    shipping_profit: float
    storage_revenue: float
    storage_profit: float
    emitter_payment: float
    emitter_total: float

    def as_dict(self) -> dict[str, float]:
        """Return a flat dictionary for reporting. / 返回用于报告的扁平字典。"""
        return {
            "transport_tariff_eur_per_t": self.transport_tariff_eur_per_t,
            "injection_tariff_eur_per_t": self.injection_tariff_eur_per_t,
            "shipping_revenue_eur": self.shipping_revenue,
            "shipping_profit_eur": self.shipping_profit,
            "storage_revenue_eur": self.storage_revenue,
            "storage_profit_eur": self.storage_profit,
            "emitter_payment_eur": self.emitter_payment,
            "emitter_total_eur": self.emitter_total,
        }


def tariff_pnl(
    ledger: EconomicLedger,
    costs: PartyCosts,
    *,
    transport_tariff_eur_per_t: float,
    injection_tariff_eur_per_t: float,
) -> TariffPnL:
    """Overlay simple tariffs so each party has an explicit profit and loss.

    叠加简单资费,使每一方都有明确的盈亏。

    Shipping charges the emitters ``transport_tariff`` per loaded tonne; storage
    charges ``injection_tariff`` per stored tonne. These are transfers, not new
    physical costs. Emitters pay both tariffs plus their own conditioning and
    vent costs.

    船运方按每装载吨向排放方收取 ``transport_tariff``;封存方按每封存吨收取
    ``injection_tariff``。它们是转移支付,而非新增物理成本。排放方支付两项资费,
    外加自身的调理与放空成本。
    """
    loaded_t = float(ledger.loaded_t)
    stored_t = float(ledger.stored_t)
    shipping_revenue = transport_tariff_eur_per_t * loaded_t
    storage_revenue = injection_tariff_eur_per_t * stored_t
    emitter_payment = shipping_revenue + storage_revenue
    return TariffPnL(
        transport_tariff_eur_per_t=transport_tariff_eur_per_t,
        injection_tariff_eur_per_t=injection_tariff_eur_per_t,
        shipping_revenue=shipping_revenue,
        shipping_profit=shipping_revenue - costs.shipping_total,
        storage_revenue=storage_revenue,
        storage_profit=storage_revenue - costs.storage_total,
        emitter_payment=emitter_payment,
        emitter_total=emitter_payment + costs.emitter_total,
    )
