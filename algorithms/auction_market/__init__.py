"""Emitter auction market for scarce shuttle capacity (probe).

面向稀缺运力的排放源竞价市场(探针实现)。

This package is a self-contained study layer on top of the existing physical
simulation. Each emitter is treated as a self-interested bidder that competes
for the limited empty-vessel dispatch slots. It deliberately does not modify
``Simulation`` or the other ``algorithms`` packages: it only *reads* the
physical environment, *reuses* the balanced rule as its baseline, and *reuses*
the economic ledger to decompose realised cost per party.

本包是建立在现有物理仿真之上的、独立的研究层。每个排放源被视为一个自私的竞标者,
竞争有限的空船调度名额。它刻意不修改 ``Simulation`` 或其他 ``algorithms`` 包:
只*读取*物理环境、*复用*平衡规则作为基线、并*复用*经济账本按主体分解实际成本。

Entry point / 入口:

    python -m algorithms.auction_market.probe --scenario northern_lights_phase1_milkrun_imbalanced
"""

from __future__ import annotations

from .bid_policy import SharedLinearBidPolicy
from .bidding import AuctionConfig, emitter_bids, projected_vent_bid
from .budgeted import BudgetConfig, BudgetedAuctionPolicy
from .features import FEATURE_NAMES, emitter_features
from .payoff import decompose_costs, per_emitter_costs, tariff_pnl
from .policies import AuctionDispatchPolicy, rule_action
from .runner import EpisodeResult, build_env, rollout

# The runnable studies live in ``probe`` and ``equilibrium`` and are invoked with
# ``python -m algorithms.auction_market.<module>``; they are intentionally not
# re-exported here to keep ``python -m`` execution warning-free.
# 可运行的研究位于 ``probe`` 与 ``equilibrium``,用 ``python -m`` 调用;此处刻意不
# 再导出,以保持 ``python -m`` 执行无警告。

__all__ = [
    "AuctionConfig",
    "AuctionDispatchPolicy",
    "BudgetConfig",
    "BudgetedAuctionPolicy",
    "EpisodeResult",
    "FEATURE_NAMES",
    "SharedLinearBidPolicy",
    "build_env",
    "decompose_costs",
    "emitter_bids",
    "emitter_features",
    "per_emitter_costs",
    "projected_vent_bid",
    "rollout",
    "rule_action",
    "tariff_pnl",
]
