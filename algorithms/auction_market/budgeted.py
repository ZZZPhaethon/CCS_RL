"""Budgeted uniform-price auction: make money actually affect allocation.

带预算约束的统一价格拍卖:让价格真正影响分配。

In the unbudgeted auction the allocation depends only on the *ranking* of bids,
so payments are pure transfers and strategic shading cannot change the physical
dispatch (Price of Anarchy = 1). This module couples payments to allocation:

  1. **Uniform clearing price** -- winners pay the highest *rejected* bid, so the
     price is endogenous: it is zero when vessels are plentiful and rises exactly
     when emitters contend for scarce capacity (congestion pricing).
  2. **Budgets** -- each emitter holds a depletable per-episode budget. An
     emitter that cannot afford the clearing price loses its slot to the next
     affordable bidder, so spending now genuinely forfeits capacity later.

在无预算拍卖中,分配只取决于出价*排序*,支付是纯转移,策略性压价无法改变物理调度
(无序代价 = 1)。本模块把支付与分配耦合:(1) **统一出清价**——赢家支付最高*落选*出价,
价格内生:运力充裕时为零,恰在排放源争夺稀缺运力时上升(拥塞定价);(2) **预算**——每个
排放源持有可耗尽的回合预算,付不起出清价就把名额让给下一个付得起的竞标者,因此"现在花钱"
真的意味着"以后失去运力"。

Allocation procedure at one decision point / 单个决策点的分配流程:

    a. eligible bidders = positive bid and positive remaining budget;
    b. greedily assign the nearest legal empty vessel in descending bid order;
    c. clearing price p = highest bid among eligible bidders that won nothing
       (zero if everyone who bid was served -- i.e. no congestion);
    d. repair: drop winners whose budget < p and offer the freed vessel to the
       next affordable, reachable bidder;
    e. charge every final winner p and decrement its budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from Simulation.entities.emitter import Emitter
from Simulation.environment import CCSEnv

from .bidding import AuctionConfig, emitter_bids
from .policies import (
    _auction_eligible_vessels,
    _travel_hours,
    baseline_emitter_map,
    may_reassign,
    rule_action,
)


_EPS = 1e-9


@dataclass(frozen=True)
class BudgetConfig:
    """Configure per-episode emitter budgets.

    配置排放源的回合预算。
    """

    # Budget = budget_factor x carbon_price x nominal_capture_tph x episode_hours.
    # A factor of 1.0 endows an emitter with the full carbon value of everything
    # it captures; smaller values bind harder. ``None`` disables budgets.
    # 预算 = budget_factor x 碳价 x 标称捕集率 x 回合小时数。取 1.0 表示按其全部捕集量
    # 的碳价值授予预算;取值越小约束越紧。``None`` 表示不启用预算。
    budget_factor: float | None = 0.25

    def __post_init__(self) -> None:
        """Validate the budget factor. / 校验预算系数。"""
        if self.budget_factor is not None and self.budget_factor < 0.0:
            raise ValueError("budget_factor must be non-negative or None.")


class BudgetedAuctionPolicy:
    """Uniform-price shuttle auction with depletable per-emitter budgets.

    带可耗尽排放源预算的统一价格运力拍卖。

    Like :class:`AuctionDispatchPolicy` it only re-targets empty vessels the rule
    was already dispatching, so the physical envelope is unchanged; unlike it,
    the clearing price and the budgets can change *who* is served.

    与 :class:`AuctionDispatchPolicy` 一样,它只重指派规则本就要派出的空船,物理包络不变;
    不同之处在于出清价与预算会改变*谁*被服务。
    """

    def __init__(
        self,
        config: AuctionConfig | None = None,
        budget: BudgetConfig | None = None,
        *,
        bid_shades: dict[str, float] | None = None,
        bid_model: object | None = None,
        commitment_margin: float = 0.0,
        defend_floor_t: float = 0.0,
    ) -> None:
        """Store auction, budget, and bidding configuration.

        保存拍卖、预算与竞价配置。
        """
        if commitment_margin < 0.0:
            raise ValueError("commitment_margin must be non-negative.")
        if defend_floor_t < 0.0:
            raise ValueError("defend_floor_t must be non-negative.")
        self.config = config or AuctionConfig()
        self.budget = budget or BudgetConfig()
        self.bid_shades = dict(bid_shades) if bid_shades else {}
        self.bid_model = bid_model
        self.commitment_margin = float(commitment_margin)
        self.defend_floor_t = float(defend_floor_t)
        self.budgets: dict[str, float] = {}
        self.payments: dict[str, float] = {}
        self.clearing_prices: list[float] = []
        self.budget_blocked_events = 0
        self._initialised = False

    def __call__(self, env: CCSEnv) -> dict[str, list[int]]:
        """Allow use as ``policy(env)``. / 允许作为 ``policy(env)`` 使用。"""
        return self.propose_action(env)

    def _initialise_budgets(self, env: CCSEnv) -> None:
        """Endow each emitter with its per-episode budget.

        为每个排放源授予其回合预算。
        """
        factor = self.budget.budget_factor
        episode_hours = float(env.config.episode_hours)
        for emitter_id in env.emitter_ids:
            if factor is None:
                self.budgets[emitter_id] = float("inf")
                continue
            emitter = env.network.entities[emitter_id]
            if not isinstance(emitter, Emitter):
                raise TypeError(f"{emitter_id} is not an Emitter.")
            self.budgets[emitter_id] = (
                factor
                * self.config.carbon_price_eur_per_t
                * float(emitter.nominal_capture_tph)
                * episode_hours
            )
            self.payments.setdefault(emitter_id, 0.0)
        self._initialised = True

    def _submitted_bids(self, env: CCSEnv) -> dict[str, float]:
        """Return each emitter's submitted bid.

        返回各排放源的提交出价。
        """
        true_values = emitter_bids(env, self.config)
        if self.bid_model is not None:
            return {
                emitter_id: float(self.bid_model.submit(env, emitter_id, value))
                for emitter_id, value in true_values.items()
            }
        return {
            emitter_id: float(self.bid_shades.get(emitter_id, 1.0)) * value
            for emitter_id, value in true_values.items()
        }

    def propose_action(self, env: CCSEnv) -> dict[str, list[int]]:
        """Return the rule action with budget-feasible auction winners applied.

        返回叠加了预算可行拍卖赢家的规则动作。
        """
        if env.simulator is None:
            raise RuntimeError("Call env.reset() before requesting an action.")
        if not self._initialised:
            self._initialise_budgets(env)

        baseline = rule_action(env)
        action = {
            "vessels": list(baseline["vessels"]),
            "wells": list(baseline["wells"]),
        }
        masks = env.vessel_action_mask()
        free = _auction_eligible_vessels(env, masks, baseline)
        if not free:
            return action

        bids = self._submitted_bids(env)
        # (a) eligible = positive bid and positive remaining budget.
        # (a) 候选 = 出价为正且尚有预算。
        ranked = sorted(
            (
                (emitter_id, bid)
                for emitter_id, bid in bids.items()
                if bid > self.config.reserve_price_eur
                and self.budgets.get(emitter_id, 0.0) > _EPS
            ),
            key=lambda kv: -kv[1],
        )
        if not ranked:
            return action

        # (b) greedy assignment in descending bid order.
        # (b) 按出价降序贪心分配。
        baseline_emitter = baseline_emitter_map(env, baseline)
        winners, used = self._assign(
            env, masks, free, ranked, set(), baseline_emitter, bids
        )

        # (c) clearing price = highest bid among eligible bidders that won nothing.
        # (c) 出清价 = 未中标候选者中的最高出价。
        clearing_price = max(
            (bid for emitter_id, bid in ranked if emitter_id not in winners),
            default=0.0,
        )

        # (d) repair: drop winners that cannot afford the clearing price.
        # (d) 修复:剔除付不起出清价的赢家。
        unaffordable = [
            emitter_id
            for emitter_id in winners
            if self.budgets.get(emitter_id, 0.0) + _EPS < clearing_price
        ]
        if unaffordable:
            self.budget_blocked_events += len(unaffordable)
            for emitter_id in unaffordable:
                used.discard(winners.pop(emitter_id))
            remaining_free = [
                (position, vessel_id)
                for position, vessel_id in free
                if vessel_id not in used
            ]
            affordable = [
                (emitter_id, bid)
                for emitter_id, bid in ranked
                if emitter_id not in winners
                and emitter_id not in unaffordable
                and self.budgets.get(emitter_id, 0.0) + _EPS >= clearing_price
            ]
            extra, used = self._assign(
                env, masks, remaining_free, affordable, used,
                baseline_emitter, bids,
            )
            winners.update(extra)

        # (e) charge winners and write their destinations into the action.
        # (e) 向赢家收费并将其目的地写入动作。
        for emitter_id, vessel_id in winners.items():
            target_action = env.vessel_go_emitter_action(emitter_id)
            action["vessels"][env.vessel_ids.index(vessel_id)] = target_action
            if self.budget.budget_factor is not None:
                self.budgets[emitter_id] -= clearing_price
            self.payments[emitter_id] = (
                self.payments.get(emitter_id, 0.0) + clearing_price
            )
        self.clearing_prices.append(clearing_price)
        return action

    def _assign(
        self,
        env: CCSEnv,
        masks: list[list[bool]],
        free: list[tuple[int, str]],
        ranked: list[tuple[str, float]],
        used: set[str],
        baseline_emitter: dict[int, str | None],
        bids: dict[str, float],
    ) -> tuple[dict[str, str], set[str]]:
        """Assign the nearest legal free vessel in descending bid order.

        按出价降序把最近的合法空船分配给竞标者。
        """
        winners: dict[str, str] = {}
        for emitter_id, _bid in ranked:
            target_action = env.vessel_go_emitter_action(emitter_id)
            candidates = [
                (position, vessel_id)
                for position, vessel_id in free
                if vessel_id not in used
                and masks[position][target_action]
                and may_reassign(
                    emitter_id,
                    baseline_emitter.get(position),
                    bids,
                    self.commitment_margin,
                    self.defend_floor_t * self.config.carbon_price_eur_per_t,
                )
            ]
            if not candidates:
                continue
            _position, vessel_id = min(
                candidates,
                key=lambda pv: _travel_hours(env, pv[1], emitter_id),
            )
            winners[emitter_id] = vessel_id
            used.add(vessel_id)
        return winners, used
