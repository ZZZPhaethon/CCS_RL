"""Baseline rule action and the greedy shuttle-auction dispatch policy.

基线规则动作与贪心运力拍卖调度策略。

The auction reuses the balanced cluster-shuttle rule as its safe default and
then *reassigns only the empty, dispatchable vessels* to the highest-bidding
emitters. Everything else (loading, unloading, terminal returns, well rates,
masks) is exactly the existing rule, so the auction can only differ in *which
source a free vessel is sent to* -- the precise point of contention between
emitters.

拍卖复用平衡 cluster-shuttle 规则作为安全默认,随后*只把空载、可调度的船*重新分配
给出价最高的排放源。其余一切(装载、卸载、返港、注入率、掩码)完全沿用现有规则,
因此拍卖只可能在"一艘空船被派往哪个源"上有所不同——这正是排放源之间的争夺点。
"""

from __future__ import annotations

from math import inf

from Simulation.control.baselines import (
    balanced_capture_assignment,
    make_cluster_shuttle_policy,
)
from Simulation.environment import CCSEnv

from .bidding import AuctionConfig, emitter_bids


_EPS = 1e-9


def rule_action(env: CCSEnv) -> dict[str, list[int]]:
    """Return the balanced cluster-shuttle rule action (the shared baseline).

    返回平衡 cluster-shuttle 规则动作(共享基线)。
    """
    assignment = balanced_capture_assignment(env)
    env.set_goal_assignment(assignment)
    action = make_cluster_shuttle_policy(env, assignment)(env)
    action["wells"] = [
        env.highest_feasible_well_rate_index(well_id)
        for well_id in env.well_ids
    ]
    return {
        "vessels": list(action["vessels"]),
        "wells": list(action["wells"]),
    }


class AuctionDispatchPolicy:
    """Send each free vessel to the highest-bidding reachable emitter.

    把每艘空船派往出价最高且可达的排放源。

    The policy is stateless across hours: because a dispatched vessel leaves the
    berth on the next step, it simply drops out of the eligible set, so the
    auction naturally re-runs for whoever is still empty and berthed.

    该策略在小时之间无状态:一艘被派出的船下一步即离开泊位,自动退出候选集,
    因此拍卖会自然地为仍然空载靠泊的船重新进行。
    """

    def __init__(
        self,
        config: AuctionConfig | None = None,
        *,
        bid_shades: dict[str, float] | None = None,
        payment_rule: str = "none",
        bid_model: object | None = None,
        commitment_margin: float = 0.0,
        defend_floor_t: float = 0.0,
        adaptive_commitment: object | None = None,
        protect_horizon_h: float = 0.0,
    ) -> None:
        """Store the auction configuration, bidding strategy, and diagnostics.

        保存拍卖配置、竞价策略与诊断信息。

        ``bid_shades`` scales each emitter's *true* value into its *submitted*
        bid (1.0 = truthful, <1 = shading, >1 = aggressive). ``payment_rule`` is
        ``none`` (allocation only), ``first_price`` (winner pays its own bid), or
        ``second_price`` (winner pays the highest rejected bid; truthful). If a
        ``bid_model`` with ``submit(env, emitter_id, true_value)`` is given, it
        overrides ``bid_shades`` and produces the submitted bid from state.

        ``bid_shades`` 把每个排放源的*真实*价值缩放为*提交*出价。``payment_rule``
        为 ``none`` / ``first_price`` / ``second_price``。若给定带
        ``submit(env, emitter_id, true_value)`` 的 ``bid_model``,则覆盖
        ``bid_shades``,依据状态生成提交出价。
        """
        if payment_rule not in {"none", "first_price", "second_price"}:
            raise ValueError(f"Unknown payment_rule: {payment_rule!r}.")
        if commitment_margin < 0.0:
            raise ValueError("commitment_margin must be non-negative.")
        self.config = config or AuctionConfig()
        self.bid_shades = dict(bid_shades) if bid_shades else {}
        self.payment_rule = payment_rule
        self.bid_model = bid_model
        self.commitment_margin = float(commitment_margin)
        self.defend_floor_t = float(defend_floor_t)
        # When set, the defend floor is recomputed from state each decision.
        # 若设置,则每次决策依据状态重新计算防守下限。
        self.adaptive_commitment = adaptive_commitment
        self.last_defend_floor_t = float(defend_floor_t)
        # A rule destination that would overflow within this horizon is never
        # raided: taking its vessel merely moves the shortage (zero-sum).
        # 若规则目的地会在该时域内溢出,则永不被抢:抢走它的船只是搬移短缺(零和)。
        self.protect_horizon_h = float(protect_horizon_h)
        self.last_bids: dict[str, float] = {}
        self.last_winners: dict[str, str] = {}
        self.payments: dict[str, float] = {}

    def __call__(self, env: CCSEnv) -> dict[str, list[int]]:
        """Allow the policy to be used directly as ``policy(env)``.

        允许该策略作为 ``policy(env)`` 直接调用。
        """
        return self.propose_action(env)

    def propose_action(self, env: CCSEnv) -> dict[str, list[int]]:
        """Overlay the auction allocation on the rule baseline action.

        在规则基线动作之上叠加拍卖分配。
        """
        if env.simulator is None:
            raise RuntimeError("Call env.reset() before requesting an action.")
        baseline = rule_action(env)
        action = {
            "vessels": list(baseline["vessels"]),
            "wells": list(baseline["wells"]),
        }
        # Submitted bid = (learned model | shade) x true value; allocation uses
        # the submitted bid. / 提交出价 = (学习模型 | 压价系数) x 真实价值;分配依据之。
        true_values = emitter_bids(env, self.config)
        if self.bid_model is not None:
            submitted = {
                emitter_id: float(self.bid_model.submit(env, emitter_id, value))
                for emitter_id, value in true_values.items()
            }
        else:
            submitted = {
                emitter_id: float(self.bid_shades.get(emitter_id, 1.0)) * value
                for emitter_id, value in true_values.items()
            }
        masks = env.vessel_action_mask()
        empty = _auction_eligible_vessels(env, masks, baseline)
        baseline_emitter = baseline_emitter_map(env, baseline)
        defend_floor_t = self._defend_floor_t(env)
        self.last_defend_floor_t = defend_floor_t
        protected: set[str] = set()
        if self.protect_horizon_h > 0.0:
            from .commitment import hours_to_overflow

            protected = {
                emitter_id
                for emitter_id, hours in hours_to_overflow(env).items()
                if hours < self.protect_horizon_h
            }

        used: set[str] = set()
        winners: dict[str, str] = {}
        bidders: list[str] = []
        # Highest bidder is served first, by the nearest legal empty vessel.
        # 出价最高者优先,由最近的合法空船服务。
        for emitter_id, bid in sorted(submitted.items(), key=lambda kv: -kv[1]):
            if bid <= self.config.reserve_price_eur:
                continue
            bidders.append(emitter_id)
            target_action = env.vessel_go_emitter_action(emitter_id)
            candidates = [
                (position, vessel_id)
                for position, vessel_id in empty
                if vessel_id not in used
                and masks[position][target_action]
                and baseline_emitter.get(position) not in protected
                and may_reassign(
                    emitter_id,
                    baseline_emitter.get(position),
                    submitted,
                    self.commitment_margin,
                    defend_floor_t * self.config.carbon_price_eur_per_t,
                )
            ]
            if not candidates:
                continue
            position, vessel_id = min(
                candidates,
                key=lambda pv: _travel_hours(env, pv[1], emitter_id),
            )
            action["vessels"][position] = target_action
            used.add(vessel_id)
            winners[emitter_id] = vessel_id

        self._settle_payments(submitted, winners, bidders)
        self.last_bids = submitted
        self.last_winners = winners
        return action

    def _defend_floor_t(self, env: CCSEnv) -> float:
        """Return the defend floor, adaptive if configured.

        返回防守下限;若已配置则按状态自适应。
        """
        if self.adaptive_commitment is None:
            return self.defend_floor_t
        from .commitment import adaptive_defend_floor_t

        return adaptive_defend_floor_t(env, self.adaptive_commitment)

    def _settle_payments(
        self,
        submitted: dict[str, float],
        winners: dict[str, str],
        bidders: list[str],
    ) -> None:
        """Charge winners under the configured payment rule (unit demand).

        按配置的支付规则向赢家收费(单位需求)。
        """
        if self.payment_rule == "none" or not winners:
            return
        if self.payment_rule == "second_price":
            losing_bids = [
                submitted[emitter_id]
                for emitter_id in bidders
                if emitter_id not in winners
            ]
            clearing_price = max(losing_bids) if losing_bids else 0.0
        for emitter_id in winners:
            price = (
                submitted[emitter_id]
                if self.payment_rule == "first_price"
                else clearing_price
            )
            self.payments[emitter_id] = self.payments.get(emitter_id, 0.0) + price


def baseline_emitter_map(
    env: CCSEnv,
    baseline: dict[str, list[int]],
) -> dict[int, str | None]:
    """Return the emitter the rule sends each vessel to, by vessel position.

    按船舶位置返回规则为其选择的排放源。
    """
    action_to_emitter = {
        env.vessel_go_emitter_action(emitter_id): emitter_id
        for emitter_id in env.emitter_ids
    }
    return {
        position: action_to_emitter.get(int(baseline["vessels"][position]))
        for position in range(len(env.vessel_ids))
    }


def may_reassign(
    emitter_id: str,
    baseline_emitter: str | None,
    submitted: dict[str, float],
    margin: float,
    defend_floor_eur: float = 0.0,
) -> bool:
    """Return whether the auction may pull a vessel off the rule's choice.

    返回拍卖是否可以把一艘船从规则的选择上拉走。

    With ``margin = 0`` the auction always overrides the rule (purely reactive).
    A positive margin demands the winner outbid the rule's own destination by
    ``(1 + margin)``, so the market keeps the rule's partition -- its implicit
    commitment -- unless deviating is clearly worth it. This prevents greedy
    urgency from starving a source that is never the momentary maximum.

    ``margin = 0`` 时拍卖总是覆盖规则(纯反应式)。正边际要求赢家的出价超过规则目的地
    出价的 ``(1 + margin)`` 倍,因此市场会保持规则的分区(其隐含承诺),除非偏离明显
    值得。这可避免贪心紧迫度饿死那些从不是当下最紧急的源。

    ``defend_floor_eur`` closes the starvation loophole: a myopic bid is zero
    whenever a buffer still has headroom, so a rule destination with no *imminent*
    vent would otherwise be stolen by any positive bid, however large the margin.
    The floor defends it as if it had that much value at risk, representing the
    option value of keeping a source served before it becomes critical.

    ``defend_floor_eur`` 堵住饿死漏洞:缓冲仍有余量时近视出价为 0,若不设下限,规则目的地
    会被任何正出价抢走(无论边际多大)。该下限按"至少有这么多价值处于风险中"来防守它,
    代表"在源变紧急之前持续服务"的期权价值。
    """
    if baseline_emitter is None or baseline_emitter == emitter_id:
        return True
    if margin <= 0.0 and defend_floor_eur <= 0.0:
        return True
    defended = max(submitted.get(baseline_emitter, 0.0), defend_floor_eur)
    return submitted.get(emitter_id, 0.0) > (1.0 + margin) * defended


def _auction_eligible_vessels(
    env: CCSEnv,
    masks: list[list[bool]],
    baseline: dict[str, list[int]],
) -> list[tuple[int, str]]:
    """Return empty vessels the rule is already dispatching to fetch CO2.

    返回规则已经在派往排放源取货的空载船舶。

    Only vessels whose baseline action is a "go to emitter" action are auctioned:
    the rule has already decided to send this empty vessel to fetch CO2, so the
    auction merely re-targets *which* source it serves. Vessels the rule keeps in
    place to keep loading (a WAIT action) are never pulled away, which prevents
    the myopic thrashing of redirecting a vessel mid-loading.

    只有基线动作是"驶向某排放源"的船才进入拍卖:规则已决定派该空船去取货,拍卖只
    改变它服务*哪个*源。规则为继续装载而保留在原地(WAIT)的船永不会被拉走,从而
    避免把正在装载的船中途改道这种近视抖动。
    """
    assert env.simulator is not None
    state = env.simulator.state
    emitter_actions = {
        env.vessel_go_emitter_action(emitter_id)
        for emitter_id in env.emitter_ids
    }
    result: list[tuple[int, str]] = []
    for position, vessel_id in enumerate(env.vessel_ids):
        vessel_state = env.simulator.vessel_states[vessel_id]
        cargo_t = float(state.entity_inventory_t.get(vessel_id, 0.0))
        if (
            vessel_state["mode"] == "berthed"
            and cargo_t <= _EPS
            and int(baseline["vessels"][position]) in emitter_actions
            and any(masks[position][action] for action in emitter_actions)
        ):
            result.append((position, vessel_id))
    return result


def _travel_hours(env: CCSEnv, vessel_id: str, destination_id: str) -> float:
    """Estimate current sailing hours, used to pick the nearest vessel.

    估计当前航行小时数,用于选择最近的船。
    """
    assert env.simulator is not None
    route = env._routes[vessel_id]
    origin_id = env._weather_reference_origin(vessel_id)
    if origin_id == destination_id:
        return 0.0
    leg_id = f"{origin_id}->{destination_id}"
    speed_factor = env._weather_speed_at(leg_id, vessel_id, 0)
    speed_knots = float(
        route.get("speed_knots") or env.config.default_speed_knots
    )
    effective_speed = speed_knots * max(0.0, float(speed_factor))
    if effective_speed <= _EPS:
        return inf
    distance_km = env._leg_distance_km(origin_id, destination_id, route)
    return max(0.0, distance_km / (effective_speed * 1.852))
