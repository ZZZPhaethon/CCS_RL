"""Generate one episode's time-varying disturbance trajectories.

A :class:`Scenario` is the exogenous part of an episode: randomized initial
conditions plus per-hour trajectories for capture availability, weather speed
factors, well maintenance and nominal injectivity factors. It writes the
current step's values into :class:`PhysicalState`; it never chooses actions.

Runtime operations read those values through
``sim.scenario_generation.disturbance_resolver``, which applies the "state
override first, nominal entity fallback second" rule.

生成单个回合的时变扰动轨迹。
:class:`Scenario` 是回合中的外生部分：包含随机初始条件，以及捕集可用率、
天气航速系数、注入井维护和注入能力的逐小时轨迹。它将当前时间步的值写入
:class:`PhysicalState`，但从不选择动作。运行时操作通过
``sim.scenario_generation.disturbance_resolver`` 读取这些值，并遵循“状态覆盖
优先、实体名义值兜底”的规则。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..entities.emitter import Emitter
from ..entities.state import PhysicalState
from ..entities.storage import InjectionWell, Reservoir
from ..entities.terminal import Terminal
from ..entities.vessel import Vessel


@dataclass
class ScenarioConfig:
    """Configure stochastic disturbances and initial-state randomisation.

    默认设置会产生温和但非平凡的随机性。
    配置随机扰动生成器和初始状态随机化。
    """

    episode_hours: int = 168
    time_step_hours: float = 1.0

    # Capture multiplier: Gaussian profile noise plus occasional outages.
    # 捕集系数：高斯曲线噪声叠加偶发停机。
    capture_noise_std: float = 0.30
    capture_outage_rate_per_week: float = 0.5
    capture_outage_mean_hours: float = 12.0
    capture_high_output_rate_per_week: float = 0.5
    capture_high_output_mean_hours: float = 48.0
    capture_high_output_multiplier_range: tuple[float, float] = (1.25, 1.75)

    # Global weather affects vessel speed. ``window`` samples occasional
    # slowdown windows; ``block`` resamples a shared factor at fixed intervals.
    # 全局天气影响船速。``window`` 生成偶发减速窗口；``block`` 每隔固定时间
    # 重新采样一个共享系数。
    weather_process: str = "window"
    weather_window_rate_per_week: float = 0.3
    weather_window_mean_hours: float = 48.0
    weather_window_speed_factor_range: tuple[float, float] = (0.6, 0.8)
    weather_update_hours: float = 24.0
    weather_update_speed_factor_range: tuple[float, float] = (0.75, 1.0)

    # Data-driven leg-wave slowdown stress. A value of 1.0 preserves CSV speed
    # factors; values above 1.0 amplify slowdowns in rough weather.
    # 基于数据的航段波浪减速强度。值为 1.0 时保持 CSV 航速系数不变；大于 1.0
    # 会放大恶劣天气下的减速。
    leg_wave_slowdown_multiplier: float = 1.0
    leg_wave_speed_factor_floor: float = 0.0

    # Injection-well maintenance windows. / 注入井维护窗口。
    well_maintenance_rate_per_week: float = 0.3
    well_maintenance_mean_hours: float = 24.0

    # Initial-condition randomisation as fractions of capacity.
    # 按容量比例进行初始条件随机化。
    randomize_initial_inventory: bool = True
    emitter_initial_fill_range: tuple[float, float] = (0.0, 0.5)
    terminal_initial_fill_range: tuple[float, float] = (0.0, 0.5)

    # Warm-start the slow reservoir-pressure variable; disabled by default.
    # When enabled, short episodes can begin from mid-life reservoir pressures
    # during long-horizon evaluations, such as annual rollouts.
    # 对缓慢变化的储层压力变量进行热启动随机化，默认关闭。启用后，短回合可在
    # 长时域评估（例如年度滚动）中从储层中期压力状态开始。
    warm_start: bool = False
    reservoir_initial_pressure_fill_range: tuple[float, float] = (0.0, 0.5)


@dataclass
class Scenario:
    """Store a sampled episode and its precomputed disturbance trajectories.

    存储一个已采样回合及其预计算的逐时间步扰动轨迹。

    Attributes:
        time_step_hours: Duration of one simulation step, in hours.
            / 单个仿真时间步的时长，单位为小时。
        n_steps: Number of steps in the episode. / 回合中的时间步数量。
        initial_inventory_t: Initial inventory by entity ID, in tonnes.
            / 按实体 ID 记录的初始库存，单位为吨。
        emitter_availability: Capture multipliers by emitter and time step.
            / 按排放源和时间步记录的捕集系数。
        vessel_speed_factor: Sailing-speed multipliers by vessel and step.
            / 按船舶和时间步记录的航速系数。
        leg_speed_factor: Sailing-speed multipliers by route leg and step.
            / 按航段和时间步记录的航速系数。
        well_available: Well availability flags by well and time step.
            / 按注入井和时间步记录的可用性标记。
        injectivity_factor: Injection-capacity multipliers by well and step.
            / 按注入井和时间步记录的注入能力系数。
        seed: Optional random seed used to sample the episode.
            / 生成该回合时使用的可选随机种子。
    """

    time_step_hours: float
    n_steps: int
    initial_inventory_t: dict[str, float] = field(default_factory=dict)
    emitter_availability: dict[str, list[float]] = field(default_factory=dict)
    vessel_speed_factor: dict[str, list[float]] = field(default_factory=dict)
    leg_speed_factor: dict[str, list[float]] = field(default_factory=dict)
    well_available: dict[str, list[bool]] = field(default_factory=dict)
    injectivity_factor: dict[str, list[float]] = field(default_factory=dict)
    seed: int | None = None

    def step_index(self, time_h: float) -> int:
        """Return the step index for a time, clamped to the episode horizon.

        返回给定时间对应的时间步索引，并将其限制在回合时域内。
        """
        index = int(round(time_h / self.time_step_hours))
        return max(0, min(self.n_steps - 1, index))

    def apply_initial(self, state: PhysicalState) -> None:
        """Write this scenario's starting inventories into a physical state.

        将该场景的初始库存写入物理状态。
        """
        for entity_id, inventory_t in self.initial_inventory_t.items():
            state.entity_inventory_t[entity_id] = inventory_t

    def apply_to_state(self, state: PhysicalState, time_h: float) -> None:
        """Write disturbance overrides for the step beginning at ``time_h``.

        将从 ``time_h`` 开始的时间步的扰动覆盖值写入状态。
        """
        i = self.step_index(time_h)
        state.emitter_availability = {
            key: values[i]
            for key, values in self.emitter_availability.items()
        }
        state.vessel_speed_factor = {
            key: values[i]
            for key, values in self.vessel_speed_factor.items()
        }
        state.leg_speed_factor = {
            key: values[i]
            for key, values in self.leg_speed_factor.items()
        }
        state.well_available = {
            key: values[i]
            for key, values in self.well_available.items()
        }
        state.injectivity_factor = {
            key: values[i]
            for key, values in self.injectivity_factor.items()
        }


class ScenarioGenerator:
    """Sample reproducible :class:`Scenario` objects for a network.

    为网络采样可复现的 :class:`Scenario` 对象。
    """

    def __init__(
        self,
        config: ScenarioConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialise the generator with optional defaults and a base seed.

        使用可选的默认配置和基础随机种子初始化生成器。
        """
        self.config = config or ScenarioConfig()
        self.seed = seed

    def sample(self, network, seed: int | None = None) -> Scenario:
        """Sample all disturbance trajectories and initial inventories.

        A supplied seed overrides the generator's base seed for this episode.

        采样全部扰动轨迹和初始库存。
        若提供种子，则其会在本回合中覆盖生成器的基础种子。
        """
        config = self.config
        episode_seed = seed if seed is not None else self.seed
        master = random.Random(episode_seed)
        # Use independent streams so one channel does not shift the others.
        # 使用独立随机流，避免切换某一扰动通道改变其他通道的随机序列。
        capture_rng = random.Random(master.random())
        weather_rng = random.Random(master.random())
        maintenance_rng = random.Random(master.random())
        # Preserve initial-inventory samples for existing fixed seeds.
        # 为已有固定种子保留初始库存的采样结果。
        master.random()
        init_rng = random.Random(master.random())

        dt = config.time_step_hours
        n_steps = max(1, int(round(config.episode_hours / dt)))

        emitters = _ids_of_type(network, Emitter)
        vessels = _ids_of_type(network, Vessel)
        wells = _ids_of_type(network, InjectionWell)
        terminals = network._entities_of_type(Terminal)
        reservoirs = network._entities_of_type(Reservoir)

        emitter_availability = {
            emitter_id: _capture_availability_series(
                capture_rng,
                n_steps,
                dt,
                config,
            )
            for emitter_id in emitters
        }
        weather_speed = _weather_speed_series(weather_rng, n_steps, config)
        vessel_speed_factor = {
            vessel_id: list(weather_speed) for vessel_id in vessels
        }
        well_available = {
            well_id: _availability_from_outage(
                _outage_series(
                    maintenance_rng, n_steps, dt,
                    config.well_maintenance_rate_per_week,
                    config.well_maintenance_mean_hours,
                )
            )
            for well_id in wells
        }
        injectivity_factor = {well_id: [1.0] * n_steps for well_id in wells}
        initial_inventory_t = self._initial_inventory(
            network, init_rng, emitters, terminals, reservoirs
        )

        return Scenario(
            time_step_hours=dt,
            n_steps=n_steps,
            initial_inventory_t=initial_inventory_t,
            emitter_availability=emitter_availability,
            vessel_speed_factor=vessel_speed_factor,
            well_available=well_available,
            injectivity_factor=injectivity_factor,
            seed=episode_seed,
        )

    def _initial_inventory(
        self,
        network,
        rng,
        emitters,
        terminals,
        reservoirs,
    ) -> dict[str, float]:
        """Sample inventory values for enabled initial-state randomisation.

        为已启用的初始状态随机化采样库存值。
        """
        inventory: dict[str, float] = {}
        config = self.config
        if config.randomize_initial_inventory:
            lo_e, hi_e = config.emitter_initial_fill_range
            for emitter_id in emitters:
                emitter = network.entities[emitter_id]
                inventory[emitter_id] = (
                    rng.uniform(lo_e, hi_e) * emitter.buffer_capacity_t
                )
            lo_t, hi_t = config.terminal_initial_fill_range
            for terminal_id, terminal in terminals.items():
                inventory[terminal_id] = (
                    rng.uniform(lo_t, hi_t) * terminal.storage_capacity_t
                )
        if config.warm_start:
            # Pre-fill reservoirs so pressure starts at a mid-life condition.
            # This slow variable is not exposed by a short cold-start episode.
            # 预填充储层，使压力可从中期状态开始；短时冷启动回合通常无法体现这类
            # 缓慢变化的变量。
            lo_r, hi_r = config.reservoir_initial_pressure_fill_range
            for reservoir_id, reservoir in reservoirs.items():
                inventory[reservoir_id] = (
                    rng.uniform(lo_r, hi_r)
                    * reservoir.pressure_limited_capacity_t()
                )
        return inventory


def _ids_of_type(network, entity_type: type) -> list[str]:
    """Return entity IDs whose entities have the requested type.

    返回实体类型匹配时对应的实体 ID。
    """
    return list(network._entities_of_type(entity_type))


def _clamp(value: float, lo: float, hi: float) -> float:
    """Restrict a value to the closed interval ``[lo, hi]``.

    将数值限制在闭区间 ``[lo, hi]`` 内。
    """
    return max(lo, min(hi, value))


def _outage_series(
    rng,
    n_steps: int,
    dt: float,
    rate_per_week: float,
    mean_hours: float,
) -> list[bool]:
    """Generate a boolean outage trajectory using a start/stop process.

    使用开始/结束随机过程生成布尔停机轨迹。
    """
    if rate_per_week <= 0.0 or mean_hours <= 0.0:
        return [False] * n_steps
    start_p = _clamp(rate_per_week * dt / 168.0, 0.0, 1.0)
    end_p = _clamp(dt / mean_hours, 0.0, 1.0)
    series: list[bool] = []
    in_outage = False
    for _ in range(n_steps):
        if in_outage:
            series.append(True)
            if rng.random() < end_p:
                in_outage = False
        elif rng.random() < start_p:
            in_outage = True
            series.append(True)
        else:
            series.append(False)
    return series


def _availability_from_outage(outage: list[bool]) -> list[bool]:
    """Convert outage flags into availability flags.

    将停机标记转换为可用性标记。
    """
    return [not is_out for is_out in outage]


def _capture_availability_series(
    rng,
    n_steps: int,
    dt: float,
    config: ScenarioConfig,
) -> list[float]:
    """Generate capture multipliers with outages, noise, and high-output
    windows.

    生成包含停机、噪声和高产出窗口的捕集系数。
    """
    outage = _outage_series(
        rng,
        n_steps,
        dt,
        config.capture_outage_rate_per_week,
        config.capture_outage_mean_hours,
    )
    series: list[float] = []
    for is_out in outage:
        if is_out:
            series.append(0.0)
            continue
        if config.capture_noise_std > 0.0:
            noisy = rng.gauss(1.0, config.capture_noise_std)
        else:
            noisy = 1.0
        series.append(max(0.0, noisy))
    high_output = _factor_window_series(
        rng,
        n_steps,
        dt,
        config.capture_high_output_rate_per_week,
        config.capture_high_output_mean_hours,
        config.capture_high_output_multiplier_range,
        inactive_value=1.0,
    )
    return [
        availability * factor
        for availability, factor in zip(series, high_output)
    ]


def _weather_speed_series(
    rng,
    n_steps: int,
    config: ScenarioConfig,
) -> list[float]:
    """Generate non-negative vessel-speed multipliers from the weather process.

    根据天气过程生成非负船舶航速系数。
    """
    if config.weather_process == "block":
        return _weather_update_speed_series(rng, n_steps, config)
    if config.weather_process != "window":
        raise ValueError(
            f"Unknown weather_process: {config.weather_process!r}"
        )
    window = _factor_window_series(
        rng,
        n_steps,
        config.time_step_hours,
        config.weather_window_rate_per_week,
        config.weather_window_mean_hours,
        config.weather_window_speed_factor_range,
        inactive_value=1.0,
    )
    return [min(1.0, max(0.0, speed_factor)) for speed_factor in window]


def _weather_update_speed_series(
    rng,
    n_steps: int,
    config: ScenarioConfig,
) -> list[float]:
    """Generate speed factors that remain fixed between weather updates.

    生成在相邻天气更新之间保持不变的航速系数。
    """
    update_steps = max(
        1,
        int(round(config.weather_update_hours / config.time_step_hours)),
    )
    lo, hi = config.weather_update_speed_factor_range
    series: list[float] = []
    for start in range(0, n_steps, update_steps):
        speed_factor = min(1.0, max(0.0, rng.uniform(lo, hi)))
        series.extend([speed_factor] * min(update_steps, n_steps - start))
    return series


def _factor_window_series(
    rng,
    n_steps: int,
    dt: float,
    rate_per_week: float,
    mean_hours: float,
    value_range: tuple[float, float],
    *,
    inactive_value: float,
) -> list[float]:
    """Generate piecewise-constant event windows using a start/stop process.

    使用开始/结束随机过程生成分段恒定的事件窗口。
    """
    if rate_per_week <= 0.0 or mean_hours <= 0.0:
        return [inactive_value] * n_steps
    start_p = _clamp(rate_per_week * dt / 168.0, 0.0, 1.0)
    end_p = _clamp(dt / mean_hours, 0.0, 1.0)
    lo, hi = value_range
    series: list[float] = []
    active = False
    active_value = inactive_value
    for _ in range(n_steps):
        if active:
            series.append(active_value)
            if rng.random() < end_p:
                active = False
        elif rng.random() < start_p:
            active = True
            active_value = rng.uniform(lo, hi)
            series.append(active_value)
        else:
            series.append(inactive_value)
    return series
