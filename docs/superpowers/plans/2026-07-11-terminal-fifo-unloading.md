# 终端 FIFO 卸载实施计划

> **供智能执行器使用：** 必须使用 `superpowers:executing-plans` 在当前会话逐项执行。每项均采用测试驱动开发（TDD），先观察测试按预期失败，再写最小实现。

**目标：** 将 FIFO 卸载下沉为共享物理层硬约束，使所有控制算法只能卸载每个终端的队首船，同时允许多船停靠、半载离港和返港重新排队。

**架构：** `PhysicalState` 持久化每个终端的卸载队列；`operations/unloading.py` 提供队列同步函数并在货物流转边界强制 FIFO。环境和规则基线复用同一队列生成合法动作，但最终裁决始终由物理层完成。

**技术栈：** Python 3、`dataclasses`、项目现有 `unittest`/`pytest` 测试套件。

## 全局约束

- 每个终端每个时步最多卸载一艘船。
- 多艘船可同时保留在 `vessel_berths` 中。
- 部分卸载的队首船保持队首；半载离港后出队；返回后加入队尾。
- 同时首次观测到达的船只用 vessel ID 打破平局。
- 不改变动作空间、observation 布局、航行、库存、管输、注入或模型参数。
- 不修改用户现有的 `task_plan.md`、`findings.md` 和 `progress.md`。

---

### 任务 1：持久化共享队列并在物理卸载边界强制 FIFO

**文件：**

- 修改：`src/sim/entities/state.py`
- 修改：`src/sim/operations/unloading.py`
- 测试：`tests/test_physical_layer.py`

**接口：**

- 产生：`PhysicalState.terminal_unload_queues: dict[str, list[str]]`
- 产生：`sync_terminal_unload_queue(network, terminal, state, excluded_vessel_ids=None) -> list[str]`
- 消费：现有 `terminal_unload_request_capacity(...)` 和 `project_terminal_unload(...)`

- [ ] **步骤 1：写入会失败的状态复制和 FIFO 物理约束测试**

在 `tests/test_physical_layer.py` 增加真实网络测试，覆盖以下断言：

```python
def test_terminal_fifo_queue_survives_state_copy_and_serialization(self):
    state = PhysicalState(terminal_unload_queues={"terminal": ["ship_b", "ship_a"]})
    copied = state.copy()
    self.assertEqual(copied.terminal_unload_queues, {"terminal": ["ship_b", "ship_a"]})
    self.assertIsNot(copied.terminal_unload_queues["terminal"], state.terminal_unload_queues["terminal"])
    self.assertEqual(state.as_dict()["terminal_unload_queues"], {"terminal": ["ship_b", "ship_a"]})
```

再用两艘船、一个终端、管道和注入井构建网络，按两个时步先后让 `ship_b`、`ship_a` 出现在终端；断言请求 `ship_a` 时流量为零且产生 `fifo_unload_required`，请求 `ship_b` 时只有 `ship_b` 卸载。随后让 `ship_b` 半载离港，断言 `ship_a` 前移；让 `ship_b` 返回，断言队列为 `["ship_a", "ship_b"]`。

- [ ] **步骤 2：运行新增测试并确认按预期失败**

运行：

```powershell
python -m pytest tests/test_physical_layer.py -k "terminal_fifo" -q
```

预期：失败原因是 `PhysicalState` 尚无 `terminal_unload_queues`，或物理层仍允许显式选择非队首船。

- [ ] **步骤 3：为 `PhysicalState` 增加深复制和序列化支持**

在 `PhysicalState` 中增加：

```python
terminal_unload_queues: dict[str, list[str]] = field(default_factory=dict)
```

在 `copy()` 和 `as_dict()` 中均使用：

```python
terminal_unload_queues={
    terminal_id: list(queue)
    for terminal_id, queue in self.terminal_unload_queues.items()
}
```

- [ ] **步骤 4：实现最小共享队列同步和 FIFO 裁决**

在 `operations/unloading.py` 增加公开同步函数。它保留仍停靠且有货的旧成员，过滤 `excluded_vessel_ids`，再把未入队的新成员按 vessel ID 排序后追加：

```python
def sync_terminal_unload_queue(
    network,
    terminal: Terminal,
    state: PhysicalState,
    excluded_vessel_ids: set[str] | None = None,
) -> list[str]:
    excluded = excluded_vessel_ids or set()
    queue = state.terminal_unload_queues.setdefault(terminal.entity_id, [])
    eligible = {
        vessel_id
        for vessel_id in network._entities_of_type(Vessel)
        if vessel_id not in excluded
        and state.vessel_berths.get(vessel_id) == terminal.entity_id
        and state.entity_inventory_t.get(vessel_id, 0.0) > 1e-9
    }
    queue[:] = [vessel_id for vessel_id in queue if vessel_id in eligible]
    queue.extend(sorted(eligible.difference(queue)))
    return queue
```

让 `_terminal_vessels_for_action(...)` 始终从同步后的队首选择：泊位不可用或队列为空时返回空列表；显式目标只有等于队首才返回 `[head]`；无显式目标时返回 `[head]`。在 `project_terminal_unload(...)` 中，显式请求已停靠、有货但不是队首时增加：

```python
Violation(
    "fifo_unload_required",
    str(requested_vessel_id),
    requested_t,
    0.0,
    requested_t,
    "Unload request rejected because another vessel is first in the terminal FIFO queue.",
)
```

该情况不得同时误报 `berth_required`。

- [ ] **步骤 5：运行物理层针对性测试并确认通过**

运行：

```powershell
python -m pytest tests/test_physical_layer.py -k "terminal_fifo or unload" -q
```

预期：新增 FIFO 测试和既有卸载测试全部通过。

- [ ] **步骤 6：提交物理层实现**

```powershell
git add src/sim/entities/state.py src/sim/operations/unloading.py tests/test_physical_layer.py
git commit -m "Enforce terminal FIFO unloading in physical state"
```

### 任务 2：让环境和规则基线生成共享 FIFO 队首动作

**文件：**

- 修改：`src/sim/environment/env.py`
- 修改：`src/sim/control/rule_based.py`
- 修改：`docs/physical_layer_v1_cn.md`
- 测试：`tests/test_env.py`
- 测试：`tests/test_rule_based_actions.py`

**接口：**

- 消费：任务 1 的 `sync_terminal_unload_queue(...)`
- 产生：环境和规则基线始终请求共享队首船；物理层仍负责最终裁决。

- [ ] **步骤 1：写入会失败的控制器 FIFO 测试**

在 `tests/test_env.py` 构造 vessel ID 逆序到达：先让字典序较大的船进入终端并调用 `_terminal_unload_head(...)` 建立队列，再让较小 ID 的船到达；断言两船均停靠时仍返回先到船。把先到船加入 `departing` 后断言下一艘前移，再移除离港状态并重新停靠，断言返回船位于队尾。

在 `tests/test_rule_based_actions.py` 保留现有 FIFO 测试，并增加断言：生成器不再持有私有 `_terminal_unload_queues`，动作目标来自 `PhysicalState.terminal_unload_queues`。

- [ ] **步骤 2：运行控制器测试并确认按预期失败**

运行：

```powershell
python -m pytest tests/test_env.py tests/test_rule_based_actions.py -k "fifo or unload" -q
```

预期：环境仍按 vessel ID 排序，或规则基线仍使用私有队列，因此新增断言失败。

- [ ] **步骤 3：复用共享队列生成合法动作**

在 `env.py` 和 `rule_based.py` 导入 `sync_terminal_unload_queue`。环境的 `_terminal_unload_head(...)` 调用该函数并把 `departing` 作为排除集合，返回队列首项。规则基线删除私有 `_terminal_unload_queues`，在每次生成动作前同步 `PhysicalState` 中各终端队列，并由 `_is_fifo_unload_head(...)` 查询共享队列。

规则基线的 `_requested_unload_supply_t(...)` 只计算一个共享队首船在本时步可卸载的吨数；终端泊位数为零时返回零。

- [ ] **步骤 4：更新中文物理层说明**

在 `docs/physical_layer_v1_cn.md` 的“当前简化”中明确：允许多船停靠；所有控制器共享物理状态中的 FIFO 队列；每个终端每步只卸队首船；半载离港后出队，返回后排队尾；非队首请求产生 `fifo_unload_required`。

- [ ] **步骤 5：运行控制器和物理层测试并确认通过**

运行：

```powershell
python -m pytest tests/test_env.py tests/test_rule_based_actions.py tests/test_physical_layer.py -q
```

预期：全部通过。

- [ ] **步骤 6：运行完整回归测试**

运行：

```powershell
python -m pytest -q
```

预期：完整测试套件通过，无新增警告或错误。

- [ ] **步骤 7：提交控制器集成和文档**

```powershell
git add src/sim/environment/env.py src/sim/control/rule_based.py docs/physical_layer_v1_cn.md tests/test_env.py tests/test_rule_based_actions.py
git commit -m "Use shared FIFO queue for terminal unloading"
```
