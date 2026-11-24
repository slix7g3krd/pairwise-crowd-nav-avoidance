"""确定性的群体寻路避让工具集。

本模块针对群体寻路中常见的六类问题给出可组合的纯函数实现：

1. 避让阻尼：密集人群中按邻居密度衰减避让力，并用指数平滑抑制来回振荡；
2. 导航版本号：导航网格脏块内仍有移动 agent 时延迟提交增量更新；
3. 确定性绕行：绕行侧由稳定哈希决定，不依赖 ``PYTHONHASHSEED``；
4. 门容量与排队：门只允许容量内的通行量，等待队列按 FIFO + 稳定次序；
5. 重寻路节流：每个 agent 有确定性错峰的冷却时间，掉队 agent 保序重入；
6. 群体分层：按优先级确定性地把 agent 分到多个层，限制同帧参与避让的数量。

模块内没有全局可变状态，相同输入在任意进程、任意时刻都会得到相同输出。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "avoid_damping",
    "nav_version",
    "deterministic_side",
    "gate_capacity",
    "repath_throttle",
    "layer_group",
    "queue_order",
    "validation",
    "audit",
    "fallback",
    "density_grid",
    "metrics",
    "lost_agent",
    "local_avoid",
    "determinism",
]

_EPS = 1e-9
_DEFAULT_LAYERS = 4
_MAX_LAYERS = 32


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    if low > high:
        low, high = high, low
    return low if value < low else high if value > high else value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_int(value: Any, default: int = 0) -> int:
    number = _as_float(value, float("nan"))
    if not math.isfinite(number):
        return default
    return int(number)


def _count(value: Any) -> int:
    """把 ``None`` / 数字 / 可迭代对象统一成非负计数。"""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if _is_number(value):
        return max(0, int(value))
    if isinstance(value, (str, bytes)):
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _stable_uint(*parts: Any) -> int:
    """跨进程稳定的 64 位无符号哈希（blake2b，不使用内置 ``hash``）。"""
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(b"\x1f")
        digest.update(str(part).encode("utf-8", "backslashreplace"))
    return int.from_bytes(digest.digest(), "big")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _agent_value(agent: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = _get(agent, key, None)
        if value is not None:
            return value
    return default


def _agent_id(agent: Any, default: Any = 0) -> Any:
    if agent is None:
        return default
    if isinstance(agent, Mapping):
        for key in ("id", "agent_id", "index"):
            if key in agent:
                return agent[key]
        return default
    if isinstance(agent, (str, bytes)):
        return agent.decode() if isinstance(agent, bytes) else agent
    for key in ("id", "agent_id", "index"):
        if hasattr(agent, key):
            return getattr(agent, key)
    if _is_number(agent):
        return agent
    return default


def _point(value: Any, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, Mapping):
        if "x" in value or "y" in value:
            return (
                _as_float(value.get("x"), default[0]),
                _as_float(value.get("y"), default[1]),
            )
        return default
    if hasattr(value, "x") and hasattr(value, "y"):
        return (_as_float(value.x, default[0]), _as_float(value.y, default[1]))
    try:
        x, y = value
    except (TypeError, ValueError):
        return default
    return (_as_float(x, default[0]), _as_float(y, default[1]))


def _agent_pos(agent: Any) -> tuple[float, float]:
    if isinstance(agent, Mapping):
        for key in ("pos", "position"):
            if key in agent:
                return _point(agent[key])
        return _point(agent)
    for key in ("pos", "position"):
        if hasattr(agent, key):
            return _point(getattr(agent, key))
    return _point(agent)


def _agent_vel(agent: Any) -> tuple[float, float]:
    if isinstance(agent, Mapping):
        for key in ("vel", "velocity"):
            if key in agent:
                return _point(agent[key])
        return (0.0, 0.0)
    for key in ("vel", "velocity"):
        if hasattr(agent, key):
            return _point(getattr(agent, key))
    return (0.0, 0.0)


def _priority(agent: Any) -> float:
    return _as_float(_agent_value(agent, "priority", "rank", default=0.0), 0.0)


def _as_agents(value: Any) -> list[Any]:
    """把列表、``id -> agent`` 映射或单个 agent 统一成列表。"""
    if value is None or _is_number(value) or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Mapping):
        agents: list[Any] = []
        for agent_id, record in value.items():
            if isinstance(record, Mapping):
                materialized = dict(record)
                materialized.setdefault("id", agent_id)
                agents.append(materialized)
            else:
                agents.append(record)
        return agents
    if isinstance(value, Sequence):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _as_events(value: Any) -> list[Any]:
    if value is None or _is_number(value) or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _density_value(value: Any) -> float:
    """把密度输入（数字、邻居列表、带 ``density`` 字段的字典）变成标量。"""
    if value is None:
        return 0.0
    if isinstance(value, Mapping):
        if "density" in value:
            return max(0.0, _as_float(value["density"], 0.0))
        for key in ("crowd_density", "neighbors", "nearby"):
            if key in value:
                return float(_count(value[key]))
        return 0.0
    if _is_number(value):
        return max(0.0, float(value))
    if hasattr(value, "neighbors"):
        return float(_count(getattr(value, "neighbors")))
    return float(_count(value))


# ---------------------------------------------------------------------------
# 1. 避让阻尼与局部避让
# ---------------------------------------------------------------------------


def avoid_damping(
    density: Any = 1.0,
    *,
    falloff: float = 0.55,
    min_factor: float = 0.08,
    max_factor: float = 1.0,
    previous: float | None = None,
    smoothing: float = 0.35,
) -> float:
    """根据邻居密度计算避让力阻尼系数，返回 ``[min_factor, max_factor]``。

    密度越高，阻尼越小，避免密集人群中避让力互相叠加造成抖动。
    传入 ``previous`` 时对结果做一步指数平滑（``smoothing`` 越大越接近
    当前原始值），用于抑制相邻两帧之间方向相反的力。
    """
    crowd = _density_value(density)
    raw = 1.0 / (1.0 + max(0.0, _as_float(falloff, 0.0)) * crowd)
    if previous is not None:
        previous_value = _clamp(_as_float(previous, 1.0), min_factor, max_factor)
        alpha = _clamp(_as_float(smoothing, 0.0), 0.0, 1.0)
        raw = previous_value + alpha * (raw - previous_value)
    return _clamp(raw, _as_float(min_factor, 0.0), _as_float(max_factor, 1.0))


def local_avoid(
    agent: Any = None,
    neighbors: Any = None,
    *,
    seed: int = 0,
    radius: float = 2.0,
    strength: float = 1.0,
    previous_damping: float | None = None,
    side_bias: float = 0.35,
) -> dict[str, Any]:
    """计算单个 agent 的局部避让力。

    邻居越近推力越大；总力乘以 :func:`avoid_damping` 的密度阻尼，再叠加
    一个由 :func:`deterministic_side` 决定的侧向分量，保证同一场景的
    绕行方向稳定。返回力向量、阻尼、绕行侧和邻居数。
    """
    agent_id = _agent_id(agent)
    position = _agent_pos(agent)
    radius = max(_EPS, _as_float(radius, 2.0))
    strength = max(0.0, _as_float(strength, 1.0))
    side_bias = _clamp(_as_float(side_bias, 0.0), 0.0, 1.0)

    force_x = 0.0
    force_y = 0.0
    considered = 0
    for neighbor in _as_agents(neighbors):
        other_x, other_y = _agent_pos(neighbor)
        dx = position[0] - other_x
        dy = position[1] - other_y
        distance = math.hypot(dx, dy)
        if distance >= radius:
            continue
        considered += 1
        if distance <= _EPS:
            # 完全重叠时用稳定侧向方向分开，避免除以零和随机抖动。
            side = deterministic_side(agent_id, seed=seed, gate_id=_get(agent, "gate_id"))
            force_x += side * 1.0
            continue
        weight = (radius - distance) / radius
        force_x += dx / distance * weight
        force_y += dy / distance * weight

    damping = avoid_damping(considered, previous=previous_damping)
    magnitude = math.hypot(force_x, force_y)
    if magnitude > _EPS:
        direction_x = force_x / magnitude
        direction_y = force_y / magnitude
    else:
        direction_x, direction_y = 1.0, 0.0

    side = deterministic_side(agent_id, seed=seed, gate_id=_get(agent, "gate_id"))
    # 法向分量与推力正交，用来稳定地从障碍/人群侧面绕行。
    final_x = direction_x * damping + (-direction_y) * side * side_bias
    final_y = direction_y * damping + direction_x * side * side_bias
    final_magnitude = math.hypot(final_x, final_y)
    if final_magnitude > strength > _EPS:
        scale = strength / final_magnitude
        final_x *= scale
        final_y *= scale

    return {
        "force": (final_x, final_y),
        "damping": damping,
        "side": side,
        "neighbors": considered,
    }


# ---------------------------------------------------------------------------
# 2. 导航网格版本号
# ---------------------------------------------------------------------------


def nav_version(
    state: Any = None,
    *,
    current: int | None = None,
    dirty: Any = None,
    moving_in_dirty: Any = None,
    commit: bool = False,
) -> int:
    """返回导航网格版本号。

    ``commit=True`` 表示请求提交一批增量更新。只有当存在脏块且脏块内
    没有正在移动的 agent 时版本号才 ``+1``；否则保持旧版本，让移动中的
    agent 继续使用一致的导航快照。
    """
    if state is not None and not _is_number(state) and not isinstance(state, (str, bytes)):
        if isinstance(state, Mapping) or hasattr(state, "version"):
            current = _get(state, "version", current)
            dirty = _get(state, "dirty", _get(state, "dirty_tiles", dirty))
            moving_in_dirty = _get(
                state, "moving_in_dirty", _get(state, "agents_in_dirty", moving_in_dirty)
            )
            commit = bool(_get(state, "commit", commit))
    elif _is_number(state) and current is None:
        current = state

    version = max(0, _as_int(current, 0))
    dirty_count = _count(dirty)
    moving_count = _count(moving_in_dirty)
    if commit and dirty_count > 0 and moving_count == 0:
        return version + 1
    return version


# ---------------------------------------------------------------------------
# 3. 确定性绕行
# ---------------------------------------------------------------------------


def deterministic_side(
    agent_id: Any = 0,
    *,
    seed: int = 0,
    gate_id: Any = None,
) -> int:
    """返回 ``-1`` 或 ``+1``，表示固定不变的绕行侧。

    侧别由 ``(seed, gate_id, agent_id)`` 的稳定哈希决定，同一 agent 在
    同一门/种子下永远选择同一侧；不同 agent 会被分散到两侧，避免所有人
    同时往一边挤。
    """
    if isinstance(agent_id, Mapping) or (
        not isinstance(agent_id, (int, float, str, bytes)) and hasattr(agent_id, "id")
    ):
        agent = agent_id
        agent_id = _agent_id(agent)
        gate_id = _agent_value(agent, "gate_id", "gate", default=gate_id)
        seed = _as_int(_agent_value(agent, "seed", default=seed), 0)

    bits = _stable_uint("side", seed, gate_id, agent_id)
    return -1 if (bits & 1) else 1


# ---------------------------------------------------------------------------
# 4. 门容量与排队
# ---------------------------------------------------------------------------


def gate_capacity(
    gate: Any = None,
    *,
    capacity: int | None = None,
    in_transit: int = 0,
) -> int:
    """返回门当前还可通行的数量（容量减去正在通行的数量，最小为 0）。

    ``gate`` 可以是容量数字，也可以是带 ``capacity`` / ``in_transit``
    （或 ``occupancy`` / ``crossing``）字段的字典/对象。
    """
    if gate is not None and not _is_number(gate):
        gate_cap = _get(gate, "capacity", None)
        if gate_cap is not None:
            capacity = gate_cap
        in_transit = _agent_value(
            gate, "in_transit", "occupancy", "crossing", default=in_transit
        )
    elif capacity is None:
        capacity = gate

    total = max(0, _as_int(capacity, 0))
    used = max(0, _as_int(in_transit, 0))
    return max(0, total - used)


def queue_order(
    agents: Any = None,
    *,
    capacity: int | None = None,
) -> list[Any]:
    """返回门的排队顺序（agent id 列表）。

    严格 FIFO：先按 ``queued_at`` / ``entered_at`` 排序；同一时刻入队的
    用稳定的 ``sequence`` 或原始下标决定次序，绝不使用随机或字典顺序。
    ``capacity`` 给定时只返回前 ``capacity`` 个。
    """
    decorated: list[tuple[float, float, str, int, Any]] = []
    for index, agent in enumerate(_as_agents(agents)):
        entered = _as_float(
            _agent_value(agent, "queued_at", "queue_time", "entered_at", default=0.0),
            0.0,
        )
        sequence = _as_float(
            _agent_value(agent, "sequence", "seq", default=float(index)), float(index)
        )
        agent_id = _agent_id(agent, index)
        decorated.append((entered, sequence, str(agent_id), index, agent_id))

    decorated.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    ordered = [item[4] for item in decorated]
    if capacity is not None:
        ordered = ordered[: max(0, _as_int(capacity, 0))]
    return ordered


# ---------------------------------------------------------------------------
# 5. 重寻路节流与掉队 agent
# ---------------------------------------------------------------------------


def repath_throttle(
    agent: Any = None,
    *,
    now: float = 0.0,
    last_repath: float | None = None,
    interval: float = 0.75,
    jitter: float = 0.25,
    seed: int = 0,
) -> bool:
    """判断 agent 现在是否允许重新寻路。

    冷却时间为 ``interval`` 再加一段由稳定哈希决定、范围在
    ``[0, jitter * interval]`` 内的错峰偏移，避免所有 agent 同一帧重寻路。
    ``last_repath`` 为 ``None`` 表示从未寻路，直接允许。
    """
    agent_id = _agent_id(agent)
    if isinstance(agent, Mapping) or hasattr(agent, "id"):
        now = _agent_value(agent, "now", default=now)
        last_repath = _agent_value(
            agent, "last_repath", "last_repath_at", default=last_repath
        )
        interval = _agent_value(agent, "interval", "repath_interval", default=interval)
        seed = _as_int(_agent_value(agent, "seed", default=seed), 0)

    if last_repath is None:
        return True

    interval = max(0.0, _as_float(interval, 0.0))
    phase = (_stable_uint("repath", seed, agent_id) % 1000) / 1000.0
    offset = phase * max(0.0, _as_float(jitter, 0.0)) * interval
    next_allowed = _as_float(last_repath, 0.0) + interval + offset
    return _as_float(now, 0.0) + _EPS >= next_allowed


def lost_agent(
    agent: Any = None,
    *,
    queue: Any = None,
    now: float = 0.0,
    interval: float = 0.75,
    seed: int = 0,
) -> dict[str, Any]:
    """为掉队 agent 生成重寻路计划，保证不把原有队列打散。

    若 agent 仍在等待队列中，保留其原始槽位（只重算路径，不重新入队）；
    重寻路本身受 :func:`repath_throttle` 节流，冷却未结束时先原地等待。
    """
    agent_id = _agent_id(agent)
    waiting = _as_agents(queue)
    slot: int | None = None
    for index, queued in enumerate(waiting):
        if str(_agent_id(queued, index)) == str(agent_id):
            slot = index
            break

    if isinstance(agent, Mapping):
        now = _agent_value(agent, "now", default=now)
        interval = _agent_value(agent, "interval", "repath_interval", default=interval)

    allowed = repath_throttle(agent, now=now, interval=interval, seed=seed)
    side = deterministic_side(
        agent_id,
        seed=seed,
        gate_id=_agent_value(agent, "gate_id", "gate", default=None),
    )

    if not allowed:
        action = "hold"
    elif slot is not None:
        action = "rejoin"
    else:
        action = "repath"

    return {
        "agent": agent_id,
        "action": action,
        "repath": allowed,
        "keep_queue_slot": slot is not None,
        "queue_slot": slot,
        "side": side,
    }


# ---------------------------------------------------------------------------
# 6. 群体分层
# ---------------------------------------------------------------------------


def layer_group(
    agents: Any = None,
    *,
    layers: int | None = None,
    max_layers: int = _MAX_LAYERS,
) -> dict[str, Any]:
    """把 agent 确定性分层，返回层号和层内成员。

    - agent 自带 ``layer`` 字段时按该字段分组；
    - 否则按优先级（高优先在前）排序后轮转分配到 ``layers`` 个层，
      使各层人数尽量均衡，同优先级用 id 稳定排序。

    调用方可以每帧只更新一个层，降低密集人群中的同时计算量。
    """
    if layers is None and _is_number(agents):
        layers = int(agents)
        agents = None

    items = _as_agents(agents)
    layer_count = max(
        1,
        min(
            max(1, _as_int(max_layers, _MAX_LAYERS)),
            _as_int(layers, _DEFAULT_LAYERS),
        ),
    )

    buckets: dict[int, list[Any]] = {}
    explicit_layers = [_agent_value(agent, "layer", default=None) for agent in items]
    use_explicit = bool(items) and all(value is not None for value in explicit_layers)

    if use_explicit:
        for agent, value in zip(items, explicit_layers):
            layer = max(0, _as_int(value, 0))
            buckets.setdefault(layer, []).append(_agent_id(agent))
    else:
        ranked = sorted(
            range(len(items)),
            key=lambda index: (
                -_priority(items[index]),
                str(_agent_id(items[index], index)),
                index,
            ),
        )
        for position, index in enumerate(ranked):
            buckets.setdefault(position % layer_count, []).append(
                _agent_id(items[index], index)
            )

    for layer in range(layer_count):
        buckets.setdefault(layer, [])

    layers_dict = {layer: list(buckets[layer]) for layer in sorted(buckets)}
    by_agent: dict[Any, int] = {}
    for layer, members in layers_dict.items():
        for agent_id in members:
            by_agent[agent_id] = layer

    return {
        "layer_count": len(layers_dict),
        "layers": layers_dict,
        "by_agent": by_agent,
    }


# ---------------------------------------------------------------------------
# 校验、审计、回退、密度网格、指标
# ---------------------------------------------------------------------------


def validation(scene: Any = None) -> dict[str, Any]:
    """校验场景配置，返回 ``{"ok": bool, "issues": [...], ...}``。

    检查重复 agent id、非法半径/坐标、门容量与等待人数、导航版本号、
    层数等约束。空场景或非映射输入视为合法空场景。
    """
    if not isinstance(scene, Mapping):
        scene = {}

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    agents = _as_agents(scene.get("agents"))
    seen_ids: dict[str, int] = {}

    for index, agent in enumerate(agents):
        agent_id = _agent_id(agent, index)
        key = str(agent_id)
        seen_ids[key] = seen_ids.get(key, 0) + 1

        radius = _agent_value(agent, "radius", default=None)
        if radius is not None and _as_float(radius, 0.0) <= 0.0:
            issues.append({"code": "non_positive_radius", "agent": agent_id})

        pos_x, pos_y = _agent_pos(agent)
        vel_x, vel_y = _agent_vel(agent)
        if not all(math.isfinite(v) for v in (pos_x, pos_y, vel_x, vel_y)):
            issues.append({"code": "invalid_agent_state", "agent": agent_id})

    for key, count in sorted(seen_ids.items()):
        if count > 1:
            issues.append(
                {"code": "duplicate_agent_id", "agent": key, "count": count}
            )

    gates = scene.get("gates", {})
    if isinstance(gates, Mapping):
        gate_items = list(gates.items())
    elif isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)):
        gate_items = list(enumerate(gates))
    else:
        gate_items = []

    for gate_id, gate in gate_items:
        capacity = gate_capacity(gate)
        if capacity <= 0:
            issues.append(
                {"code": "non_positive_gate_capacity", "gate": str(gate_id)}
            )
        waiting = _count(_agent_value(gate, "queue", "waiting", "pending", default=()))
        if waiting > capacity:
            issues.append(
                {
                    "code": "gate_queue_over_capacity",
                    "gate": str(gate_id),
                    "waiting": waiting,
                    "capacity": capacity,
                }
            )
        elif waiting == capacity and capacity > 0:
            warnings.append(
                {"code": "gate_queue_full", "gate": str(gate_id), "capacity": capacity}
            )

    version = scene.get("nav_version", scene.get("version", 0))
    if not _is_number(version) or _as_int(version, -1) < 0:
        issues.append({"code": "invalid_nav_version"})

    layer_count = scene.get("layers", scene.get("layer_count", _DEFAULT_LAYERS))
    if _as_int(layer_count, 0) < 1:
        issues.append({"code": "invalid_layer_count"})

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "counts": {
            "agents": len(agents),
            "gates": len(gate_items),
            "issues": len(issues),
            "warnings": len(warnings),
        },
    }


def audit(events: Any = None) -> dict[str, Any]:
    """审计事件流，检查门超载、重寻路过密、队列槽位冲突和脏块提交。

    返回 ``{"ok": bool, "violations": [...], "counts": {...}}``。
    非序列输入视为空事件流。
    """
    items = _as_events(events)
    counts: dict[str, int] = {}
    violations: list[dict[str, Any]] = []
    occupied_slots: dict[tuple[str, str], Any] = {}

    for index, event in enumerate(items):
        kind = str(_agent_value(event, "type", "event", "kind", default=""))
        counts[kind] = counts.get(kind, 0) + 1

        if kind in {"gate_admit", "gate_pass", "admit"}:
            capacity = _agent_value(event, "capacity", default=None)
            admitted = max(
                0, _as_int(_agent_value(event, "admitted", "count", default=1), 1)
            )
            if capacity is not None and admitted > _as_int(capacity, 0):
                violations.append(
                    {
                        "code": "gate_over_capacity",
                        "event_index": index,
                        "admitted": admitted,
                        "capacity": _as_int(capacity, 0),
                    }
                )
        elif kind in {"repath", "repath_request"}:
            last_repath = _agent_value(event, "last_repath", default=None)
            now = _as_float(_agent_value(event, "now", "time", default=0.0), 0.0)
            interval = _as_float(_agent_value(event, "interval", default=0.75), 0.75)
            if last_repath is not None and now - _as_float(last_repath, 0.0) < interval - _EPS:
                violations.append(
                    {
                        "code": "repath_too_soon",
                        "event_index": index,
                        "agent": _agent_value(event, "agent", "agent_id", default=None),
                    }
                )
        elif kind in {"queue_assign", "queue"}:
            gate = str(_agent_value(event, "gate", "gate_id", default=""))
            slot = _agent_value(event, "slot", "position", "index", default=None)
            agent_id = _agent_value(event, "agent", "agent_id", "id", default=None)
            if slot is not None:
                key = (gate, str(slot))
                if key in occupied_slots and str(occupied_slots[key]) != str(agent_id):
                    violations.append(
                        {
                            "code": "queue_slot_conflict",
                            "event_index": index,
                            "gate": gate,
                            "slot": slot,
                        }
                    )
                else:
                    occupied_slots[key] = agent_id
        elif kind in {"nav_commit", "navmesh_commit"}:
            moving = _count(
                _agent_value(event, "moving_in_dirty", "moving_agents", default=0)
            )
            if moving > 0:
                violations.append(
                    {
                        "code": "nav_commit_with_moving_agents",
                        "event_index": index,
                        "moving": moving,
                    }
                )

    return {
        "ok": not violations,
        "violations": violations,
        "counts": dict(sorted(counts.items())),
        "event_count": len(items),
    }


def fallback(state: Any = None) -> dict[str, Any]:
    """导航失效或路径过期时的安全回退。

    导航版本不一致时先原地保持，不清空队列槽位；掉队 agent 走保序重入。
    返回动作、速度、是否允许重寻路和保留队列槽位标记。
    """
    if not isinstance(state, Mapping):
        state = {}

    path_version = state.get("path_version")
    current_version = state.get("nav_version")
    stale = state.get("nav_stale")
    if stale is None and path_version is not None and current_version is not None:
        stale = path_version != current_version
    stale = bool(stale)
    lost = bool(state.get("lost", False))
    keep_slot = state.get("queue_slot") is not None or bool(state.get("queued", False))

    allowed = state.get("repath_allowed")
    if allowed is None:
        allowed = repath_throttle(
            state,
            now=_as_float(state.get("now", 0.0), 0.0),
            last_repath=state.get("last_repath"),
            interval=_as_float(state.get("interval", 0.75), 0.75),
            seed=_as_int(state.get("seed", 0), 0),
        )
    allowed = bool(allowed)

    if lost and stale:
        action = "hold_rejoin"
        reason = "navmesh_stale_and_agent_lost"
    elif lost:
        action = "rejoin"
        reason = "agent_lost"
    elif stale:
        action = "hold"
        reason = "navmesh_stale"
    else:
        action = "continue"
        reason = "path_current"

    if action in {"hold", "hold_rejoin"}:
        velocity = (0.0, 0.0)
    else:
        velocity = _point(state.get("last_valid_velocity", state.get("velocity", (0.0, 0.0))))

    return {
        "action": action,
        "velocity": velocity,
        "repath": allowed,
        "keep_queue_slot": keep_slot,
        "reason": reason,
    }


def density_grid(points: Any = None, *, cell_size: float = 1.0) -> dict[str, Any]:
    """构建空间哈希网格，用于快速统计局部密度。

    ``points`` 可以是坐标序列或带 ``pos`` 的 agent 列表；数字输入会被
    当作 ``cell_size``。返回每个单元格内的点下标与计数。
    """
    if _is_number(points):
        cell_size = float(points)
        points = None

    size = _as_float(cell_size, 1.0)
    if size <= 0.0:
        size = 1.0

    cells: dict[str, list[int]] = {}
    for index, point in enumerate(_as_agents(points)):
        x, y = _agent_pos(point)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        key = f"{math.floor(x / size)},{math.floor(y / size)}"
        cells.setdefault(key, []).append(index)

    ordered = {
        key: cells[key]
        for key in sorted(
            cells, key=lambda text: tuple(int(part) for part in text.split(","))
        )
    }
    counts = {key: len(members) for key, members in ordered.items()}
    return {
        "cell_size": size,
        "cells": ordered,
        "counts": counts,
        "cell_count": len(ordered),
        "point_count": sum(counts.values()),
    }


def metrics(agents: Any = None, *, events: Any = None, dt: float = 1.0) -> dict[str, Any]:
    """汇总群体指标：速度、阻尼、振荡频率、门吞吐与重寻路次数。"""
    items = _as_agents(agents)
    speeds: list[float] = []
    dampings: list[float] = []
    reversals = 0
    repaths = 0

    for agent in items:
        velocity_x, velocity_y = _agent_vel(agent)
        speeds.append(math.hypot(velocity_x, velocity_y))
        damping = _agent_value(agent, "damping", "damping_factor", default=None)
        if damping is not None:
            dampings.append(_clamp(_as_float(damping, 1.0), 0.0, 1.0))
        reversals += max(
            0,
            _as_int(
                _agent_value(agent, "reversals", "direction_changes", default=0), 0
            ),
        )

    throughput = 0
    max_queue = sum(
        1 for agent in items if bool(_agent_value(agent, "queued", default=False))
    )
    for event in _as_events(events):
        kind = str(_agent_value(event, "type", "event", "kind", default=""))
        if kind in {"gate_admit", "gate_pass", "admit"}:
            throughput += max(
                0, _as_int(_agent_value(event, "admitted", "count", default=1), 1)
            )
        elif kind in {"queue", "queue_length", "gate_queue"}:
            length = _agent_value(event, "length", "size", "waiting", "queue_len", default=None)
            if length is None:
                length = _count(_agent_value(event, "queue", default=()))
            max_queue = max(max_queue, _as_int(length, 0))
        elif kind in {"reversal", "direction_change"}:
            reversals += 1
        elif kind in {"repath", "repath_request"}:
            repaths += 1

    delta = max(_EPS, _as_float(dt, 1.0))
    agent_count = len(items)
    return {
        "agents": agent_count,
        "mean_speed": sum(speeds) / len(speeds) if speeds else 0.0,
        "max_speed": max(speeds) if speeds else 0.0,
        "mean_damping": sum(dampings) / len(dampings) if dampings else 1.0,
        "reversals": reversals,
        "oscillation_rate": reversals / max(_EPS, delta * max(1, agent_count)),
        "gate_throughput": throughput,
        "max_queue": max_queue,
        "repaths": repaths,
        "dt": delta,
    }


def determinism(
    scene: Any = None,
    *,
    runs: int = 2,
    seed: int = 0,
) -> bool:
    """重复规划同一场景，验证排队、绕行侧、分层与密度网格完全一致。"""
    scene = scene if isinstance(scene, Mapping) else {}
    agents = scene.get("agents", ())
    gate_id = scene.get("gate_id", scene.get("gate"))
    cell_size = _as_float(scene.get("cell_size", 1.0), 1.0)

    def plan() -> tuple[Any, ...]:
        items = _as_agents(agents)
        ordered = tuple(queue_order(items, capacity=scene.get("queue_capacity")))
        grouping = layer_group(items, layers=scene.get("layers"))
        sides = tuple(
            sorted(
                (
                    str(_agent_id(agent, index)),
                    deterministic_side(
                        _agent_id(agent, index), seed=seed, gate_id=gate_id
                    ),
                )
                for index, agent in enumerate(items)
            )
        )
        grid = density_grid(items, cell_size=cell_size)
        return (ordered, grouping["layers"], grouping["by_agent"], sides, grid["cells"])

    expected = plan()
    for _ in range(max(1, _as_int(runs, 2)) - 1):
        if plan() != expected:
            return False
    return True
