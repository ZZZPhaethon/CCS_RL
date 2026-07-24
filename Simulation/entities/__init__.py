"""Expose physical entities and state/result records for the simulation.

导出仿真所需的物理实体、状态和结果记录类型。
"""

from .emitter import Emitter
from .manifold import SubseaManifold
from .pipeline import Pipeline
from .state import PhysicalState, StepResult, Violation
from .storage import InjectionWell, Reservoir
from .terminal import Terminal
from .vessel import Vessel

__all__ = [
    "Emitter",
    "InjectionWell",
    "PhysicalState",
    "Pipeline",
    "Reservoir",
    "StepResult",
    "SubseaManifold",
    "Terminal",
    "Vessel",
    "Violation",
]
