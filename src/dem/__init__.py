"""DEM (Dynamic Expectation Maximization) / ADEM implementation in JAX.

This package implements Friston's DEM and ADEM algorithms for active inference
using JAX for automatic differentiation and JIT compilation.

Modules:
    core: Mathematical foundations (D operator, R matrix, generalized coordinates)
    model: Generative model base class and linear model
    inference: D-step (state inference via VFE minimization)
    action: Action update (ADEM action update)
    agent: DEMAgent / ADEMAgent (integration classes)
"""

from src.dem.core import (
    make_D_matrix,
    make_R_matrix,
    make_tilde_precision,
    shift_operator,
)
from src.dem.model import DEMModel, LinearDEMModel
from src.dem.inference import DStep, compute_vfe
from src.dem.action import ActionUpdate
from src.dem.agent import DEMAgent, ADEMAgent

__all__ = [
    "make_D_matrix",
    "make_R_matrix",
    "make_tilde_precision",
    "shift_operator",
    "DEMModel",
    "LinearDEMModel",
    "DStep",
    "compute_vfe",
    "ActionUpdate",
    "DEMAgent",
    "ADEMAgent",
]
