"""GPU (CUDA) implementations of period-finding algorithms.

Re-exports the CUDA-backed Cython classes under a clean namespace:
    from periodfind.gpu import ConditionalEntropy, AOV, LombScargle
"""

from periodfind.ce import ConditionalEntropy
from periodfind.aov import AOV
from periodfind.ls import LombScargle

__all__ = ["ConditionalEntropy", "AOV", "LombScargle"]
