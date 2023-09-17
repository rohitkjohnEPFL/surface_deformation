# Import all the necessary modules
import numpy as np
from numba import jit
from yadeGrid.yadeTypes import Vector3D, F64


@jit(nopython=True)  # type: ignore
def crossProduct(a: Vector3D, b: Vector3D) -> Vector3D:
    """Cross product of two vectors"""
    a1, a2, a3 = a
    b1, b2, b3 = b

    return np.array([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])


@jit(nopython=True)  # type: ignore
def dotProduct(a: Vector3D, b: Vector3D) -> F64:
    """Dot product of two vectors"""
    a1, a2, a3 = a
    b1, b2, b3 = b

    return F64(a1 * b1 + a2 * b2 + a3 * b3)


@jit(nopython=True)  # type: ignore
def norm(a: Vector3D) -> F64:
    """Norm of a vector"""
    return F64(np.sqrt(dotProduct(a, a)))
