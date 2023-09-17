# Import all the necessary modules
import numpy as np
from typing import cast
from numba import jit
import numpy.typing as npt


# Type alias for ndArray for floats
Vector3D = npt.NDArray[np.float64]



@jit  # type: ignore
def crossProduct(a: Vector3D, b: Vector3D) -> Vector3D:
    """Cross product of two vectors"""
    a1, a2, a3 = a
    b1, b2, b3 = b

    return np.array([[a2 * b3 - a3 * b2], [a3 * b1 - a1 * b3], [a1 * b2 - a2 * b1]])


@jit  # type: ignore
def dotProduct(a: Vector3D, b: Vector3D) -> np.float64:
    """Dot product of two vectors"""
    a1, a2, a3 = a
    b1, b2, b3 = b

    return cast(np.float64, a1 * b1 + a2 * b2 + a3 * b3)


@jit  # type: ignore
def norm(a: Vector3D) -> np.float64:
    """Norm of a vector"""
    return cast(np.float64, np.sqrt(dotProduct(a, a)))
