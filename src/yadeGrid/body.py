# Import all the necessary modules
import numpy as np
from attrs import define, field
from numba import jit
from yadeGrid.yadeTypes import Vector3D, F64
from numpy.typing import NDArray
from typing import Any


# ------------------------------------------------------------------------------------------------ #
#                                                                                       QUATERNION #
# ------------------------------------------------------------------------------------------------ #
@define
class Quaternion:
    components: NDArray[np.float64] = field(default=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    # This way we can pass the quaternion as ndarray to JITed functions

    @property
    def a(self) -> np.float64:
        return np.float64(self.components[0])

    @property
    def b(self) -> np.float64:
        return np.float64(self.components[1])

    @property
    def c(self) -> np.float64:
        return np.float64(self.components[2])

    @property
    def d(self) -> np.float64:
        return np.float64(self.components[3])

    def __eq__(self, other: Any) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return np.array_equal(self.components, other.components)

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        if isinstance(other, Quaternion):
            result = multiply_quat(self.components, other.components)
            return Quaternion(components=result)
        else:
            raise TypeError(f"Cannot multiply Quaternion with {type(other)}")

    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.components * np.array([1, -1, -1, -1]))

    def norm(self) -> np.float64:
        return F64(norm_quat(self.components))

    def normalize(self) -> 'Quaternion':
        return self / self.norm()

    @jit(nopython=True)  # type: ignore
    def conv_2axisAngle(self) -> 'AxisAngle':
        angle = 2 * np.arccos(self.a)
        axis = np.array([self.b, self.c, self.d]) / np.sqrt(1 - self.a**2)
        return AxisAngle(axis=axis, angle=angle)

    def __truediv__(self, scalar: np.float64) -> 'Quaternion':
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Quaternion by zero")

        return Quaternion(np.array([self.a, self.b, self.c, self.d]) / scalar)


# ---------------------------------------------------------------------------------- multiply_quat #
# JIT functions are given ndarray as input and output. The methods in quaternion functions
# handle the task of constructing the quaternion from returned the ndarray
@jit(nopython=True)  # type: ignore
def multiply_quat(q1, q2):
    result = np.zeros(4, dtype=np.float64)
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    result[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    result[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

    return result


# -------------------------------------------------------------------------------------- norm_quat #
@jit(nopython=True)  # type: ignore
def norm_quat(q1):
    return np.sqrt(q1[0]**2 + q1[1]**2 + q1[2]**2 + q1[3]**2)


# ------------------------------------------------------------------------------------------------ #
#                                                                                        AXISANGLE #
# ------------------------------------------------------------------------------------------------ #
@define
class AxisAngle:
    axis: Vector3D = field(default=np.array([1, 0, 0]))
    angle: float = field(default=0.0)

    def __repr__(self) -> str:
        return f"AxisAngle(axis={self.axis}, angle={self.angle})"

    def __str__(self) -> str:
        return f"({self.angle}, {self.axis},)"

    def __attrs_post_init__(self) -> None:
        self.axis = self.axis / np.linalg.norm(self.axis)

    def conv_2quaternion(self) -> Quaternion:
        return Quaternion(np.array([
            np.cos(self.angle / 2),
            self.axis[0] * np.sin(self.angle / 2),
            self.axis[1] * np.sin(self.angle / 2),
            self.axis[2] * np.sin(self.angle / 2)
        ]))


# ------------------------------------------------------------------------------------------------ #
#                                                                                             BODY #
# ------------------------------------------------------------------------------------------------ #
@define(kw_only=True)
class Body:
    # State variables
    pos: Vector3D      = field(default=np.array([0, 0, 0]))   # Position of the body
    vel: Vector3D      = field(default=np.array([0, 0, 0]))   # Velocity of the body
    ori: Quaternion    = field(default=Quaternion())              # Orientation of the body
    angVel: Vector3D   = field(default=np.array([0, 0, 0]))   # Angular velocity of the body

    # Constant variables
    density: np.float64     = field(default=1.0)           # Density of the body
    mass: np.float64        = field(default=0.0)           # Mass of the body
    radius: np.float64      = field(default=0.0)           # Radius of the body
    inertia: np.float64     = field(default=0.0)           # Diagonal inertia tensor of the body
    id: int                 = field(default=0)             # Id of the body

    # Force variables
    force: Vector3D    = field(default=np.array([0, 0, 0]))   # Force acting on the body
    torque: Vector3D   = field(default=np.array([0, 0, 0]))   # Torque acting on the body

    def reset_forceTorque(self) -> None:
        self.force  = np.array([0, 0, 0])
        self.torque = np.array([0, 0, 0])
