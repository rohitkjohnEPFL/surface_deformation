# Import all the necessary modules
import numpy as np
from attrs import define, field
from numba import jit
from yadeGrid.yadeTypes import Vector3D, F64, QuatComps
from yadeGrid.vectorFunc import norm
from numpy.typing import NDArray
from typing import Any


# ------------------------------------------------------------------------------------------------ #
#                                                                                       QUATERNION #
# ------------------------------------------------------------------------------------------------ #
@define
class Quaternion:
    components: NDArray[F64] = field(default=np.array([1.0, 0.0, 0.0, 0.0], dtype=F64))
    # This way we can pass the quaternion as ndarray to JITed functions

    @property
    def a(self) -> F64:
        return F64(self.components[0])

    @property
    def b(self) -> F64:
        return F64(self.components[1])

    @property
    def c(self) -> F64:
        return F64(self.components[2])

    @property
    def d(self) -> F64:
        return F64(self.components[3])

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

    def norm(self) -> F64:
        return F64(norm_quat(self.components))

    def normalize(self) -> None:
        normalized = self / self.norm()
        self.components = normalized.components

    def inverse(self) -> 'Quaternion':
        return self.conjugate() / self.norm()**2

    def conv_2axisAngle(self) -> 'AxisAngle':
        self.normalize()
        angle = 2 * np.arccos(self.a)
        axis = np.array([self.b, self.c, self.d])
        axisNorm = norm(axis)
        if axisNorm != 0:
            axis = axis / axisNorm
        return AxisAngle(axis=axis, angle=angle)

    def __truediv__(self, scalar: F64) -> 'Quaternion':
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Quaternion by zero")

        return Quaternion(np.array([self.a, self.b, self.c, self.d]) / scalar)


# ---------------------------------------------------------------------------------- multiply_quat #
# JIT functions are given ndarray as input and output. The methods in quaternion functions
# handle the task of constructing the quaternion from returned the ndarray
@jit(nopython=True)  # type: ignore
def multiply_quat(q1: QuatComps, q2: QuatComps) -> QuatComps:
    result: QuatComps = np.zeros(4, dtype=F64)
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    result[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    result[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

    return result


# -------------------------------------------------------------------------------------- norm_quat #
@jit(nopython=True)  # type: ignore
def norm_quat(q1: QuatComps) -> F64:
    return F64(np.sqrt(q1[0]**2 + q1[1]**2 + q1[2]**2 + q1[3]**2))


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
        if norm(self.axis) != 0:
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
    density: F64     = field(default=1.0)           # Density of the body
    mass: F64        = field(default=0.0)           # Mass of the body
    radius: F64      = field(default=0.0)           # Radius of the body
    inertia: F64     = field(default=0.0)           # Diagonal inertia tensor of the body
    id: int                 = field(default=0)             # Id of the body

    # Force variables
    force: Vector3D    = field(default=np.array([0, 0, 0]))   # Force acting on the body
    torque: Vector3D   = field(default=np.array([0, 0, 0]))   # Torque acting on the body

    # Simulation parameters
    DynamicQ: bool = field(default=False)   # Whether the body is affected by forces or not
    BlockedDOFs: str = field(default="")    # Which degrees of freedom are blocked, e.g. "xyz" or "xyzXYZ"
                                            # x, y, z, X, Y, Z correspond to the 6 DOFs of the body
                                            # x, y, z are the translational DOFs
                                            # X, Y, Z are the rotational DOFs

    def reset_forceTorque(self) -> None:
        self.force  = np.array([0, 0, 0])
        self.torque = np.array([0, 0, 0])

    def add_Forces(self, force: Vector3D) -> None:
        self.force = self.force + force

    def add_Torques(self, torque: Vector3D) -> None:
        self.torque = self.torque + torque
