# Import all the necessary modules
import numpy as np
from typing import cast
from attrs import define, field
from numba import jit
from yadeGrid.yadeTypes import Vector3D

# ------------------------------------------------------------------------------------------------ #
#                                                                                       QUATERNION #
# ------------------------------------------------------------------------------------------------ #


@define
class Quaternion:
    a: np.float64 = field(default=1.0)   # Real part of the quaternion
    b: np.float64 = field(default=0.0)   # First imaginary part of the quaternion
    c: np.float64 = field(default=0.0)   # Second imaginary part of the quaternion
    d: np.float64 = field(default=0.0)   # Third imaginary part of the quaternion

    @jit  # type: ignore
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        if isinstance(other, Quaternion):
            return Quaternion(
                a=self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d,
                b=self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c,
                c=self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b,
                d=self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a
            )
        else:
            raise TypeError(f"Cannot multiply Quaternion with {type(other)}")

    @jit  # type: ignore
    def conjugate(self) -> 'Quaternion':
        return Quaternion(a=self.a, b=-self.b, c=-self.c, d=-self.d)

    @jit  # type: ignore
    def norm(self) -> np.float64:
        return cast(np.float64, np.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2))

    @jit  # type: ignore
    def normalize(self) -> np.float64:
        return cast(np.float64, self / self.norm())

    @jit  # type: ignore
    def conv_2axisAngle(self) -> 'AxisAngle':
        angle = 2 * np.arccos(self.a)
        axis = np.array([self.b, self.c, self.d]) / np.sqrt(1 - self.a**2)
        return AxisAngle(axis=axis, angle=angle)

    @jit  # type: ignore
    def __truediv__(self, scalar: np.float64) -> 'Quaternion':
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Quaternion by zero")

        return Quaternion(
            a=self.a / scalar,
            b=self.b / scalar,
            c=self.c / scalar,
            d=self.d / scalar
        )


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

    def conv_2quaternion(self) -> Quaternion:
        return Quaternion(
            a=np.cos(self.angle / 2),
            b=self.axis[0] * np.sin(self.angle / 2),
            c=self.axis[1] * np.sin(self.angle / 2),
            d=self.axis[2] * np.sin(self.angle / 2)
        )


# ------------------------------------------------------------------------------------------------ #
#                                                                                             BODY #
# ------------------------------------------------------------------------------------------------ #
@define(kw_only=True)
class Body:
    # State variables
    pos: Vector3D      = field(default=np.array([[0, 0, 0]]))   # Position of the body
    vel: Vector3D      = field(default=np.array([[0, 0, 0]]))   # Velocity of the body
    ori: Quaternion    = field(default=Quaternion())              # Orientation of the body
    angVel: Vector3D   = field(default=np.array([[0, 0, 0]]))   # Angular velocity of the body

    # Constant variables
    density: np.float64     = field(default=0.0)           # Density of the body
    mass: np.float64        = field(default=0.0)           # Mass of the body
    radius: np.float64      = field(default=0.0)           # Radius of the body
    inertia: np.float64     = field(default=0.0)           # Diagonal inertia tensor of the body
    id: int                 = field(default=0)             # Id of the body

    # Force variables
    force: Vector3D    = field(default=np.array([[0, 0, 0]]))   # Force acting on the body
    torque: Vector3D   = field(default=np.array([[0, 0, 0]]))   # Torque acting on the body

    def reset_forceTorque(self) -> None:
        self.force  = np.array([[0, 0, 0]])
        self.torque = np.array([[0, 0, 0]])
