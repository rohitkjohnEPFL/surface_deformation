# Import all the necessary modules
import numpy as np
from attrs import define, field
from yadeGrid.body import Body, Quaternion
from yadeGrid.vectorFunc import norm, normalise, dotProduct
from yadeGrid.yadeTypes import Vector3D, F64
# from numba import jit


@define(slots=True)
class Interaction:
    # Required attributes
    body1: Body
    body2: Body
    young_mod: F64   = field(default=70e9)  # Aluminium
    poisson: F64     = field(default=0.35)  # Aluminium

    # Calculated attributes
    shear_mod: F64   = field(default=1e6)
    k_normal: F64    = field(default=0.0)
    k_shear: F64     = field(default=0.0)
    k_bending: F64   = field(default=0.0)
    k_torsion: F64   = field(default=0.0)
    normal: Vector3D        = field(default=np.array([0, 0, 0]))
    edge_length: F64 = field(default=0.0)
    relative_pos: Vector3D  = field(default=np.array([0, 0, 0], dtype=F64))
    relative_ori: Quaternion = field(default=Quaternion())

    # Default initialised attributes
    normal_force: Vector3D   = field(default=np.array([0, 0, 0], dtype=F64))
    shear_force: Vector3D    = field(default=np.array([0, 0, 0], dtype=F64))
    bending_moment: Vector3D = field(default=np.array([0, 0, 0], dtype=F64))
    torsion_moment: Vector3D = field(default=np.array([0, 0, 0], dtype=F64))

    normal_defo: F64         = field(default=0.0)
    shear_defo: Vector3D     = field(default=np.array([0, 0, 0], dtype=F64))
    bending_defo: Vector3D   = field(default=np.array([0, 0, 0], dtype=F64))
    torsion_defo: Vector3D   = field(default=np.array([0, 0, 0], dtype=F64))

    def __attrs_post_init__(self) -> None:
        '''
        Calculates the stiffnesses of the interaction and
        assigning mass of the edge to the nodes
        '''
        self.shear_mod = self.young_mod / (2 * (1 + self.poisson))


        # Assigning mass of the grid edge to the nodes
        edge_length: F64 = norm(self.body1.pos - self.body2.pos)
        rad: F64         = self.body1.radius
        halfVol: F64     = 0.5 * np.pi * rad**2 * edge_length
        density: F64     = self.body1.density
        mass: F64        = density * halfVol
        geomInert: F64   = 2. / 5. * mass * rad**2

        # assigning edge length
        self.edge_length  = edge_length
        self.relative_pos = self.body2.pos - self.body1.pos

        # Each interaction adds half the mass and half the moment of inertia
        # of the cylinder to each node
        self.body1.mass = self.body1.mass + mass
        self.body2.mass = self.body2.mass + mass

        self.body1.inertia = self.body1.inertia + geomInert
        self.body2.inertia = self.body2.inertia + geomInert


        # Calculating the normal vector, 2 wrt 1
        self.normal = (self.body2.pos - self.body1.pos) / edge_length


        # calculating the stiffnesses
        area = np.pi * rad**2
        torsionalAreaMoment  = np.pi * rad**4 / 2
        bendingAreaMoment    = np.pi * rad**4 / 4

        self.k_normal  = self.young_mod * area / edge_length
        self.k_torsion = self.shear_mod * torsionalAreaMoment / edge_length
        self.k_shear   = 12.0 * self.young_mod * bendingAreaMoment / edge_length**3
        self.k_bending =        self.young_mod * bendingAreaMoment / edge_length

    def __repr__(self) -> str:
        return f"Interaction between {self.body1.id} and {self.body2.id}. Stiffness \n \
                k_normal  = {self.k_normal}, \n \
                k_shear   = {self.k_shear}, \n \
                k_bending = {self.k_bending}, \n \
                k_torsion = {self.k_torsion}"

    def __str__(self) -> str:
        return f"Interaction between {self.body1.id} and {self.body2.id}. Stiffness \n \
                k_normal  = {self.k_normal}, \n \
                k_shear   = {self.k_shear}, \n \
                k_bending = {self.k_bending}, \n \
                k_torsion = {self.k_torsion}"

    def reset_ForceTorque(self) -> None:
        self.normal_force = np.array([0, 0, 0], dtype=F64)
        self.shear_force  = np.array([0, 0, 0], dtype=F64)
        self.bending_moment = np.array([0, 0, 0], dtype=F64)
        self.torsion_moment = np.array([0, 0, 0], dtype=F64)

    def calc_ForcesTorques(self) -> None:
        self.update_normal()
        self.update_relativePos()
        self.calc_NormalForce()


    def update_normal(self) -> None:
        self.normal = normalise(self.body2.pos - self.body1.pos)

    def update_relativePos(self) -> None:
        self.relative_pos = self.body2.pos - self.body1.pos
        ori1 = self.body1.ori
        ori2 = self.body2.ori
        ori1_inv = ori1.inverse()
        self.relative_ori = ori1_inv * ori2

    # @jit(nopython=True)  # type: ignore
    def calc_NormalForce(self) -> None:
        defo              = norm(self.body2.pos - self.body1.pos) - self.edge_length
        self.normal_force = self.k_normal * defo * self.normal

        # If you want to use numba, use the following code
        # self.normal_force = calc_NormalForce_JIT(self.body1.pos, self.body2.pos, self.normal, self.edge_length, self.k_normal)

    def calc_torsionMoment(self) -> None:
        axisAngle = self.relative_ori.conv_2axisAngle()
        twist     = axisAngle.angle * dotProduct(axisAngle.axis, self.normal)
        self.torsion_moment = self.k_torsion * twist * self.normal


# @jit(nopython=True)  # type: ignore
# def calc_NormalForce_JIT(pos1, pos2, normal, edge_length, k_normal) -> Vector3D:
#     '''
#     Calculates the normal force between two nodes
#     '''
#     defo = norm(pos2 - pos1) - edge_length
#     return k_normal * defo * normal
