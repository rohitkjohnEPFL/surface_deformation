# Import all the necessary modules
import numpy as np
from attrs import define, field
from yadeGrid.body import Body, Quaternion, AxisAngle
from yadeGrid.vectorFunc import norm, normalise, dotProduct, crossProduct
from yadeGrid.yadeTypes import Vector3D, F64
# from numba import jit


@define(slots=True)
class Interaction:
    # Required attributes
    body1: Body
    body2: Body
    dt: F64
    young_mod: F64   = field(default=70e9)  # Aluminium
    poisson: F64     = field(default=0.35)  # Aluminium

    # Calculated attributes
    shear_mod: F64   = field(default=1e6)
    k_normal: F64    = field(default=0.0)
    k_shear: F64     = field(default=0.0)
    k_bending: F64   = field(default=0.0)
    k_torsion: F64   = field(default=0.0)
    edge_length: F64 = field(default=0.0)

    # Calculated states
    relative_pos: Vector3D  = field(default=np.array([0, 0, 0], dtype=F64))
    relative_ori: Quaternion = field(default=Quaternion())
    relative_ori_AA: AxisAngle = field(default=AxisAngle())
    orthonormal_axis: Vector3D = field(default=np.array([0, 0, 0], dtype=F64))
    twist_axis: Vector3D       = field(default=np.array([0, 0, 0], dtype=F64))
    prev_normal: Vector3D      = field(default=np.array([0, 0, 0], dtype=F64))
    curr_normal: Vector3D      = field(default=np.array([0, 0, 0]))
    relativeVelocity: Vector3D = field(default=np.array([0, 0, 0], dtype=F64))
    shearInc: Vector3D         = field(default=np.array([0, 0, 0], dtype=F64))


    # Default initialised attributes
    normal_force: Vector3D   = field(default=np.array([0, 0, 0], dtype=F64))
    shear_force: Vector3D    = field(default=np.array([0, 0, 0], dtype=F64))
    bending_moment: Vector3D = field(default=np.array([0, 0, 0], dtype=F64))
    torsion_moment: Vector3D = field(default=np.array([0, 0, 0], dtype=F64))

    normal_defo: F64         = field(default=0.0)
    shear_defo: Vector3D     = field(default=np.array([0, 0, 0], dtype=F64))
    bending_defo: Vector3D   = field(default=np.array([0, 0, 0], dtype=F64))
    torsion_defo: F64        = field(default=0.0)

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
        self.curr_normal = (self.body2.pos - self.body1.pos) / edge_length
        self.prev_normal = self.curr_normal


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
        # The force is calculated with respect to body 1.
        # The force on body 2 is the negative of this force
        self.update_currNormal()
        self.update_relativePos()
        self.calc_NormalForce()
        self.calc_torsionMoment()
        self.calc_bendingMoment()
        


    def update_currNormal(self) -> None:
        # print(self.body2.pos)
        # print(self.body1.pos)
        self.curr_normal = normalise(self.body2.pos - self.body1.pos)

    def update_relativePos(self) -> None:
        self.relative_pos = self.body2.pos - self.body1.pos
        ori1 = self.body1.ori
        ori2 = self.body2.ori
        ori1_inv = ori1.inverse()
        self.relative_ori = ori1_inv * ori2
        self.relative_ori_AA = self.relative_ori.conv_2axisAngle()

    # @jit(nopython=True)  # type: ignore
    def calc_NormalForce(self) -> None:
        # The force is calculated with respect to body 1.
        # The force on body 2 is the negative of this force
        defo              = norm(self.body2.pos - self.body1.pos) - self.edge_length
        self.normal_defo  = defo
        self.normal_force = self.k_normal * defo * self.curr_normal

        # If you want to use numba, use the following code
        # self.normal_force = calc_NormalForce_JIT(self.body1.pos, self.body2.pos, self.normal, self.edge_length, self.k_normal)

    def calc_torsionMoment(self) -> None:
        axisAngle: AxisAngle  = self.relative_ori_AA
        twist: F64            = axisAngle.angle * dotProduct(axisAngle.axis, self.curr_normal)
        self.torsion_defo   = twist
        self.torsion_moment = self.k_torsion * twist * self.curr_normal

    def calc_bendingMoment(self) -> None:
        axisAngle: AxisAngle = self.relative_ori_AA
        twist: F64           = axisAngle.angle * dotProduct(axisAngle.axis, self.curr_normal)
        bending: Vector3D    = axisAngle.angle * axisAngle.axis - twist * self.curr_normal
        self.bending_defo    = bending
        self.bending_moment  = self.k_bending * bending

    def precompute_ForShear(self) -> None:
        '''To compute the shear increment, shear force is calculated using an incremental formulation

        see: https://www.sciencedirect.com/science/article/pii/S0925857413001936?via%3Dihub

        for the implementation

        check the bool Law2_ScGeom6D_CohFrictPhys_CohesionMoment::go() function in 
        https://gitlab.com/yade-dev/trunk/-/blob/master/pkg/dem/CohesiveFrictionalContactLaw.cpp?ref_type=heads'''
        self.orthonormal_axis = np.cross(self.prev_normal, self.curr_normal)
        angle                 = 0.5 * dotProduct(self.body1.angVel + self.body2.angVel, self.curr_normal)
        self.twist_axis       = angle * self.curr_normal
        realtiveVel           = self.calc_IncidentVel()
        realtiveVel           = realtiveVel - dotProduct(realtiveVel, self.curr_normal) * self.curr_normal
        self.shearInc         = realtiveVel * self.dt

    def calc_IncidentVel(self) -> Vector3D:
        rad: F64 = self.body1.radius
        center2center_dist: F64  = norm(self.body2.pos - self.body1.pos)
        penetrationDepth: F64    = 2.0 * rad - center2center_dist

        # This alpha value is used to avoid granular ratcheting.
        # See the Vector3r ScGeom::getIncidentVel() function in
        # https://gitlab.com/yade-dev/trunk/-/blob/master/pkg/dem/ScGeom.cpp?ref_type=heads
        # around line 66
        alpha: F64               = (rad + rad) / (rad + rad - penetrationDepth)

        tangentialVel2: Vector3D = crossProduct(self.body2.angVel, - rad * self.curr_normal)
        tangentialVel1: Vector3D = crossProduct(self.body1.angVel,   rad * self.curr_normal)
        relativeVelocity = (self.body2.vel - self.body1.vel) * alpha + tangentialVel2 - tangentialVel1
        return relativeVelocity

    def rotate_shearForce(self) -> None:
        self.shear_force = self.shear_force - crossProduct(self.shear_force, self.orthonormal_axis)
        self.shear_force = self.shear_force - crossProduct(self.shear_force, self.twist_axis)

    def calc_ShearForce(self) -> None:
        self.precompute_ForShear()
        self.rotate_shearForce()
        self.shear_force = self.shear_force - self.k_shear * self.shearInc


# @jit(nopython=True)  # type: ignore
# def calc_NormalForce_JIT(pos1, pos2, normal, edge_length, k_normal) -> Vector3D:
#     '''
#     Calculates the normal force between two nodes
#     '''
#     defo = norm(pos2 - pos1) - edge_length
#     return k_normal * defo * normal
