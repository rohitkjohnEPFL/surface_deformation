# Import all the necessary modules
import numpy as np
from attrs import define, field
from yadeGrid.body import Body
from yadeGrid.vectorFunc import norm
from yadeGrid.yadeTypes import Vector3D, F64
from numpy import array

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

    def reset_ForceTorque(self):
        self.normal_force = np.array([0, 0, 0], dtype=F64)
        self.shear_force  = np.array([0, 0, 0], dtype=F64)
        self.bending_moment = np.array([0, 0, 0], dtype=F64)
        self.torsion_moment = np.array([0, 0, 0], dtype=F64)

    def update_normal(self):
        self.normal = (self.body2.pos - self.body1.pos) / norm(self.body2.pos - self.body1.pos)

    def calc_NormalForce(self):
        self.update_normal()
        defo              = norm(self.body2.pos - self.body1.pos) - self.edge_length
        self.normal_force = self.k_normal * defo * self.normal

    
