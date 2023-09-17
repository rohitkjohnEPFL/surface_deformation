# Import all the necessary modules
import numpy as np
from attrs import define, field
from yadeGrid.body import Body
from yadeGrid.vectorFunc import norm
from yadeGrid.yadeTypes import Vector3D, F64


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

    def __attrs_post_init__(self) -> None:
        '''
        Calculates the stiffnesses of the interaction and
        assigning mass of the edge to the nodes
        '''
        self.shear_mod = self.young_mod / (2 * (1 + self.poisson))


        # Assigning mass of the grid
        len: F64     = norm(self.body1.pos - self.body2.pos)
        rad: F64     = self.body1.radius
        halfVol: F64 = 0.5 * np.pi * rad**2 * len
        density: F64 = self.body1.density
        mass: F64    = density * halfVol
        geomInert: F64 = 2. / 5. * mass * rad**2


        # Each interaction adds half the mass and half the moment of inertia
        # of the cylinder to each node
        self.body1.mass = self.body1.mass + mass
        self.body2.mass = self.body2.mass + mass

        self.body1.inertia = self.body1.inertia + geomInert
        self.body2.inertia = self.body2.inertia + geomInert


        # Calculating the normal vector, 2 wrt 1
        self.normal = (self.body2.pos - self.body1.pos) / len


        # calculating the stiffnesses
        area = np.pi * rad**2
        torsionalAreaMoment  = np.pi * rad**4 / 2
        bendingAreaMoment    = np.pi * rad**4 / 4

        self.k_normal  = self.young_mod * area / len
        self.k_torsion = self.shear_mod * torsionalAreaMoment / len
        self.k_shear   = 12.0 * self.young_mod * bendingAreaMoment / len**3
        self.k_bending =        self.young_mod * bendingAreaMoment / len
