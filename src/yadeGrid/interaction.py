# Import all the necessary modules
import numpy as np
from typing import List, Tuple, Dict, Union
from attrs import define, frozen, field, Factory
from numba import jit
from yadeGrid.body import *
from yadeGrid.vectorFunc import *

import numpy.typing as npt


@define
class Interaction:
    body1: Body
    body2: Body
    young_mod: np.float64   = field(default=1e6)
    shear_mod: np.float64   = field(default=1e6)
    poisson: np.float64     = field(default=0.3)
    k_normal: np.float64    = field(default=0.0)
    k_shear: np.float64     = field(default=0.0)
    k_bending: np.float64   = field(default=0.0)
    k_torsion: np.float64   = field(default=0.0)
    normal: Vector3D        = field(default=np.array([0,0,0]))

    def __attrs_post_init__(self)->None:
        self.shear_mod = self.young_mod/(2*(1+self.poisson))

        # Assigning mass of the grid
        len: np.float64     = norm(self.body1.pos - self.body2.pos)
        rad: np.float64     = self.body1.radius
        vol: np.float64     = np.float64(0.5)
        density: np.float64 = self.body1.density
        mass: np.float64    = density*vol
        geomInert: np.float64 = 2./5.*mass*rad**2

