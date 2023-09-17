from yadeGrid import Body, Interaction, F64
from unittest import TestCase
import numpy as np
from numpy.testing import assert_almost_equal


class test_Interaction(TestCase):
    def test_initialisation(self):
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        pos3    = np.array([3, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)

        body1 = Body(pos=pos1, radius=rad, density=density)
        body2 = Body(pos=pos2, radius=rad, density=density)
        body3 = Body(pos=pos3, radius=rad, density=density)
        inter1 = Interaction(body1, body2, young_mod=young, poisson=poisson)
        inter2 = Interaction(body2, body3, young_mod=young, poisson=poisson)

        self.assertEqual(inter1.body1, body1)
        self.assertEqual(inter1.body2, body2)
        self.assertEqual(inter2.body1, body2)
        self.assertEqual(inter2.body2, body3)

    def test_initialisation_bodyMass(self):
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        pos3    = np.array([3, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)

        len_12 = np.linalg.norm(pos2 - pos1)
        halfVol_12 = 0.5 * np.pi * rad**2 * len_12
        halfMass_12 = density * halfVol_12

        len_23 = np.linalg.norm(pos3 - pos2)
        halfVol_23 = 0.5 * np.pi * rad**2 * len_23
        halfMass_23 = density * halfVol_23

        body1 = Body(pos=pos1, radius=rad, density=density)
        body2 = Body(pos=pos2, radius=rad, density=density)
        body3 = Body(pos=pos3, radius=rad, density=density)
        inter1 = Interaction(body1, body2, young_mod=young, poisson=poisson)
        inter2 = Interaction(body2, body3, young_mod=young, poisson=poisson)

        self.assertEqual(body1.mass, halfMass_12)
        self.assertEqual(body2.mass, halfMass_12 + halfMass_23)
        self.assertEqual(body3.mass, halfMass_23)

    def test_initialisation_bodyInertia(self):
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        pos3    = np.array([3, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)

        len_12 = np.linalg.norm(pos2 - pos1)
        halfVol_12 = 0.5 * np.pi * rad**2 * len_12
        halfMass_12 = density * halfVol_12
        geomInert_12 = 2. / 5. * halfMass_12 * rad**2

        len_23 = np.linalg.norm(pos3 - pos2)
        halfVol_23 = 0.5 * np.pi * rad**2 * len_23
        halfMass_23 = density * halfVol_23
        geomInert_23 = 2. / 5. * halfMass_23 * rad**2

        body1 = Body(pos=pos1, radius=rad, density=density)
        body2 = Body(pos=pos2, radius=rad, density=density)
        body3 = Body(pos=pos3, radius=rad, density=density)
        inter1 = Interaction(body1, body2, young_mod=young, poisson=poisson)
        inter2 = Interaction(body2, body3, young_mod=young, poisson=poisson)

        self.assertEqual(body1.inertia, geomInert_12)
        self.assertEqual(body2.inertia, geomInert_12 + geomInert_23)
        self.assertEqual(body3.inertia, geomInert_23)

    def test_stiffnessCalculation(self):
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        pos3    = np.array([3, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)

        # Geometry
        area    = np.pi * rad**2
        torsionalAreaMoment  = np.pi * rad**4 / 2
        bendingAreaMoment    = np.pi * rad**4 / 4

        len_12 = np.linalg.norm(pos2 - pos1)
        len_23 = np.linalg.norm(pos3 - pos2)

        # Initialise body and interactions
        body1 = Body(pos=pos1, radius=rad, density=density)
        body2 = Body(pos=pos2, radius=rad, density=density)
        body3 = Body(pos=pos3, radius=rad, density=density)
        inter1 = Interaction(body1, body2, young_mod=young, poisson=poisson)
        inter2 = Interaction(body2, body3, young_mod=young, poisson=poisson)

        # Check shear modulus
        shear_mod = young / (2 * (1 + poisson))
        self.assertEqual(inter1.shear_mod, shear_mod)
        self.assertEqual(inter2.shear_mod, shear_mod)

        # Calculated stiffnesses
        k_normal_12  = young * area / len_12
        k_torsion_12 = shear_mod * torsionalAreaMoment / len_12
        k_shear_12   = 12.0 * young * bendingAreaMoment / len_12**3
        k_bending_12 = young * bendingAreaMoment / len_12

        k_normal_23  = young * area / len_23
        k_torsion_23 = shear_mod * torsionalAreaMoment / len_23
        k_shear_23   = 12.0 * young * bendingAreaMoment / len_23**3
        k_bending_23 = young * bendingAreaMoment / len_23

        # Check stiffnesses
        self.assertEqual(inter1.k_normal, k_normal_12)
        self.assertEqual(inter1.k_torsion, k_torsion_12)
        self.assertEqual(inter1.k_shear, k_shear_12)
        self.assertEqual(inter1.k_bending, k_bending_12)

        self.assertEqual(inter2.k_normal, k_normal_23)
        self.assertEqual(inter2.k_torsion, k_torsion_23)
        self.assertEqual(inter2.k_shear, k_shear_23)
        self.assertEqual(inter2.k_bending, k_bending_23)


    def test_fromYade(self):
        '''The values were taken from a YADE simulation with the following script:
            # -*- encoding=utf-8 -*-

            from builtins import zip
            from builtins import range
            from yade import pack, geom, qt
            from yade.gridpfacet import *
            from pylab import *

            O.engines = [
                    ForceResetter(),
                    InsertionSortCollider([Bo1_GridConnection_Aabb()]),
                    InteractionLoop(
                            [Ig2_GridNode_GridNode_GridNodeGeom6D()],
                            [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys(setCohesionNow=True, setCohesionOnNewContacts=False)],
                            [Law2_ScGeom6D_CohFrictPhys_CohesionMoment()]
                    ),
                    NewtonIntegrator(gravity=(0, 0, -10), damping=0.1, label='newton')
            ]

            O.materials.append(
                    CohFrictMat(young=70e9, poisson=0.35, density=2700, frictionAngle=radians(10), normalCohesion=1e7, 
                    shearCohesion=1e7, momentRotationLaw=True, label='mat')
            )

            ### Parameters ###
            r = 0.1

            ### Create all nodes first
            nodeIds = []
            poses = [[0,0,0], [1,0,0], [3,0,0]]
            for i in poses:
                nodeIds.append(O.bodies.append(gridNode(i, r, wire=False, fixed=False, material='mat', color=color)))

            ### Create connections between the nodes
            connectionIds = []
            for i, j in zip(nodeIds[:-1], nodeIds[1:]):
                connectionIds.append(O.bodies.append(gridConnection(i, j, r, color=color)))

            ### Set a fixed node
            O.bodies[0].dynamic = False

            O.dt = 1e-06
            O.saveTmp()
            qt.View()
        '''
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        pos3    = np.array([3, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)

        # Initialise body and interactions
        body1 = Body(pos=pos1, radius=rad, density=density)
        body2 = Body(pos=pos2, radius=rad, density=density)
        body3 = Body(pos=pos3, radius=rad, density=density)
        inter1 = Interaction(body1, body2, young_mod=young, poisson=poisson)
        inter2 = Interaction(body2, body3, young_mod=young, poisson=poisson)

        # Check mass and inertia
        assert_almost_equal(body1.mass, 42.41150082346221)
        assert_almost_equal(body2.mass, 127.23450247038664)
        assert_almost_equal(body3.mass, 84.82300164692442)

        assert_almost_equal(body1.inertia, 0.16964600329384888)
        assert_almost_equal(body2.inertia, 0.5089380098815466)
        assert_almost_equal(body3.inertia, 0.33929200658769776)

        # Check stiffnesses
        self.assertEqual(inter1.k_normal, 2199114857.5128555)
        self.assertEqual(inter1.k_torsion, 4072434.921320102)
        self.assertEqual(inter1.k_shear, 65973445.725385666)
        self.assertEqual(inter1.k_bending, 5497787.143782139)

        self.assertEqual(inter2.k_normal, 1099557428.7564278)
        self.assertEqual(inter2.k_torsion, 2036217.460660051)
        self.assertEqual(inter2.k_shear, 8246680.715673208)
        self.assertEqual(inter2.k_bending, 2748893.5718910694)
