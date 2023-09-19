from yadeGrid import Body, Interaction, F64, Quaternion
from yadeGrid.vectorFunc import normalise
from unittest import TestCase
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal, assert_array_almost_equal
import json


class test_Interaction(TestCase):
    def test_initialisation(self) -> None:
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

    def test_edgeLengthCalculation(self) -> None:
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)

        body1 = Body(pos=pos1, radius=rad, density=density)
        body2 = Body(pos=pos2, radius=rad, density=density)
        inter1 = Interaction(body1, body2, young_mod=young, poisson=poisson)
        self.assertEqual(inter1.edge_length, 1.0)

    def test_initialisation_bodyMass(self) -> None:
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
        Interaction(body1, body2, young_mod=young, poisson=poisson)
        Interaction(body2, body3, young_mod=young, poisson=poisson)

        self.assertEqual(body1.mass, halfMass_12)
        self.assertEqual(body2.mass, halfMass_12 + halfMass_23)
        self.assertEqual(body3.mass, halfMass_23)

    def test_initialisation_bodyInertia(self) -> None:
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
        Interaction(body1, body2, young_mod=young, poisson=poisson)
        Interaction(body2, body3, young_mod=young, poisson=poisson)

        self.assertEqual(body1.inertia, geomInert_12)
        self.assertEqual(body2.inertia, geomInert_12 + geomInert_23)
        self.assertEqual(body3.inertia, geomInert_23)

    def test_stiffnessCalculation(self) -> None:
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


    def test_StiffnessFromYade(self) -> None:
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

    def test_updateNormal(self) -> None:
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)
        b1 = Body(pos=pos1, radius=rad, density=density)
        b2 = Body(pos=pos2, radius=rad, density=density)
        inter = Interaction(b1, b2, young_mod=young, poisson=poisson)

        # Check normal vector
        norm1 = normalise(pos2 - pos1)
        assert_almost_equal(inter.normal, norm1)

        # Checking update normal
        pos3   = np.array([0, 1, 0], dtype=F64)
        b1.pos = np.array(pos3, dtype=F64)
        norm2  = normalise(pos2 - pos3)
        inter.update_normal()
        assert_almost_equal(inter.normal, norm2)

        # Checking update raises ZeroDivisionError
        b1.pos = np.array([1, 0, 0], dtype=F64)
        with self.assertRaises(ZeroDivisionError):
            inter.update_normal()

    def test_normalForceCalculation(self) -> None:
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)
        b1 = Body(pos=pos1, radius=rad, density=density)
        b2 = Body(pos=pos2, radius=rad, density=density)
        inter = Interaction(b1, b2, young_mod=young, poisson=poisson)

        # Check normal force
        inter.update_normal()
        inter.calc_NormalForce()
        assert_array_equal(inter.normal_force, np.array([0, 0, 0], dtype=F64))

        # Rotate body 2 and check normal force
        pos3 = np.array([0, 1, 0], dtype=F64)
        b2.pos = pos3
        inter.update_normal()
        inter.calc_NormalForce()
        assert_almost_equal(inter.normal_force, np.array([0, 0, 0], dtype=F64))

        # Move body 2 and check normal force
        pos4 = np.array([1.5, 0, 0], dtype=F64)
        b2.pos = pos4
        inter.update_normal()
        inter.calc_NormalForce()
        forceExp = (pos4 - np.array([1, 0, 0], dtype=F64)) * inter.k_normal
        assert_almost_equal(inter.normal_force, forceExp)

    def test_normalForceFromYade(self) -> None:
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)
        b1 = Body(pos=pos1, radius=rad, density=density)
        b2 = Body(pos=pos2, radius=rad, density=density)
        inter = Interaction(b1, b2, young_mod=young, poisson=poisson)


        with open(".\\tests\\yadeResults\\normalForceTest.json", "r") as file:
            yadeResult = json.load(file)

        yadePos   = yadeResult["pos"]
        yadeForce = yadeResult["force"]

        forceCalc = []
        for pos in yadePos:
            b2.pos = np.array([pos, 0., 0.], dtype=F64)
            inter.update_normal()
            inter.calc_NormalForce()

            # Minus because force is calculated in terms of body 1 and it is equal and opposite
            # for body 2
            forceCalc.append(-inter.normal_force[0])

        assert_array_equal(forceCalc, yadeForce)


    def test_relativePos(self) -> None:
        rad     = F64(0.1)
        density = F64(2700.0)
        young   = F64(70e9)
        poisson = F64(0.35)

        ang1    = np.pi / 4
        axis1   = normalise(np.array([0, 4, 1], dtype=F64))
        ori1    = Quaternion(np.array([np.cos(ang1 / 2), *np.sin(ang1 / 2) * axis1]))
        ori1Inv = ori1.inverse()

        ang2    = np.pi / 2
        axis2   = normalise(np.array([0, 1, 5], dtype=F64))
        ori2    = Quaternion(np.array([np.cos(ang2 / 2), *np.sin(ang2 / 2) * axis2]))

        relative_ori = ori1Inv * ori2
        relative_ori.normalize()

        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        relative_pos = pos2 - pos1

        b1 = Body(pos=pos1, radius=rad, density=density, ori=ori1)
        b2 = Body(pos=pos2, radius=rad, density=density, ori=ori2)
        inter = Interaction(b1, b2, young_mod=young, poisson=poisson)
        inter.update_relativePos()

        self.assertEqual(inter.relative_ori, relative_ori)
        assert_array_equal(inter.relative_pos, relative_pos)

    def test_torsionMoment(self) -> None:
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)

        b1 = Body(pos=pos1, radius=rad, density=density)
        b2 = Body(pos=pos2, radius=rad, density=density)
        inter = Interaction(b1, b2, young_mod=young, poisson=poisson)

        # Check normal force
        inter.update_normal()
        inter.update_relativePos()
        inter.calc_torsionMoment()
        assert_array_equal(inter.torsion_moment, np.array([0, 0, 0], dtype=F64))

        # Rotate body 2 perpendicular to normal and ckech torsion
        ang1    = np.pi / 4
        axis1   = normalise(np.array([0, 0, 1], dtype=F64))
        ori1    = Quaternion(np.array([np.cos(ang1 / 2), *np.sin(ang1 / 2) * axis1]))

        b2.ori = ori1
        inter.update_normal()
        inter.update_relativePos()
        inter.calc_torsionMoment()
        assert_almost_equal(inter.torsion_moment, np.array([0, 0, 0], dtype=F64))

        # Rotate body 2 along normal and check torsion
        ang2    = np.pi / 20
        axis2   = normalise(np.array([1, 0, 0], dtype=F64))
        ori2    = Quaternion(np.array([np.cos(ang2 / 2), *np.sin(ang2 / 2) * axis2]))

        b2.ori = ori2
        inter.update_normal()
        inter.update_relativePos()
        inter.calc_torsionMoment()
        MomentExp = inter.normal * ang2 * inter.k_torsion
        assert_almost_equal(inter.torsion_moment, MomentExp)

    def test_torsionMomentYade(self) -> None:
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)
        b1 = Body(pos=pos1, radius=rad, density=density)
        b2 = Body(pos=pos2, radius=rad, density=density)
        inter = Interaction(b1, b2, young_mod=young, poisson=poisson)

        with open(".\\tests\\yadeResults\\torsionMomentTestAngVel_5e4.json", "r") as file:
            yadeResult = json.load(file)
            yadeResultOri: list[list[float]] = yadeResult["ori"]
            yadeResultMoment: list[list[float]] = yadeResult["moment"]

        yadeOri   = [Quaternion(np.array(ori)) for ori in yadeResultOri]
        yadeMoment = yadeResultMoment

        momentCalc = []
        for ori in yadeOri:
            b2.ori = ori
            inter.update_normal()
            inter.update_relativePos()
            inter.calc_torsionMoment()

            # Minus because force is calculated in terms of body 1 and it is equal and opposite
            # for body 2
            momentCalc.append(-inter.torsion_moment[0])

        assert_array_almost_equal(momentCalc, yadeMoment)

    def test_bendingMoment(self) -> None:
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)
        b1 = Body(pos=pos1, radius=rad, density=density)
        b2 = Body(pos=pos2, radius=rad, density=density)
        inter = Interaction(b1, b2, young_mod=young, poisson=poisson)

        # Check normal force
        inter.update_normal()
        inter.update_relativePos()
        inter.calc_NormalForce()
        inter.calc_torsionMoment()
        inter.calc_bendingMoment()
        assert_array_equal(inter.bending_moment, np.array([0, 0, 0], dtype=F64))

        # Rotate body 2 parallel to normal and ckech bending
        ang1    = np.pi / 4
        axis1   = normalise(np.array([1, 0, 0], dtype=F64))
        ori1    = Quaternion(np.array([np.cos(ang1 / 2), *np.sin(ang1 / 2) * axis1]))

        b2.ori = ori1
        inter.update_normal()
        inter.update_relativePos()
        inter.calc_NormalForce()
        inter.calc_torsionMoment()
        inter.calc_bendingMoment()
        assert_almost_equal(inter.bending_moment, np.array([0, 0, 0], dtype=F64))

        # Rotate body 2 along normal and check torsion
        ang2    = np.pi / 20
        axis2   = normalise(np.array([0, 1, 0], dtype=F64))
        ori2    = Quaternion(np.array([np.cos(ang2 / 2), *np.sin(ang2 / 2) * axis2]))

        b2.ori = ori2
        inter.update_normal()
        inter.update_relativePos()
        inter.calc_NormalForce()
        inter.calc_torsionMoment()
        inter.calc_bendingMoment()
        MomentExp = axis2 * ang2 * inter.k_bending
        assert_almost_equal(inter.bending_moment, MomentExp)


    def test_bendingMomentYade(self) -> None:
        rad     = F64(0.1)
        density = F64(2700.0)
        pos1    = np.array([0, 0, 0], dtype=F64)
        pos2    = np.array([1, 0, 0], dtype=F64)
        young   = F64(70e9)
        poisson = F64(0.35)
        b1 = Body(pos=pos1, radius=rad, density=density)
        b2 = Body(pos=pos2, radius=rad, density=density)
        inter = Interaction(b1, b2, young_mod=young, poisson=poisson)

        with open(".\\tests\\yadeResults\\bendingMomentTest.json", "r") as file:
            yadeResult = json.load(file)
            yadeResultOri: list[list[float]] = yadeResult["ori"]
            yadeResultMoment: list[list[float]] = yadeResult["moment"]

        yadeOri   = [Quaternion(np.array(ori)) for ori in yadeResultOri]
        yadeMoment = yadeResultMoment

        momentCalc = []
        for ori in yadeOri:
            b2.ori = ori
            inter.update_normal()
            inter.update_relativePos()
            inter.calc_torsionMoment()
            inter.calc_bendingMoment()

            # Minus because force is calculated in terms of body 1 and it is equal and opposite
            # for body 2
            momentCalc.append(-inter.bending_moment[1])

        assert_array_almost_equal(momentCalc, yadeMoment)

    # TODO test bending moment calculated not from YADE
