from yadeGrid import Body, Quaternion, AxisAngle
from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from random import random


# ------------------------------------------------------------------------------------------------ #
#                                                                                        TEST_BODY #
# ------------------------------------------------------------------------------------------------ #
class test_body(TestCase):
    def test_body(self) -> None:
        body = Body()
        # test states are initialized correctly to 0
        assert_array_equal(body.pos, np.array([0, 0, 0]))
        assert_array_equal(body.vel, np.array([0, 0, 0]))
        self.assertEqual(body.ori, Quaternion())
        assert_array_equal(body.angVel, np.array([0, 0, 0]))

        # Test the forces are initialized correctly to 0
        assert_array_equal(body.force, np.array([0, 0, 0]))
        assert_array_equal(body.torque, np.array([0, 0, 0]))

        # Test the constants are initialized correctly to 0
        self.assertEqual(body.mass, 0.0)
        self.assertEqual(body.inertia, 0.0)

        # Test the force reset function
        body.force = np.array([[1, 2, 3]])
        body.torque = np.array([[4, 5, 6]])
        body.reset_forceTorque()
        assert_array_equal(body.force, np.array([0, 0, 0]))
        assert_array_equal(body.torque, np.array([0, 0, 0]))

    def test_addForce(self) -> None:
        body = Body()
        assert_array_equal(body.force, np.array([0, 0, 0]))
        assert_array_equal(body.torque, np.array([0, 0, 0]))

        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])

        body.add_Forces(vec1)
        assert_array_equal(body.force, vec1)
        assert_array_equal(body.torque, np.array([0, 0, 0]))

        body.add_Forces(vec2)
        assert_array_equal(body.force, vec1 + vec2)
        assert_array_equal(body.torque, np.array([0, 0, 0]))

    def test_addTorque(self) -> None:
        body = Body()
        assert_array_equal(body.force, np.array([0, 0, 0]))
        assert_array_equal(body.torque, np.array([0, 0, 0]))

        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])

        body.add_Torques(vec1)
        assert_array_equal(body.force, np.array([0, 0, 0]))
        assert_array_equal(body.torque, vec1)

        body.add_Torques(vec2)
        assert_array_equal(body.force, np.array([0, 0, 0]))
        assert_array_equal(body.torque, vec1 + vec2)

    def test_addForceTorques(self) -> None:
        body = Body()
        assert_array_equal(body.force, np.array([0, 0, 0]))
        assert_array_equal(body.torque, np.array([0, 0, 0]))

        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])

        body.add_Forces(vec1)
        body.add_Torques(vec2)
        assert_array_equal(body.force, vec1)
        assert_array_equal(body.torque, vec2)

        body.add_Forces(vec2)
        body.add_Torques(vec1)
        assert_array_equal(body.force, vec1 + vec2)
        assert_array_equal(body.torque, vec1 + vec2)


# ------------------------------------------------------------------------------------------------ #
#                                                                                  TEST_QUATERNION #
# ------------------------------------------------------------------------------------------------ #
class test_quaternion(TestCase):
    def test_default_initialization(self) -> None:
        q = Quaternion()
        self.assertEqual(q.a, np.float64(1.0))
        self.assertEqual(q.b, np.float64(0.0))
        self.assertEqual(q.c, np.float64(0.0))
        self.assertEqual(q.d, np.float64(0.0))

    def test_custom_initialization(self) -> None:
        q = Quaternion(np.array([np.float64(2.0), np.float64(3.0), np.float64(4.0), np.float64(5.0)]))
        self.assertEqual(q.a, np.float64(2.0))
        self.assertEqual(q.b, np.float64(3.0))
        self.assertEqual(q.c, np.float64(4.0))
        self.assertEqual(q.d, np.float64(5.0))

    def test_negative_initialization(self) -> None:
        q = Quaternion(np.array([np.float64(-1.0), np.float64(-2.0), np.float64(-3.0), np.float64(-4.0)]))
        self.assertEqual(q.a, np.float64(-1.0))
        self.assertEqual(q.b, np.float64(-2.0))
        self.assertEqual(q.c, np.float64(-3.0))
        self.assertEqual(q.d, np.float64(-4.0))

    def test_multiplication_identity(self) -> None:
        q1 = Quaternion()
        q2 = Quaternion(np.array([np.float64(-1.0), np.float64(-2.0), np.float64(-3.0), np.float64(-4.0)]))
        result = q1 * q2
        self.assertEqual(result, q2)

    def test_multiplication_custom(self) -> None:
        q1 = Quaternion(components=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        q2 = Quaternion(components=np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        result = q1 * q2
        # Expected results based on quaternion multiplication formula
        self.assertEqual(result.components[0], np.float64(-36.0))
        self.assertEqual(result.components[1], np.float64(6.0))
        self.assertEqual(result.components[2], np.float64(12.0))
        self.assertEqual(result.components[3], np.float64(12.0))

    def test_conjugate_positive(self) -> None:
        q = Quaternion(components=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        result = q.conjugate()
        self.assertEqual(result.components[0], np.float64(1.0))
        self.assertEqual(result.components[1], np.float64(-2.0))
        self.assertEqual(result.components[2], np.float64(-3.0))
        self.assertEqual(result.components[3], np.float64(-4.0))

    def test_conjugate_negative(self) -> None:
        q = Quaternion(components=np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64))
        result = q.conjugate()
        self.assertEqual(result.components[0], np.float64(-1.0))
        self.assertEqual(result.components[1], np.float64(2.0))
        self.assertEqual(result.components[2], np.float64(3.0))
        self.assertEqual(result.components[3], np.float64(4.0))

    def test_norm(self) -> None:
        q = Quaternion(components=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        assert_almost_equal(q.norm(), np.float64(np.sqrt(30)))

    def test_normalize(self) -> None:
        q = Quaternion(components=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        norm_q = q.norm()
        a_norm = 1.0 / norm_q
        b_norm = 2.0 / norm_q
        c_norm = 3.0 / norm_q
        d_norm = 4.0 / norm_q
        q.normalize()
        assert_almost_equal(q.a, a_norm)
        assert_almost_equal(q.b, b_norm)
        assert_almost_equal(q.c, c_norm)
        assert_almost_equal(q.d, d_norm)

    def test_inverse(self) -> None:
        q = Quaternion(components=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        qinverse = q.inverse()

        # Manually calculating the inverse
        a, b, c, d   = q.components
        norm_squared = a**2 + b**2 + c**2 + d**2
        inverse_quaternion = np.array([a, -b, -c, -d]) / norm_squared
        qInvTest           = Quaternion(components=inverse_quaternion)

        # Testing
        self.assertEqual(qinverse, qInvTest)

    def test_division_by_scalar(self) -> None:
        q = Quaternion(components=np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64))
        result = q / np.float64(2.0)
        self.assertEqual(result.components[0], np.float64(1.0))
        self.assertEqual(result.components[1], np.float64(2.0))
        self.assertEqual(result.components[2], np.float64(3.0))
        self.assertEqual(result.components[3], np.float64(4.0))

    def test_division_by_zero(self) -> None:
        q = Quaternion(components=np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64))
        with self.assertRaises(ZeroDivisionError):
            q / np.float64(0.0)

    def test_convert2AxisAngle(self) -> None:
        q = Quaternion(components=np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0, 0], dtype=np.float64))
        axisAngle = q.conv_2axisAngle()
        assert_almost_equal(axisAngle.axis, np.array([1, 0, 0]))
        assert_almost_equal(axisAngle.angle, np.pi / 2.0)

    def test_convert2AxisAngleNonUnit(self) -> None:
        ang = np.pi / 2.0
        axis = np.array([1, 0, 0])
        scale = random()

        q = Quaternion(components=scale * np.array([np.cos(ang / 2), *axis * np.sin(ang / 2)]))
        axisAngle = q.conv_2axisAngle()
        assert_almost_equal(axisAngle.axis, np.array([1, 0, 0]))
        assert_almost_equal(axisAngle.angle, np.pi / 2.0)


# ------------------------------------------------------------------------------------------------ #
#                                                                                        AXISANGLE #
# ------------------------------------------------------------------------------------------------ #
class test_AxisAngle(TestCase):
    def test_default_initialization(self) -> None:
        axisAngle = AxisAngle()
        assert_array_equal(axisAngle.axis, np.array([1, 0, 0]))
        self.assertEqual(axisAngle.angle, 0.0)

    def test_custom_initialization(self) -> None:
        axisAngle = AxisAngle(axis=np.array([5, 1, 9]), angle=np.pi / 2.0)
        assert_array_equal(axisAngle.axis, np.array([5, 1, 9]) / np.linalg.norm([5, 1, 9]))
        self.assertEqual(axisAngle.angle, np.pi / 2.0)

    def test_negative_initialization(self) -> None:
        axisAngle = AxisAngle(axis=np.array([-5, -1, -9]), angle=np.pi / 2.0)
        assert_array_equal(axisAngle.axis, np.array([-5, -1, -9]) / np.linalg.norm([-5, -1, -9]))
        self.assertEqual(axisAngle.angle, np.pi / 2.0)

    def test_conversion_to_quaternion(self) -> None:
        axisAngle = AxisAngle(axis=np.array([5, 1, 9]), angle=np.pi / 2.0)
        quaternion = axisAngle.conv_2quaternion()
        assert_almost_equal(quaternion.a, np.cos(np.pi / 4.0))
        assert_almost_equal(quaternion.b, np.sin(np.pi / 4.0) * 5 / np.sqrt(107))
        assert_almost_equal(quaternion.c, np.sin(np.pi / 4.0) * 1 / np.sqrt(107))
        assert_almost_equal(quaternion.d, np.sin(np.pi / 4.0) * 9 / np.sqrt(107))

    def test_conversion_to_quaternion_2(self) -> None:
        axis = np.array([random(), random(), random()])
        ang  = random()
        norm = np.linalg.norm(axis)

        axisAngle = AxisAngle(axis=axis, angle=ang)
        quaternion = axisAngle.conv_2quaternion()

        assert_almost_equal(quaternion.a, np.cos(ang / 2.0))
        assert_almost_equal(quaternion.b, np.sin(ang / 2.0) * axis[0] / norm)
        assert_almost_equal(quaternion.c, np.sin(ang / 2.0) * axis[1] / norm)
        assert_almost_equal(quaternion.d, np.sin(ang / 2.0) * axis[2] / norm)
