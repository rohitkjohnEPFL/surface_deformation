from yadeGrid import Body, Quaternion
from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal


# ------------------------------------------------------------------------------------------------ #
#                                                                                        TEST_BODY #
# ------------------------------------------------------------------------------------------------ #
class test_body(TestCase):
    def test_body(self):
        body = Body()
        # test states are initialized correctly to 0
        assert_array_equal(body.pos, np.array([[0, 0, 0]]))
        assert_array_equal(body.vel, np.array([[0, 0, 0]]))
        self.assertEqual(body.ori, Quaternion())
        assert_array_equal(body.angVel, np.array([[0, 0, 0]]))

        # Test the forces are initialized correctly to 0
        assert_array_equal(body.force, np.array([[0, 0, 0]]))
        assert_array_equal(body.torque, np.array([[0, 0, 0]]))

        # Test the constants are initialized correctly to 0
        self.assertEqual(body.mass, 0.0)
        self.assertEqual(body.inertia, 0.0)

        # Test the force reset function
        body.force = np.array([[1, 2, 3]])
        body.torque = np.array([[4, 5, 6]])
        body.reset_forceTorque()
        assert_array_equal(body.force, np.array([[0, 0, 0]]))
        assert_array_equal(body.torque, np.array([[0, 0, 0]]))


# ------------------------------------------------------------------------------------------------ #
#                                                                                  TEST_QUATERNION #
# ------------------------------------------------------------------------------------------------ #
class test_quaternion(TestCase):
    def test_default_initialization(self):
        q = Quaternion()
        self.assertEqual(q.a, np.float64(1.0))
        self.assertEqual(q.b, np.float64(0.0))
        self.assertEqual(q.c, np.float64(0.0))
        self.assertEqual(q.d, np.float64(0.0))

    def test_custom_initialization(self):
        q = Quaternion(np.array([np.float64(2.0), np.float64(3.0), np.float64(4.0), np.float64(5.0)]))
        self.assertEqual(q.a, np.float64(2.0))
        self.assertEqual(q.b, np.float64(3.0))
        self.assertEqual(q.c, np.float64(4.0))
        self.assertEqual(q.d, np.float64(5.0))

    def test_negative_initialization(self):
        q = Quaternion(np.array([np.float64(-1.0), np.float64(-2.0), np.float64(-3.0), np.float64(-4.0)]))
        self.assertEqual(q.a, np.float64(-1.0))
        self.assertEqual(q.b, np.float64(-2.0))
        self.assertEqual(q.c, np.float64(-3.0))
        self.assertEqual(q.d, np.float64(-4.0))

    def test_multiplication_identity(self):
        q1 = Quaternion()
        q2 = Quaternion(np.array([np.float64(-1.0), np.float64(-2.0), np.float64(-3.0), np.float64(-4.0)]))
        result = q1 * q2
        self.assertEqual(result, q2)

    def test_multiplication_custom(self):
        q1 = Quaternion(components=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        q2 = Quaternion(components=np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        result = q1 * q2
        # Expected results based on quaternion multiplication formula
        self.assertEqual(result.components[0], np.float64(-36.0))
        self.assertEqual(result.components[1], np.float64(6.0))
        self.assertEqual(result.components[2], np.float64(12.0))
        self.assertEqual(result.components[3], np.float64(12.0))

    def test_conjugate_positive(self):
        q = Quaternion(components=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        result = q.conjugate()
        self.assertEqual(result.components[0], np.float64(1.0))
        self.assertEqual(result.components[1], np.float64(-2.0))
        self.assertEqual(result.components[2], np.float64(-3.0))
        self.assertEqual(result.components[3], np.float64(-4.0))

    def test_conjugate_negative(self):
        q = Quaternion(components=np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64))
        result = q.conjugate()
        self.assertEqual(result.components[0], np.float64(-1.0))
        self.assertEqual(result.components[1], np.float64(2.0))
        self.assertEqual(result.components[2], np.float64(3.0))
        self.assertEqual(result.components[3], np.float64(4.0))

    def test_norm(self):
        q = Quaternion(components=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        assert_almost_equal(q.norm(), np.float64(np.sqrt(30)))

    def test_normalize(self):
        q = Quaternion(components=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        norm_q = q.norm()
        normalized_q = q.normalize()
        assert_almost_equal(normalized_q.components[0], np.float64(q.components[0] / norm_q))
        assert_almost_equal(normalized_q.components[1], np.float64(q.components[1] / norm_q))
        assert_almost_equal(normalized_q.components[2], np.float64(q.components[2] / norm_q))
        assert_almost_equal(normalized_q.components[3], np.float64(q.components[3] / norm_q))

    def test_division_by_scalar(self):
        q = Quaternion(components=np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64))
        result = q / np.float64(2.0)
        self.assertEqual(result.components[0], np.float64(1.0))
        self.assertEqual(result.components[1], np.float64(2.0))
        self.assertEqual(result.components[2], np.float64(3.0))
        self.assertEqual(result.components[3], np.float64(4.0))

    def test_division_by_zero(self):
        q = Quaternion(components=np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64))
        with self.assertRaises(ZeroDivisionError):
            q / np.float64(0.0)
