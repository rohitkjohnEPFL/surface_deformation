from yadeGrid.vectorFunc import crossProduct, dotProduct, norm, normalise
from yadeGrid.yadeTypes import F64
from numpy import array
from unittest import TestCase
from numpy.testing import assert_array_equal, assert_almost_equal


# ------------------------------------------------------------------------------------------------ #
#                                                                                TEST_CROSSPRODUCT #
# ------------------------------------------------------------------------------------------------ #
class test_crossProduct(TestCase):
    def test_crossProduct(self) -> None:
        vec1 = array([1., 2., 3.])
        vec2 = array([4., 5., 6.])
        vec12ans = array([-3., 6., -3.])
        assert_array_equal(crossProduct(vec1, vec2), vec12ans)

        vecX = array([1., 0., 0.])
        vecY = array([0., 1., 0.])
        vecZ = array([0., 0., 1.])
        vec0 = array([0., 0., 0.])
        assert_array_equal(crossProduct(vecX, vecX), vec0)
        assert_array_equal(crossProduct(vecY, vecY), vec0)
        assert_array_equal(crossProduct(vecZ, vecZ), vec0)

        assert_array_equal(crossProduct(vecX, vecY), vecZ)
        assert_array_equal(crossProduct(vecY, vecZ), vecX)
        assert_array_equal(crossProduct(vecZ, vecX), vecY)

        assert_array_equal(crossProduct(vecY, vecX), -vecZ)
        assert_array_equal(crossProduct(vecZ, vecY), -vecX)
        assert_array_equal(crossProduct(vecX, vecZ), -vecY)


# ------------------------------------------------------------------------------------------------ #
#                                                                                  TEST_DOTPRODUCT #
# ------------------------------------------------------------------------------------------------ #
class test_dotProduct(TestCase):
    def test_dotProduct(self) -> None:
        vec1 = array([1., 2., 3.])
        vec2 = array([4., 5., 6.])
        assert_array_equal(dotProduct(vec1, vec2), 32.)

        vecX = array([1., 0., 0.])
        vecY = array([0., 1., 0.])
        vecZ = array([0., 0., 1.])
        assert_array_equal(dotProduct(vecX, vecX), 1.)
        assert_array_equal(dotProduct(vecY, vecY), 1.)
        assert_array_equal(dotProduct(vecZ, vecZ), 1.)

        assert_array_equal(dotProduct(vecX, vecY), 0.)
        assert_array_equal(dotProduct(vecY, vecZ), 0.)
        assert_array_equal(dotProduct(vecZ, vecX), 0.)


# ------------------------------------------------------------------------------------------------ #
#                                                                                        TEST_NORM #
# ------------------------------------------------------------------------------------------------ #
class test_norm(TestCase):
    def test_norm(self) -> None:
        vec1 = array([1., 2., 3.])
        assert_almost_equal(norm(vec1), F64(3.7416573867739413))

        vecX = array([1., 0., 0.])
        vecY = array([0., 1., 0.])
        vecZ = array([0., 0., 1.])
        assert_almost_equal(norm(vecX), F64(1.))
        assert_almost_equal(norm(vecY), F64(1.))
        assert_almost_equal(norm(vecZ), F64(1.))

        vec2 = array([1., 1., 0])
        vec3 = array([1., 1., 1.])
        assert_almost_equal(norm(vec2), F64(1.4142135623730951))
        assert_almost_equal(norm(vec3), F64(1.7320508075688772))


class test_normalise(TestCase):
    def test_normalise(self) -> None:
        vec1    = array([1., 2., 3.])
        vec1ans = array([0.2672612419124244, 0.5345224838248488, 0.8017837257372732])
        assert_array_equal(normalise(vec1), vec1ans)

    def test_withUnitVectors(self) -> None:
        vecx   = array([1., 0., 0.])
        vecy   = array([0., 1., 0.])
        vecz   = array([0., 0., 1.])
        assert_array_equal(normalise(vecx), vecx)
        assert_array_equal(normalise(vecy), vecy)
        assert_array_equal(normalise(vecz), vecz)

    def test_withZeroVector(self) -> None:
        vec0 = array([0., 0., 0.])
        with self.assertRaises(ZeroDivisionError):
            normalise(vec0)
