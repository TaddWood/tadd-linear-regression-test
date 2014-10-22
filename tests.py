__author__ = 'taddwoodCAN'

from linearregressionfunction import LinearRegression
import unittest


class LinearRegressionTests(unittest.TestCase):
    def setUp(self):
        self.X = [1, 2, 3, 4]
        self.Y = [357.14, 53.57, 48.78, 10.48]
        self.LR = LinearRegression(self.X, self.Y)

    def test_det(self):
        self.assertAlmostEqual(self.LR.det, 20.0, places = 2)

    def test_a(self):
        self.assertAlmostEqual(self.LR.a, -104.477, places = 2)

    def test_b(self):
        self.assertAlmostEqual(self.LR.b, 378.685, places = 2)

    def test_RR(self):
        self.assertAlmostEqual(self.LR.RR, 0.702499064, places = 2)

    def test_lists_length(self):
        with self.assertRaises(ValueError):
            LinearRegression([1, 2], [1, 2, 3])

    def test_empty_lists(self):
        with self.assertRaises(ValueError):
            LinearRegression([], [])

    def test_single_value_lists(self):
        self.assertTrue(LinearRegression([1], [2]))
