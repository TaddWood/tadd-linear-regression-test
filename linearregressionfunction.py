import math


class LinearRegression(object):
    """
    Summary
        Linear regression of y = ax + b
    Usage
        real, real, real = linreg(list, list)
    Returns coefficients to the regression line "y=ax+b" from x[] and y[], and R^2 Value

    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = len(self.X)
        self.Sx = 0.0
        self.Sy = 0.0
        self.Sxx = 0.0
        self.Syy = 0.0
        self.Sxy = 0.0
        if len(self.X) != len(self.Y):
            raise ValueError('unequal length')
        if len(self.X) == 0 or len(self.Y) == 0:
            raise ValueError('empty list')
        self.sums()

    def sums(self):
        for x, y in zip(self.X, self.Y):
            self.Sx += x
            self.Sy += y
            self.Sxx += math.pow(x, 2)
            self.Syy += math.pow(y, 2)
            self.Sxy += x*y

    @property
    def det(self):
        return self.Sxx * self.N - math.pow(self.Sx, 2)

    @property
    def a(self):
        return (self.Sxy * self.N - self.Sy * self.Sx) / self.det

    @property
    def b(self):
        return (self.Sxx * self.Sy - self.Sx * self.Sxy) / self.det

    @property
    def RR(self):
        mean_error = residual = 0.0
        for x, y in zip(self.X, self.Y):
            mean_error += math.pow(y - self.Sy / self.N, 2)
            residual += math.pow(y - self.a * x - self.b, 2)
        return 1 - residual / mean_error
