import math


def linreg(X, Y):
    """
    Summary
        Linear regression of y = ax + b
    Usage
        real, real, real = linreg(list, list)
    Returns coefficients to the regression line "y=ax+b" from x[] and y[], and R^2 Value

    """
    if len(X) != len(Y):
        raise ValueError('unequal length')
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx += x
        Sy += y
        Sxx += math.pow(x, 2)
        Syy += math.pow(y, 2)
        Sxy += x*y
    det = Sxx * N - math.pow(Sx, 2)
    a, b = (Sxy * N - Sy * Sx) / det, (Sxx * Sy - Sx * Sxy) / det
    mean_error = residual = 0.0
    for x, y in zip(X, Y):
        mean_error += math.pow(y - Sy / N, 2)
        residual += math.pow(y - a * x - b, 2)
    RR = 1 - residual / mean_error
    return a, b, RR

if __name__=='__main__':
    #testing
    X = [1, 2, 3, 4]
    Y = [357.14, 53.57, 48.78, 10.48]
    print(linreg(X, Y))
    #should be:
    #Slope	Y-Int	R
    #-104.477	378.685	0.702499064
