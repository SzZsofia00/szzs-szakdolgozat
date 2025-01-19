import numpy as np
from exceptions import *

class ExampleDifferentialEquations:
    """
    Collection of differential equations.
    """

    def exponential_growth(self,t,x):
        if len(x) != 1:
            raise DimensionError("Dimension error: The length of x must be 1.")

        dxdt = 0.5 * x
        return dxdt

    def logistic_growth(self,t,x):
        if len(x) != 1:
            raise DimensionError("Dimension error: The length of x must be 1.")

        dxdt = 0.2 * x * (8 - x)
        return dxdt

    def lorenz(self,t,xyz):
        if len(xyz) != 3:
            raise DimensionError("Dimension error: The length of xyz must be 3.")

        x,y,z = xyz
        dxdt = 10 * (y - x)
        dydt = x * (28 - z) - y
        dzdt = x * y - 8 / 3 * z
        return [dxdt, dydt, dzdt]

    def sin_cos(self,t,x):
        if len(x) != 1:
            raise DimensionError("Dimension error: The length of x must be 1.")

        dxdt = np.sin(t) + np.cos(x)
        return dxdt