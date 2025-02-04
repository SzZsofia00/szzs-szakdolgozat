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
        return [dxdt]

    def logistic_growth(self,t,x):
        if len(x) != 1:
            raise DimensionError("Dimension error: The length of x must be 1.")

        dxdt = 0.2 * x * (8 - x)
        return [dxdt]

    def linear2d(self,t,xy):
        if len(xy) != 2:
            raise DimensionError("Dimension error: The length of xy must be 2.")

        x,y = xy
        dxdt = -0.1 * x + 2 * y
        dydt = -2 * x -0.1 * y
        return [dxdt, dydt]

    def lotka_volterra(self,t,xy,alpha=2/3,beta=4/3,gamma=1,delta=1):
        if len(xy) != 2:
            raise DimensionError("Dimension error: The length of xy must be 2.")

        x,y = xy
        dxdt = alpha * x - beta * x * y
        dydt = - gamma * y + delta * x * y
        return [dxdt, dydt]

    def sis_model(self,t,xy,beta=0.5,gamma=0.12):
        if len(xy) != 2:
            raise DimensionError("Dimension error: The length of xy must be 2.")

        x,y = xy
        dxdt = - beta * x * y + gamma * y
        dydt = beta * x * y - gamma * y
        return [dxdt,dydt]

    def linear3d(self,t,xyz):
        if len(xyz) != 3:
            raise DimensionError("Dimension error: The length of xyz must be 3.")

        x,y,z = xyz
        dxdt = -0.1 * x - 2 * y
        dydt = 2 * x - 0.1 * y
        dzdt = -0.3 * z
        return [dxdt, dydt, dzdt]

    def lorenz(self,t,xyz,sigma=10,rho=28,beta=8/3):
        if len(xyz) != 3:
            raise DimensionError("Dimension error: The length of xyz must be 3.")

        x,y,z = xyz
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    def sir_model(self,t,xyz,beta=0.5,gamma=0.12):
        if len(xyz) != 3:
            raise DimensionError("Dimension error: The length of xyz must be 3.")

        #S: fogékony, I: fertőzött, R: felépült
        #beta: transmission rate, gamma: recovery rate
        x,y,z = xyz
        dxdt = - beta * x * y
        dydt = beta * x * y - gamma * y
        dzdt = gamma * y
        return [dxdt,dydt,dzdt]