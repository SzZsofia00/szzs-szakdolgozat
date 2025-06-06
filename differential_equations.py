from exceptions import *

class ExampleDifferentialEquations:
    """
    Collection of ordinary differential equations.
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

    def lotka_volterra(self,t,xy,alpha=1.1,beta=0.4,gamma=0.4,delta=0.1):
        if len(xy) != 2:
            raise DimensionError("Dimension error: The length of xy must be 2.")

        x,y = xy
        dxdt = alpha * x - beta * x * y
        dydt = - gamma * y + delta * x * y
        return [dxdt, dydt]

    def linear2d(self,t,xy):
        if len(xy) != 2:
            raise DimensionError("Dimension error: The length of xy must be 2.")

        x,y = xy
        dxdt = -0.1 * x + 2 * y
        dydt = -2 * x -0.1 * y
        return [dxdt, dydt]

    def linear3d(self,t,xyz):
        if len(xyz) != 3:
            raise DimensionError("Dimension error: The length of xyz must be 3.")

        x,y,z = xyz
        dxdt = -0.1 * x - 2 * y
        dydt = 2 * x - 0.1 * y
        dzdt = -0.3 * z
        return [dxdt, dydt, dzdt]

    #Chaotic model

    def lorenz(self,t,xyz,sigma=10,rho=28,beta=8/3):
        if len(xyz) != 3:
            raise DimensionError("Dimension error: The length of xyz must be 3.")

        x,y,z = xyz
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    def rossler(self,t,xyz,a=0.2,b=0.2,c=5.7): #ezek Rossler szamai, azota tipikusan rendre ezeket hasznaltak: 0.1,0.1,14
        if len(xyz) != 3:
            raise DimensionError("Dimension error: The length of xyz must be 3.")

        x,y,z = xyz
        dxdt = - y - z
        dydt = x + a * y
        dzdt = b + z * (x - c)
        return [dxdt, dydt, dzdt]

    def chua_circuit(self,t,xyz,alpha=10.92,beta=14):
        if len(xyz) != 3:
            raise DimensionError("Dimension error: The length of xyz must be 3.")

        x,y,z = xyz
        f = 1/16 * x**3 - 1/3 * x
        dxdt = alpha * (y - f)
        dydt = x - y + z
        dzdt = - beta * y
        return [dxdt, dydt, dzdt]