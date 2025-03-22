import numpy as np

class NumericalMethods:
  def __init__(self,func,tn:np.array,xn:np.array,h:float):
    """
    Contains some numerical methods
    :param func: The differential equation
    :param np.array tn: Time points
    :param np.array xn: Initial condition
    :param float h: Step size
    """
    self.func = func
    self.tn = tn
    self.xn = np.array(xn)
    self.h = h

  def step_with(self, method:str) -> np.ndarray:
    return self.xn + self.h * getattr(self, method)()

  def euler(self) -> np.ndarray:
    """
    Euler method
    """
    # return self.xn + self.h * np.array(self.func(self.tn,self.xn))
    return np.array(self.func(self.tn, self.xn))

  def midpoint_euler(self) -> np.ndarray:
    """
    Midpoint Euler method
    """
    k1 = np.array(self.func(self.tn,self.xn))
    k2 = np.array(self.func(self.tn + self.h/2,self.xn + self.h/2 * k1))
    # return self.xn + self.h * k2
    return k2

  def RK3(self) -> np.ndarray:
    """
    Runge Kutta 3 method
    """
    k1 = np.array(self.func(self.tn,self.xn))
    k2 = np.array(self.func(self.tn + self.h/2, self.xn + self.h/2 * k1))
    k3 = np.array(self.func(self.tn + self.h, self.xn + self.h*(-k1 + 2 * k2)))
    # return self.xn + self.h * (k1/6 + k2 * 2/3 + k3/6)
    return k1/6 + k2 * 2/3 + k3/6

  def RK4(self) -> np.ndarray:
    """
    Runge Kutta 4 method
    """
    k1 = np.array(self.func(self.tn,self.xn))
    k2 = np.array(self.func(self.tn + self.h/2,self.xn + self.h/2 * k1))
    k3 = np.array(self.func(self.tn + self.h/2,self.xn + self.h/2 * k2))
    k4 = np.array(self.func(self.tn + self.h,self.xn + self.h * k3))
    # return self.xn + self.h * (k1/6 + k2/3 + k3/3 + k4/6)
    return k1/6 + k2/3 + k3/3 + k4/6