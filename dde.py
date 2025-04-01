import ddeint
import itertools
import pysindy as ps

import numpy as np
from derivative import FiniteDifference
from mpmath import degree
from pysindy import PolynomialLibrary, STLSQ
from sklearn.preprocessing import PolynomialFeatures

from symbol_creation import *


def mackey_glass(x,t,mu=1,p=2,n=2):
    return -mu * x(t) + (p * x(t-1)) / (1 + (x(t-1))**n)

def history(t):
    return 0.5

num_of_points = 10

ts = np.linspace(1,num_of_points,num_of_points)
ys = ddeint.ddeint(mackey_glass,history,ts).flatten()
data = np.vstack([ys[-num_of_points+1:], ys[:num_of_points-1]])

print(ts)

symb = CreateSymbols(1).create_symbold_for_dde(2)
var = CreateSymbols(1).create_var_for_dde(2)
dgr = 2

fv = []
for d in range(dgr + 1):
    for i in itertools.combinations_with_replacement(symb,d):
        fv.append(sp.Mul(*i))

print(data.T)

model = ps.SINDy(differentiation_method=ps.FiniteDifference(order=2),
                 feature_library=ps.PolynomialLibrary(degree=dgr),
                 optimizer=ps.STLSQ(threshold=0.02),
                 feature_names=var)
model.fit(data.T,1)
model.print()
