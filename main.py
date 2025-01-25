import numpy as np

from solve_methods import *
from differential_equations import *
from pysindy_methods import *

#Kapott feltételek
init = [0.0,1.0,1.0]
time = [0,5]
step_size = 0.00001
method = 'euler'

#Differenciálegyenlet
e = ExampleDifferentialEquations()
diff_eq = e.lorenz

#Megoldom az ODE-t és generálok belőle adatot
so = SolveODE(diff_eq,time,init,step_size)
mtx = so.get_matrix_with_noise(method)
t = so.create_time_points()

#Modell illesztés
pm = PysindyFunctions(mtx,t,threshold=0.02)

#Egyik módszer
symb_init = pm.cr.create_symbols()
# print(diff_eq(0,symb_init))
nm = NumericalMethods(diff_eq,0,symb_init,step_size)
print(
    [sp.expand(expr) for expr in getattr(nm,method)()]
)

#Másik módszer
fv = pm.sympify_feature()
coef = pm.get_coefficients()
sol = coef * fv
print(sol)

