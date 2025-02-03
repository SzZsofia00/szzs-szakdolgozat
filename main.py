import numpy as np
import pandas as pd
from tabulate import tabulate

from solve_methods import *
from differential_equations import *
from pysindy_methods import *

#Kapott feltételek
init = [0.0,1.0,1.0]
time = [0,5]
step_size = 0.00001

methodSy = 'euler'
be_noise = False
methodNM = 'euler'

diff_eq = ExampleDifferentialEquations().lorenz
threshold = 0.02

#Megoldom az ODE-t és generálok belőle adatot
so = SolveODE(diff_eq,time,init,step_size)
mtx = so.get_matrix_with_noise(methodSy,be_noise=be_noise)
t = so.create_time_points()
pm = PysindyFunctions(mtx,t,threshold=threshold)

#Egyik módszer
fv = pm.sympify_feature()
coef = pm.get_coefficients()
sol = coef * fv
print(sol)


#Másik módszer
symb_init = pm.cr.create_symbols()
# print(diff_eq(0,symb_init))
nm = NumericalMethods(diff_eq,0,symb_init,step_size)
lst = [sp.expand(expr) for expr in getattr(nm,methodNM)()]
print(lst)


new = []
for l in lst:
    dct = l.as_coefficients_dict()
    row = []
    for f in fv:
        coeff = dct.get(f, 0)
        row.append(coeff)
    new.append(row)
new_array = np.array(new)

sol2 = new_array * fv
print(sol2)


############Pandas##################
fv_str = [str(expr) for expr in fv]
header = fv_str
df_sindy = pd.DataFrame(coef,
                  index=["sindy-vel dx","sindy-vel dy","sindy-vel dz"],
                  columns=header)
df_nm = pd.DataFrame(new_array,
                  index=["nm-vel dx","nm-vel dy","nm-vel dz"],
                  columns=header)
df = pd.concat([df_sindy,df_nm])

print(tabulate(df,headers=header))

##########COMPARISON#############
sq_dev = (coef.reshape(1, -1)[0]-new_array.reshape(1, -1)[0])**2
hossz = len(new_array.reshape(1, -1)[0])
summa = 0
for i in sq_dev:
  summa += i
print(summa/hossz)

