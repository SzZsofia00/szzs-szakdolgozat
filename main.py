import numpy as np

from solve_methods import *
from differential_equations import *
from pysindy_methods import *

#For Lorenz
# init = [0.0,1.0,1.0]
# time = [0,1]
# step_size = 0.02

init = [1]
time = [0,5]
step_size = 0.1

e = ExampleDifferentialEquations()
diff_eq = e.logistic_growth

so = SolveODE(diff_eq,time,init,step_size)

mtx = so.get_matrix_with_noise('euler')
t = so.create_time_points()

pm = PysindyFunctions(mtx,t,threshold=0.2)
pm.print_model_equations()

fv = pm.sympify_feature()
coef = pm.get_coefficients()
# rounded_coef = np.round(coef,4)
sol = coef * fv
print(sol)

