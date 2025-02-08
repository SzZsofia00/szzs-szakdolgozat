from solve_methods import *
from differential_equations import *
from pysindy_methods import *

init = [1.0]
time = [0,5]
step_size = 0.00001
method = 'euler'

#Differenciálegyenlet
e = ExampleDifferentialEquations()
diff_eq = e.logistic_growth

so = SolveODE(diff_eq,time,init,step_size)
mtx = so.get_matrix_with_noise(method)
t = so.create_time_points()

pm = PysindyFunctions(mtx,t,threshold=0.2)
pm.print_model_equations()