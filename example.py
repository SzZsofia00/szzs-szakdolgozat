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

#---- no noise -----#
so = SolveODE(diff_eq,time,init,step_size)
mtx = so.get_matrix_with_noise(method)
t = so.create_time_points()

pm = PysindyFunctions(mtx,t,threshold=0.02)
model = pm.model_fit()
print("Equation of predicted model")
pm.print_model_equations(model)

#----- noisy data ----#
mtx_noisy = so.get_matrix_with_noise(method, True)

pm_noisy = PysindyFunctions(mtx_noisy,t,threshold=0.02)
model_noisy = pm_noisy.model_fit()
print("Equation of predicted model from noisy data")
pm_noisy.print_model_equations(model_noisy)