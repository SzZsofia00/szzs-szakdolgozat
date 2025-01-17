from solve_methods import *
from examples import *
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

pm = PysindyFunctions(mtx,t,threshold=0.02)
model = pm.model_fit()
model.print()

coef = model.coefficients()
fn = model.get_feature_names()
symb = create_symbols(len(mtx))

print(coef)
print(fn)
print(symb)

var = create_variables(len(mtx))
print(var)

for i in fn:
    i.replace('^','**')
    print(i)

print(fn)

