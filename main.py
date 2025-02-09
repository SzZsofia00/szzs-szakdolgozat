from sympy import symbols, Pow

from application import *
from differential_equations import *

import timeit

parameters = {
    "diff_eq": ExampleDifferentialEquations().logistic_growth,
    "init": [1],
    "time": [0,10],
    "step_size": 0.00001,
    "methodSy": "midpoint_euler",
    "methodNM": "midpoint_euler",
    "be_noise": False,
    "degree": 3,
    "threshold": 0.02
}

# Application:
#   - num_method_coefficients
#   - ha az elozo valtozik valszeg num_method_solution is
#   - create_header-for_Df
#   - squared deviation


apl = Application(parameters)
model = apl.fit_sindy_model()
apl.create_table_of_solutions(model)
print(apl.squared_deviation(model))

### Runtime check ###
#start = timeit.default_timer()
#
# stop = timeit.default_timer()
# print('Time: ', stop - start)