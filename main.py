from sympy import symbols, Pow

from application import *
from differential_equations import *

import timeit

parameters = {
    "diff_eq": ExampleDifferentialEquations().lorenz,
    "init": [1,1,1],
    "time": [0,5],
    "step_size": 0.00001,
    "methodSy": "euler",
    "methodNM": "euler",
    "be_noise": False,
    "degree": 3,
    "threshold": 0.02
}

apl = Application(parameters)
model = apl.fit_sindy_model()
apl.create_table_of_solutions(model)
print(apl.squared_deviation(model))

# print('sindy')
# print(apl.model_feature_vector(model))
# print(apl.model_coefficients(model))
# print(apl.model_solution(model))
# print("nm")
# print(apl.num_method_with_symbols())
# print(apl.num_method_feature_vector())
# print(apl.num_method_coefficients())
# print(apl.num_method_solution())

### Runtime check ###
#start = timeit.default_timer()
#
# stop = timeit.default_timer()
# print('Time: ', stop - start)