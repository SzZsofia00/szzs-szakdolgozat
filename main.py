from application import *
from differential_equations import *

import timeit

parameters = {
    "diff_eq": ExampleDifferentialEquations().lorenz,
    "init": [1,1,1],
    "time": [0,40],
    "step_size": 0.001,
    "methodSy": "midpoint_euler",
    "methodNM": "euler",
    "be_noise": False,
    "degree": 3,
    "threshold": 0.02,
    "optimizer": "lls"
}

apl = DataframeForCoefficients(parameters)
apl.create_table_of_solutions(parameters["optimizer"])
print(apl.squared_deviation(parameters["optimizer"]))

### Runtime check ###
#start = timeit.default_timer()
#
# stop = timeit.default_timer()
# print('Time: ', stop - start)