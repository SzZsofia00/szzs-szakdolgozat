from application import *
from differential_equations import *

parameters = {
    "diff_eq": ExampleDifferentialEquations().exponential_growth,
    "init": [1],
    "time": [0,5],
    "step_size": 0.00001,
    "methodSy": "RK3",
    "methodNM": "midpoint_euler",
    "be_noise": False,
    "threshold": 0.02
}

apl = Application(parameters)
model = apl.sindy_model()
# apl.create_table_of_solutions(model)
print(apl.squared_deviation(model))

