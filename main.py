from application import *
from differential_equations import *

parameters = {
    "diff_eq": ExampleDifferentialEquations().linear3d,
    "init": [1,1,1],
    "time": [0,5],
    "step_size": 0.00001,
    "methodSy": "euler",
    "methodNM": "euler",
    "be_noise": False,
    "threshold": 0.02
}

apl = Application(parameters)
model = apl.sindy_model()
apl.create_table_of_solutions(model)
apl.squared_deviation(model)