from application import *
from differential_equations import *

parameters = {
    "diff_eq": ExampleDifferentialEquations().exponential_growth,
    "init": [1],
    "time": [0,5],
    "step_size": 0.00001,
    "methodSy": "RK4",
    "methodNM": "euler",
    "be_noise": False,
    "threshold": 0.02
}

# apl = Application(parameters)
# model = apl.sindy_model()
# # apl.create_table_of_solutions(model)
# print(apl.squared_deviation(model))

symb_init = [sp.Symbol('x')]
nm = NumericalMethods(parameters["diff_eq"], 0, symb_init, parameters["step_size"])
result = getattr(nm, parameters["methodNM"])()[0]
print(type(result))
print(result)
print([type(elem) for elem in result])

for i, expr in enumerate(result):
    print(f"Element {i}: Type={type(expr)}, Value={expr}")
    if isinstance(expr, np.ndarray):
        print(f"  └ Element {i} contains another NumPy array!")