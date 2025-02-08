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

# Melyik függvények amikben van vmi ami nem nagyon okés:
#   - applicationben nem tetszik a num_method_coefficients se


apl = Application(parameters)
model = apl.fit_sindy_model()
# apl.create_table_of_solutions(model)
# print(apl.squared_deviation(model))

lst,fv,coeff = apl.num_method_coefficients(model)
x = symbols('x')
# print("lst")
print(lst)
print(type(lst))
print(lst[0])
print(type(lst[0]))
# print("\n")

features = sorted({term for term in lst[0].atoms(Pow) if term.has(x)}, key=lambda t: t.as_base_exp()[1])
print(features)


# print("fv")
# print(fv)
# print("\n")
# print("coeff")
# print(coeff)
# print("\n")

### Runtime check ###
#start = timeit.default_timer()
#
# stop = timeit.default_timer()
# print('Time: ', stop - start)