from differential_equations import *
from application import *
import pandas as pd

import timeit

## Run test for every ODE and calculate the squared deviation for every possible combinations of the num methods. ##
start = timeit.default_timer()

parameters = {
    "diff_eq": None,
    "init": None,
    "time": [0,5],
    "step_size": 0.00001,
    "methodSy": None,
    "methodNM": None,
    "be_noise": False,
    "degree": 3,
    "threshold": 0.02
}

equations = {
    "exponential_growth": ExampleDifferentialEquations().exponential_growth,
    "logistic_growth": ExampleDifferentialEquations().logistic_growth,
    "lotka_volterra": ExampleDifferentialEquations().lotka_volterra,
    "linear2d": ExampleDifferentialEquations().linear2d,
    "linear3d": ExampleDifferentialEquations().linear3d,
    "lorenz": ExampleDifferentialEquations().lorenz,
    "rossler": ExampleDifferentialEquations().rossler,
    "chua_circuit": ExampleDifferentialEquations().chua_circuit
}

chaotic_eq = {
    "lorenz": ExampleDifferentialEquations().lorenz,
    "rossler": ExampleDifferentialEquations().rossler,
    "chua_circuit": ExampleDifferentialEquations().chua_circuit
}

methodsS = ["euler","midpoint_euler","RK3","RK4"] #method optimizer
methodsNM = ["euler","midpoint_euler","RK3","RK4"]
optimizer = ["lls","ridge","lasso","stlsq"] #gsls-t kinyirja a alacsony h

results = []
num = 0

for eq_name,eq_func in equations.items():
    for mS in methodsS:
        for mNM in methodsNM:

            start = timeit.default_timer()

            parameters["diff_eq"] = eq_func
            parameters["methodSy"] = mS
            parameters["methodNM"] = mNM
            if eq_name == "exponential_growth" or eq_name == "logistic_growth":
                parameters["init"] = [1]
            if eq_name == "linear2d" or eq_name == "lotka_volterra":
                parameters["init"] = [1,1]
            elif eq_name == "linear3d" or eq_name == "lorenz" or eq_name == "rossler" or eq_name == "chua_circuit":
                parameters["init"] = [1, 1, 1]

            row = {
                "Equation": eq_name,
                "Method Sindy": mS,
                "Method NM": mNM
            }

            for op in optimizer:
                print(eq_name,mS,mNM,op)
                dffc = DataframeForCoefficients(parameters)
                sq_dev = dffc.squared_deviation(op)

                row[f"Squared deviation {op.upper()}"] = sq_dev

                stop = timeit.default_timer()
                print('Time: ', stop - start)

            results.append(row)
            print(num)
            num+=1

df = pd.DataFrame(results)
df.to_csv("res.csv", index=False)
print("Results saved to results.csv")

stop = timeit.default_timer()
print('Time: ', stop - start)