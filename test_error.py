from differential_equations import *
from application import *
import pandas as pd

## Run test for every ODE and calculate the squared deviation for every possible combinations of the num methods. ##

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
    "linear2d": ExampleDifferentialEquations().linear2d,
    "lotka_volterra": ExampleDifferentialEquations().linear2d,
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

methodsS = ["euler","midpoint_euler","RK3","RK4"]
methodsNM = ["euler","midpoint_euler","RK3","RK4"]

results = []
num = 0

for eq_name,eq_func in equations.items():
    for mS in methodsS:
        for mNM in methodsNM:

            parameters["diff_eq"] = eq_func
            parameters["methodSy"] = mS
            parameters["methodNM"] = mNM
            if eq_name == "exponential_growth" or eq_name == "logistic_growth":
                parameters["init"] = [1]
            if eq_name == "linear2d" or eq_name == "lotka_volterra":
                parameters["init"] = [1,1]
            elif eq_name == "linear3d" or eq_name == "lorenz" or eq_name == "rossler" or eq_name == "chua_circuit":
                parameters["init"] = [1, 1, 1]

            apl = Application(parameters)
            model = apl.fit_sindy_model()
            sq_dev = apl.squared_deviation(model)

            results.append({
                "Equation": eq_name,
                "Method Sindy": mS,
                "Method NM": mNM,
                "Squared deviation": sq_dev
            })

df = pd.DataFrame(results)
df.to_csv("res.csv", index=False)
print("Results saved to results.csv")