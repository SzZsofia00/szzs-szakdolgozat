from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error


class Functions:
    def __init__(self,x):
        self.x = x

    # def polynomial_dgr4(self):
    #     return 0.4 * self.x**4 + 0.8 * self.x**3 - 2 * self.x

    def polynomial(self):
        return 0.5 * self.x**7 - 3 * self.x**5 + 4 * self.x**3 - 0.2 * self.x

    def complex(self):
        return 0.5 * self.x**7 - 3 * self.x**5 + 2 * self.x**3 - self.x + 2 * np.sin(7 * self.x)

    def complex2(self):
        return 0.5 * self.x**7 - 3 * self.x**5 + 2 * self.x**3 - self.x + 3 * np.sin(2 * self.x)

    def poly(self):
        return 3 * self.x ** 2

    def linear(self):
        return 3 * self.x + 5

    def cubic(self):
        return 3 * self.x**3 - 2 * self.x**2 + 0.5 * self.x + 5

np.random.seed(42)

params = {
    "time": [-2,2],
    "number_of_points": 50,
    "scale": 2,     #scale = standard deviation
    "degree": 2,      #for features
    "method": "elastic",   # lls / ridge / lasso / elastic
    "alpha_ridge": 1.5,
    "alpha_lasso": 1.5,
    "alpha_elastic": 1.5
}

x = np.linspace(params["time"][0],params["time"][1],params["number_of_points"]).reshape(-1,1)
noise = np.random.normal(loc=0.0, scale=params["scale"], size=x.shape)
f_true = Functions(x).linear()

feature = PolynomialFeatures(params["degree"])
x_feat = feature.fit_transform(x)
get_ft = feature.get_feature_names_out(input_features=["x"])

# f_noisy = f_true + noise
f_noisy = deepcopy(f_true)
f_noisy[10] += 20

x_train,x_test,y_train,y_test = train_test_split(x_feat,f_noisy)
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_feat)

rmse = root_mean_squared_error(f_true,y_pred)
print(rmse)

# plot
plt.figure(figsize=(8, 6))
plt.scatter(x, f_noisy, color="red", label="Noisy data")
plt.plot(x, f_true, color="grey", linestyle="dashed", label="True function")
plt.plot(x, y_pred, color="blue", label="Prediction")

plt.legend()
plt.show()