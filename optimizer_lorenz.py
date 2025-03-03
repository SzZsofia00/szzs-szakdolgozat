#generate data with lorenz
# give the data for the optimizer

# i need to make two plot
# one where i plot the true function and the predicted one by x, y and z axis
# another plot: 3d plot and beta

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

import itertools

from differential_equations import *
from solve_methods import *
from symbol_creation import *

init = [1,1,1]
time = [0,40]
step_size = 1
lorenz = ExampleDifferentialEquations().lorenz
np.random.seed(42)

so = SolveODE(lorenz,time,init,step_size)
t = so.create_time_points().reshape(-1,1)
data = so.generate_data()
noise = so.generate_noise(scale=0.1)

#most megvannak az adatpontok
f_noisy = data + noise

features = PolynomialFeatures(3)
f = features.fit_transform(f_noisy.T)
features.get_feature_names_out(input_features=["x","y","z"])

models = []
for i in range(len(f_noisy)):
    model = LinearRegression().fit(f,data[i,:].reshape(-1,1))
    models.append(model)

pred_model = []
for i in models:
    pred = i.predict(f)
    pred_model.append(pred)

print(pred_model)

plt.scatter(t,data[0],label='Data',color='grey')
plt.plot(t,pred_model[0],label='LLS',color='blue')
plt.show()
