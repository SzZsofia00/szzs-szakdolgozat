from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

from differential_equations import *
from solve_methods import *
from symbol_creation import *

np.random.seed(42)

#initialize lorenz
init = [1,1,1]
time_fit = [0,40] #training data [0,30] and testing data [0,40]
step_size = 1
lorenz = ExampleDifferentialEquations().lorenz

# generate data for lorenz
so = SolveODE(lorenz,time_fit,init,step_size)
t = so.create_time_points().reshape(-1,1)
data = so.generate_data() #shape: (3,31) --> ([x,y,z],time_fit)

#features
features = PolynomialFeatures(3)
data_feat = features.fit_transform(data.T) #shape: (31,20) --> ([time_fit],features)
features.get_feature_names_out(input_features=["x","y","z"])

x_train,x_test,y_train,y_test = train_test_split(data_feat,data.T,test_size=0.25)
# reg = LinearRegression()
reg = Lasso(alpha=2)
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
# y_pred = reg.predict(data_feat)

print(y_pred)


fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = ["x", "y", "z"]
colors = ["r", "g", "b"]
length = len(y_pred)

for i in range(3):
    axes[i].plot(t[-length:], data[i,-length:], color=colors[i], label=f"True {labels[i]}")
    axes[i].plot(t[-length:], y_pred[:, i], color=colors[i], linestyle="dashed", label=f"Predicted {labels[i]}")
    axes[i].set_ylabel(labels[i])
    axes[i].legend()
    axes[i].grid()

axes[-1].set_xlabel("Time")
plt.show()


