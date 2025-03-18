from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

class Functions:
    def __init__(self,x):
        self.x = x

    def complex(self):
        return 0.5 * self.x**7 - 3 * self.x**5 + 2 * self.x**3 - self.x + 2 * np.sin(7 * self.x)

    def poly(self):
        return 3 * self.x ** 2

    def linear(self):
        return 3 * self.x + 5

    def cubic(self):
        return 3 * self.x**3 - 2 * self.x**2 + 0.5 * self.x + 5

    def linear2(self):
        return 2/3 * self.x + 10/3

class Regression:
    def __init__(self,x,data,dgr):
        self.dgr = dgr
        self.data = data
        self.features = PolynomialFeatures(self.dgr)
        self.x = self.features.fit_transform(x)
        self.alpha_ridge = params["alpha_ridge"]
        self.alpha_lasso = params["alpha_lasso"]
        self.alpha_elast = params["alpha_elastic"]

    def get_features(self):
        return self.features.get_feature_names_out(input_features=["x"])

    def get_model(self,method):
        if method=='lls':
            # model = LinearRegression()
            model = LinearRegression(fit_intercept=False)
        elif method=='ridge':
            model = Ridge(alpha=self.alpha_ridge,fit_intercept=False)
        elif method=='lasso':
            model = Lasso(alpha=self.alpha_lasso,fit_intercept=False)
        elif method=='elastic':
            model = ElasticNet(alpha=self.alpha_elast,fit_intercept=False)
        model.fit(self.x, self.data) #(data,target)
        return model

    def get_coeff(self,method):
        if method=='ridge_formula':
            return np.linalg.pinv(self.x.T @ self.x + self.alpha_ridge * np.identity(self.x.shape[1])) @ self.x.T @ self.data
        model = self.get_model(method)
        c = model.coef_.flatten()
        # c[0] = model.intercept_[0]
        # c[0] = model.intercept_
        return c

    def lls(self):
        lls_model = self.get_model('lls')
        return lls_model.predict(self.x)

    def ridge(self):
        ridge_model = self.get_model('ridge')
        return ridge_model.predict(self.x)

    def ridge_formula(self):
        return self.x @ np.linalg.pinv(self.x.T @ self.x + self.alpha_ridge * np.identity(self.x.shape[1])) @ self.x.T @ self.data

    def lasso(self):
        lasso_model = self.get_model('lasso')
        return lasso_model.predict(self.x)

    def elastic(self):
        elastic_model = self.get_model('elastic')
        return elastic_model.predict(self.x)

np.random.seed(42)
params = {
    "time": [-2,2],
    "data_points": 5,
    "func_points": 100,
    "scale": 5,     #scale = standard deviation
    "degree": 3,      #for features
    "method": "lasso",   # lls / ridge / ridge_formula / lasso / elastic
    "alpha_ridge": 0.5,
    "alpha_lasso": 0.5,
    "alpha_elastic": 0.5
}

x = np.linspace(params["time"][0],params["time"][1],params["func_points"]).reshape(-1,1)
x_data = np.linspace(params["time"][0],params["time"][1],params["data_points"]).reshape(-1,1)

noise = np.random.normal(loc=0.0, scale=params["scale"], size=x.shape)
noise_data = np.random.normal(loc=0.0, scale=params["scale"], size=x_data.shape)

f_true = Functions(x).linear()
f_data = Functions(x_data).linear()

def calculate_regression(x,noisy,method):
    reg = Regression(x=x, data=noisy, dgr=params["degree"])
    reg_method = getattr(reg, method)()
    return reg,reg_method

def plot(x_full,x_data, f_full, noisy_full,noisy_data, method):
    reg,reg_method = calculate_regression(x_full,noisy_full,method)

    x_plot = np.linspace(1, 7, 100).reshape(-1, 1)  # Smooth range for plotting

    y_lls = reg.get_model("lls").predict(reg.features.transform(x_plot))
    y_ridge = reg.get_model("ridge").predict(reg.features.transform(x_plot))
    y_lasso = reg.get_model("lasso").predict(reg.features.transform(x_plot))

    # plt.scatter(x_full, noisy_full, color="red", label="Data (with Outliers)")
    plt.plot(x_plot, y_lls, label="LLS (should overfit)", linestyle="dashed")
    plt.plot(x_plot, y_ridge, label="Ridge (should smooth)")
    plt.plot(x_plot, y_lasso, label="Lasso (should shrink coefficients)")

    plt.legend()
    plt.show()

    rmse = root_mean_squared_error(f_full,reg_method)
    norm1 = np.linalg.norm(reg.get_coeff(method).flatten(),ord=1)
    norm2 = np.linalg.norm(reg.get_coeff(method).flatten())

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title(f"{method.capitalize()} method with {rmse:.6f} RMSE")
    ax[1].set_title(f"Beta norms\n2-norm: {round(norm2,8)} | 1-norm: {round(norm1,8)}")

    ax[0].plot(x_full, f_full, label='True function', linestyle='dashed', color='gray')  # f_true
    ax[0].scatter(x_data, noisy_data, label='Noisy data', color='red')                      # f_noisy
    ax[0].plot(x_full,reg_method,label=method,color='blue')
    ax[0].legend()

    bars = ax[1].bar(x=reg.get_features(), height=abs(reg.get_coeff(method).flatten()))
    for bar in bars:
        height = bar.get_height()  # Get bar height
        ax[1].text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    upper_bound = max(7,max(abs(reg.get_coeff(method).flatten())))
    ax[1].set_ylim(0,upper_bound+1)
    # plt.savefig(f"Poly with {type} with {method.capitalize()} method with scale = {params['scale']} and dgr = {params['degree']}.png")
    plt.show()

def plot_with_noise():
    f_noisy = f_true + noise
    f_noisy_data = f_data + noise_data
    plot(x,x_data,f_true,f_noisy,f_noisy_data,params["method"])

def plot_with_noise_and_outlier(point=0,value=20):
    f_outlier = f_true + noise
    f_outlier[point] += value
    f_outlier_data = f_data + noise_data
    f_outlier_data[point] += value
    plot(x, x_data, f_true, f_outlier, f_outlier_data, params["method"])

def plot_with_outlier(point=0,value=20):
    f_norm = deepcopy(f_true)
    f_norm[point] += value
    f_norm_data = deepcopy(f_data)
    f_norm_data[point] += value
    plot(x,x_data,f_true,f_norm,f_norm_data,params["method"])

# plot_with_noise()
# plot_with_noise_and_outlier()
plot_with_outlier()

##minden egyben
def plot_all_in_one():
    f = deepcopy(f_true) + noise
    f[10] += 20

    reg = Regression(x=x, data=f, dgr=params['degree'])
    lls = getattr(reg, 'lls')()
    ridge = getattr(reg, 'ridge')()
    lasso = getattr(reg, 'lasso')()
    elastic = getattr(reg, 'elastic')()

    plt.figure(figsize=(10,6))
    plt.scatter(x,f,label='Noisy data', color='grey')
    plt.plot(x, f_true, label='True function', linestyle='dashed', color='black')
    plt.plot(x,lls,"r-", label="LLS")
    plt.plot(x, ridge, "b-", label="Ridge")
    plt.plot(x, lasso, "g-", label="Lasso")
    plt.plot(x, elastic, "m-", label="Elastic Net")

    plt.legend()
    plt.show()
# plot_all_in_one()