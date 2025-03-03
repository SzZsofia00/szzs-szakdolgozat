from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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

class Regression:
    def __init__(self,x,data,dgr,alpha_ridge,alpha_lasso,alpha_elast):
        self.dgr = dgr
        self.data = data
        self.features = PolynomialFeatures(self.dgr)
        self.x = self.features.fit_transform(x)
        self.alpha_ridge = alpha_ridge
        self.alpha_lasso = alpha_lasso
        self.alpha_elast = alpha_elast

    def get_features(self):
        return self.features.get_feature_names_out(input_features=["x"])

    def get_model(self,method):
        if method=='lls':
            print('lls')
            model = LinearRegression()
        elif method=='ridge':
            print('ridge')
            model = Ridge(alpha=self.alpha_ridge)
        elif method=='lasso':
            print('lasso')
            model = Lasso(alpha=self.alpha_lasso)
        elif method=='elastic':
            print('elastic')
            model = ElasticNet(alpha=self.alpha_elast)
        model.fit(self.x, self.data) #(data,target)
        print(model)
        return model

    def get_coeff(self,method):
        model = self.get_model(method)
        return model.coef_

    def lls(self):
        lls_model = self.get_model('lls')
        return lls_model.predict(self.x)

    def ridge(self):
        ridge_model = self.get_model('ridge')
        return ridge_model.predict(self.x)

    def lasso(self):
        lasso_model = self.get_model('lasso')
        return lasso_model.predict(self.x)

    def elastic(self):
        elastic_model = self.get_model('elastic')
        return elastic_model.predict(self.x)


params = {
    "time": [-2,2],
    "number_of_points": 50,
    "scale": 2,     #scale = standard deviation
    "degree": 5,      #for features
    "method": "lasso",   # lls / ridge / lasso / elastic
    "alpha_ridge": 0.5,
    "alpha_lasso": 0.5,
    "alpha_elastic": 0.5
}

x = np.linspace(params["time"][0],params["time"][1],params["number_of_points"]).reshape(-1,1)
np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=params["scale"], size=x.shape)
f_true = Functions(x).cubic()

def plot(x, real, noisy, dgr, method):
    reg = Regression(x=x,data=noisy,dgr=dgr,
                     alpha_ridge=params["alpha_ridge"],
                     alpha_lasso=params["alpha_lasso"],
                     alpha_elast=params["alpha_elastic"])
    reg_method = getattr(reg, method)()
    rmse = root_mean_squared_error(real,reg_method)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # fig, ax = plt.subplots(1, 2)
    ax[0].set_title(f"{method.capitalize()} method with {rmse:.6f} RMSE")
    ax[1].set_title("Beta norms")

    ax[0].plot(x, real, label='True function', linestyle='dashed', color='gray')  # f_true
    ax[0].scatter(x, noisy, label='Noisy data', color='red')                      # f_noisy
    ax[0].plot(x,reg_method,label=method,color='blue')
    ax[0].legend()

    bars = ax[1].bar(x=reg.get_features(), height=abs(reg.get_coeff(method).flatten()))
    for bar in bars:
        height = bar.get_height()  # Get bar height
        ax[1].text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax[1].set_ylim(0,8)
    # plt.savefig(f"{method.capitalize()} method with {rmse:.6f} RMSE.png")
    plt.show()

## plotting only noisy data
f_noisy = f_true + noise
plot(x, f_true, f_noisy, params["degree"], params["method"])

## plotting noisy data with outliers
f_outlier = f_true + noise
f_outlier[10] = f_outlier[10] + 20
f_outlier[40] = f_outlier[40] - 25
plot(x, f_true, f_outlier, params["degree"], params["method"])

## plotting data (no noisy) but outlier
f_norm = deepcopy(f_true)
f_norm[10] = f_norm[10] + 20
plot(x, f_true, f_norm, params["degree"], params["method"])

##minden egyben
def plot_all_in_one():
    f = deepcopy(f_true) + noise
    f[10] += 20
    dgr = 5
    # x_new = np.linspace(-2.5,2.5,50).reshape(-1,1)

    reg = Regression(x=x, data=f, dgr=dgr,
                     alpha_ridge=params["alpha_ridge"],
                     alpha_lasso=params["alpha_lasso"],
                     alpha_elast=params["alpha_elastic"])
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
plot_all_in_one()