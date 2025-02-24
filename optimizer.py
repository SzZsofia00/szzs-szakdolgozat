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

class Regression:
    def __init__(self,x,data,dgr):
        self.dgr = dgr
        self.data = data
        self.features = PolynomialFeatures(self.dgr)
        self.x = self.features.fit_transform(x)

    def get_features(self):
        return self.features.get_feature_names_out(input_features=["x"])

    def get_model(self,method):
        if method=='lls':
            model = LinearRegression()
        elif method=='ridge':
            model = Ridge()
        elif method=='lasso':
            model = Lasso()
        elif method=='elastic':
            model = ElasticNet()
        model.fit(self.x, self.data)
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
    "scale": 0.5,     #scale = standard deviation
    "degree": 7,      #for features
    "method": "lasso"   # lls / ridge / lasso / elastic
}

x = np.linspace(params["time"][0],params["time"][1],params["number_of_points"]).reshape(-1,1)
np.random.seed(523)
noise = np.random.normal(loc=0.0, scale=params["scale"], size=x.shape)
f_true = Functions(x).complex2()

def plot(x, real, noisy, dgr, method):
    reg = Regression(x=x,data=noisy,dgr=dgr)
    reg_method = getattr(reg, method)()
    rmse = root_mean_squared_error(real,reg_method)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title(f"{method.capitalize()} method with {rmse:.6f} RMSE")
    ax[1].set_title("Beta norms")

    ax[0].plot(x, real, label='True function', linestyle='dashed', color='gray')  # f_true
    ax[0].scatter(x, noisy, label='Noisy data', color='red')                      # f_noisy
    ax[0].plot(x,reg_method,label=method,color='blue')
    ax[0].legend()

    ax[1].bar(x=reg.get_features(), height=abs(reg.get_coeff(method).flatten()))
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
