from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import capitalize
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

np.random.seed(42)

class Functions:
    def __init__(self,x):
        """
        Collection of functions.
        """
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
    def __init__(self,parameters,data,time_points,time_predict):
        self.params = parameters
        self.data = data #data points
        self.features = PolynomialFeatures(params['degree'])
        self.time = self.features.fit_transform(time_points) #t for the data points
        self.time_predict = self.features.fit_transform(time_predict) #t of the entire intervall, we predict here for smoother function

    def get_model(self, method: str):
        """
        Create a fitted model for the chosen optimizing method.
        :param str method: lls, ridge, lasso or elastic
        """
        if method == 'lls':
            model = LinearRegression(fit_intercept=False)
        elif method == 'ridge':
            model = Ridge(alpha=self.params['alpha_ridge'], fit_intercept=False)
        elif method == 'lasso':
            model = Lasso(alpha=self.params['alpha_lasso'], fit_intercept=False)
        elif method == 'elastic':
            model = ElasticNet(alpha=self.params['alpha_elastic'], fit_intercept=False)
        model.fit(self.time, self.data)
        return model

    def lls(self) -> np.ndarray:
        """
        Gives back the values for the model predicted with linear least square method.
        """
        lls_model = self.get_model('lls')
        return lls_model.predict(self.time_predict)

    def ridge(self) -> np.ndarray:
        """
        Gives back the values for the model predicted with Ridge regression.
        """
        ridge_model = self.get_model('ridge')
        return ridge_model.predict(self.time_predict)

    def lasso(self) -> np.ndarray:
        """
        Gives back the values for the model predicted with LASSO regression.
        """
        lasso_model = self.get_model('lasso')
        return lasso_model.predict(self.time_predict)

    def elastic(self) -> np.ndarray:
        """
        Gives back the values for the model predicted with Elastic Net method.
        """
        elastic_model = self.get_model('elastic')
        return elastic_model.predict(self.time_predict)

    def gsls(self, ridge_alpha=0.5):
        theta = self.time
        X_dot = self.data.reshape(-1,1)

        Q = []

        lls = LinearRegression(fit_intercept=False)

        ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
        ridge.fit(theta, X_dot)
        Xi = np.array(ridge.coef_)

        C = np.linalg.norm(X_dot - theta @ Xi)

        while True:
            mask = [i for i in range(theta.shape[1]) if i not in Q]

            theta_msk = theta[:, mask]
            Xi_msk = Xi[mask]

            C = np.linalg.norm(X_dot - theta_msk @ Xi_msk)

            candidate_error = []

            for i, idx in enumerate(mask):  # kinda a map with i (sorszam) and corresponding index in the original Xi

                if theta_msk.shape[1] > 1:
                    theta_tmp = theta_msk.copy()
                    theta_tmp = np.delete(theta_tmp, i, axis=1)

                    lls.fit(theta_tmp, X_dot)
                    Xi_tmp = np.array(lls.coef_).flatten()

                    err = np.linalg.norm(X_dot - theta_tmp @ Xi_tmp)
                    candidate_error.append((err, idx))
                else:
                    candidate_error.append((C * 2, 0))

            min_error, min_index = min(candidate_error)

            if min_error < C * 1.1:
                Xi[min_index] = 0
                Q.append(min_index)
                continue

            lls.fit(theta_msk, X_dot)
            Xi_tmp = np.array(lls.coef_).flatten()
            i = 0
            for j in range(len(Xi)):
                if j not in Q:
                    Xi[j] = Xi_tmp[i]
                    i = i + 1
                else:
                    Xi[j] = 0.

            break
        return [theta @ Xi, Xi]

    def get_coeff(self,method:str) -> np.ndarray:
        """
        Get the coefficients of the features for the given method.
        :param str method: lls, ridge, lasso or elastic
        """
        if method == 'gsls':
            Xi = self.gsls()[1]
            return Xi

        model = self.get_model(method)
        print(model.coef_)
        return model.coef_.flatten()

    def get_features(self) -> np.ndarray:
        """
        Gives back the feature's name for the regression model.
        """
        return self.features.get_feature_names_out(input_features=["x"])

#####################################

def generate_inputs(time,num_of_points, scale):
    t = np.linspace(time[0], time[1], num_of_points).reshape(-1,1)
    noise = np.random.normal(loc=0.0, scale=scale, size=t.shape)
    f = Functions(t).linear()
    return t,noise,f

def calculate_regression(parameters,time,time_full,noisy,method):
    reg = Regression(parameters=parameters,data=noisy,time_points=time,time_predict=time_full)
    reg_method = getattr(reg,method)()

    if method == 'gsls':
        reg_method = reg_method[0]

    return reg, reg_method

def plot_regression(ax, method, rmse,t,t_full,f_noisy,f_true,reg_method,point=[]):
    ax.set_title(f"{method.upper()} method with {rmse:.6f} RMSE")
    ax.plot(t_full, f_true, label='True function', linestyle='dashed', color='gray')  # f_true
    ax.scatter(t, f_noisy, label='Noisy data', color='red')                     # f_noisy
    if method == 'gsls':
        ax.plot(t, reg_method, label=method.capitalize(), color='blue')
    else:
        ax.plot(t_full, reg_method, label=method.capitalize(), color='blue')
    outlier_indices = point
    if len(outlier_indices) > 0:
        for i in outlier_indices:
            ax.scatter(t[i], f_noisy[i], color='black', s=100, marker='*', label='Outlier')
    ax.legend()

def plot_coefficients(ax,norm1,norm2,reg,method):
    ax.set_title(r"Coefficients of $\beta$")
    beta_label = fr"$||\beta||_2={round(norm2, 3)} \ ||\beta||_1={round(norm1, 3)}$"
    coeff = reg.get_coeff(method).flatten()
    bars = ax.bar(x=reg.get_features(), height=abs(coeff), label=beta_label)
    for bar in bars:
        height = bar.get_height()  # Get bar height
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    upper_bound = max(7, max(abs(coeff)))
    ax.set_ylim(0, upper_bound + 1)
    ax.legend([beta_label], handlelength=0, handletextpad=0, fontsize=20)

def plot(params,t,t_full,f_noisy,f,f_true,method,point=[]):
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 15

    reg,reg_method = calculate_regression(params,t,t_full,f_noisy,method)

    if method == 'gsls':
        rmse = root_mean_squared_error(f_noisy,reg_method)
    else:
        rmse = root_mean_squared_error(f_true, reg_method)
    norm1 = np.linalg.norm(reg.get_coeff(method).flatten(), ord=1)
    norm2 = np.linalg.norm(reg.get_coeff(method).flatten())

    fig,ax = plt.subplots(1,2,figsize=(15,5))
    plot_regression(ax[0],method,rmse,t,t_full,f_noisy,f_true,reg_method, point)
    plot_coefficients(ax[1],norm1,norm2,reg,method)
    plt.savefig(f"{method}.pdf")
    plt.show()

######################################

params = {
    'time': [-2,2],
    'points_data': 10,
    'points_for_plot': 100,
    'scale': 1,
    'degree': 5,
    'method': 'lasso',
    'alpha_ridge': 0.5,
    'alpha_lasso': 0.2,
    'alpha_elastic': 0.2
}

t,noisy,f = generate_inputs(params['time'],params['points_data'],params['scale'])
t_full,noisy_full,f_full = generate_inputs(params['time'],params['points_for_plot'],params['scale'])

def plot_with_noise():
    f_noisy = f + noisy
    plot(params,t,t_full,f_noisy,f,f_full,params['method'])

def plot_with_noise_and_outlier():
    f_noisy = f + noisy
    point = 1
    f_noisy[point] += -3
    plot(params,t,t_full,f_noisy,f,f_full,params['method'],[point])

def plot_with_outlier():
    f_noisy = deepcopy(f)
    point = 0
    f_noisy[point] += -3
    plot(params, t, t_full, f_noisy,f, f_full, params['method'],[point])

# plot_with_noise()
plot_with_noise_and_outlier()
# plot_with_outlier()









