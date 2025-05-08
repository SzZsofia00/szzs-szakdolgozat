import numpy as np
import pysindy as ps
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from solve_differential_equation import *
from pysindy_methods import *

import timeit

class Optimizers:
    def __init__(self,params,mtx,t):
        self.params = params
        # self.so = SolveODE(self.params["diff_eq"], self.params["time"], self.params["init"], self.params["step_size"])
        # self.mtx = self.so.get_matrix_with_noise(self.params["methodSy"], be_noise=self.params["be_noise"]).T
        # self.t = self.so.create_time_points()
        self.mtx = mtx
        self.t = t

    def differentiation(self):
        diff_method = ps.FiniteDifference()
        X_dot = diff_method._differentiate(self.mtx,self.t)
        return X_dot

    def feature_names(self):
        feature_library = ps.PolynomialLibrary(degree=self.params["degree"])
        feature_library.fit(self.mtx)
        return feature_library.get_feature_names()

    def process_feature(self,feature) -> str:
        """
        In the features changing the symbols so that simpy can understand the notation.
        :return str
        """
        feature = feature.replace('^', '**')
        feature = feature.replace(' ', '*')
        return feature

    def simpify_feature(self) -> list:
        """
        Generate feature vector with symbols.
        :param model: The model we already fitted
        :return list
        """
        self.cr = CreateSymbols(len(self.mtx.T))
        dc = self.cr.create_dict()
        fn = self.feature_names()
        fv = [sp.sympify(self.process_feature(feature),locals=dc) for feature in fn]
        return fv

    def feature_library(self):
        feature_library = ps.PolynomialLibrary(degree=self.params["degree"])
        feature_library.fit(self.mtx)
        theta = feature_library.transform(self.mtx)
        return theta

    def gsls(self,ridge_alpha=0.5):
        theta = self.feature_library()
        X_dot = self.differentiation()

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
                    Xi_tmp = np.array(lls.coef_)

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
            Xi_tmp = np.array(lls.coef_)
            i = 0
            for j in range(len(Xi)):
                if j not in Q:
                    Xi[j] = Xi_tmp[i]
                    i = i + 1
                else:
                    Xi[j] = 0.

            break
        return Xi.T

    def lls(self):
        X_dot = self.differentiation()
        theta = self.feature_library()

        model = LinearRegression(fit_intercept=False)
        model.fit(theta, X_dot)
        return model.coef_

    def ridge(self,alpha_ridge=0.5):
        X_dot = self.differentiation()
        theta = self.feature_library()

        model = Ridge(alpha=alpha_ridge, fit_intercept=False)
        model.fit(theta, X_dot)
        return model.coef_

    def lasso(self,alpha_lasso=0.2):
        X_dot = self.differentiation()
        theta = self.feature_library()

        model = Lasso(alpha=alpha_lasso, fit_intercept=False)
        model.fit(theta, X_dot)
        return model.coef_

    def stlsq(self):
        pm = PysindyFunctions(self.mtx.T, self.t, degree=self.params["degree"], threshold=self.params["threshold"])
        model = pm.model_fit()
        return pm.get_coefficients(model)
