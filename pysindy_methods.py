import pysindy as ps
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from symbol_creation import *

class PysindyFunctions:
    def __init__(self,matrix:np.array, t:np.array, opt_method:str, order:int=2, degree:int=3, threshold:float=0.1):
        """
        Collection of pysindy function that going to be useful.
        :param np.array matrix: Matrix with the data values (calculated by SolveODE class)
        :param np.array t: Time points (get from SolveODE class)
        :param int order: optional. The order of the finite difference method.
        :param int degree: optional. The degree of the polynomials in the feature library.
        :param float threshold: optional. Minimum magnitude for coefficients in the weight vector. Coeffs below threshold are set to zero.
        """
        self.matrix = np.stack(matrix,axis=-1)
        self.t = t
        self.differentiation_method = ps.FiniteDifference(order=order)
        self.feature_library = ps.PolynomialLibrary(degree=degree)
        self.threshold = threshold
        self.opt_method = opt_method
        self.optimizer = None
        self.cr = CreateSymbols(len(matrix))

    def get_optimizer(self):
        """
        Choose an optimizer.
        """
        if self.opt_method == 'lls':
            optimizer = LinearRegression(fit_intercept=False)
        elif self.opt_method == 'ridge':
            optimizer = ps.SINDyOptimizer(Ridge(alpha=1.5,fit_intercept=False),unbias=False)
        elif self.opt_method == 'lasso':
            optimizer = Lasso(fit_intercept=False, alpha=0.2)
        elif self.opt_method == 'stlsq':
            optimizer = ps.STLSQ(threshold=self.threshold)
        else:
            print("ERROR - You didn't choose an optimizer method (lls, ridge, lasso, stlsq)!")
            exit()
        return optimizer

    def create_model(self) -> ps.SINDy:
        """
        Create a model for the equation.
        """
        self.optimizer = self.get_optimizer()
        model = ps.SINDy(differentiation_method=self.differentiation_method,
                         feature_library=self.feature_library,
                         optimizer=self.optimizer,
                         feature_names=self.cr.create_variables())
        return model

    def model_fit(self) -> ps.SINDy:
        """
        Fit the model
        """
        model = self.create_model()
        model.fit(self.matrix,t=self.t)
        return model

    def get_model_equations(self,model,number_of_decimals:int=3) -> list:
        """
        Get the right hand side of the SINDy model equations.
        :param model: The model we already fitted
        :param int number_of_decimals: The number of decimals in the equation.
        :return list
        """
        return model.equations(number_of_decimals)

    def print_model_equations(self,model,precision=1) -> None:
        """
        Print the model's equation we fitted on the data.
        :param model: The model we already fitted
        """
        return model.print(precision=precision)

    def get_feature_names(self,model) -> list:
        """
        Get a list of names of the features used in the model.
        :param model: The model we already fitted
        :return list
        """
        return model.get_feature_names()

    def get_coefficients(self,model) -> np.ndarray:
        """
        Get the coefficients learned by the model.
        :param model: The model we already fitted
        :return np.ndarray
        """
        return model.coefficients()

    def process_feature(self,feature) -> str:
        """
        In the features changing the symbols so that simpy can understand the notation.
        :return str
        """
        feature = feature.replace('^', '**')
        feature = feature.replace(' ', '*')
        return feature

    def simpify_feature(self,model) -> list:
        """
        Generate feature vector with symbols.
        :param model: The model we already fitted
        :return list
        """
        dc = self.cr.create_dict()
        fn = self.get_feature_names(model)
        fv = [sp.sympify(self.process_feature(feature),locals=dc) for feature in fn]
        return fv