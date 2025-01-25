import pysindy as ps
import numpy as np
from symbol_creation import *

class PysindyFunctions:
    def __init__(self,matrix:np.array, t:np.array, order:int=2, degree:int=3, threshold:float=0.2):
        """
        Collection of pysindy function that going to be useful for me.
        :param np.array matrix: Matrix with the data values (calculated be SolveODE class)
        :param np.array t: Time points (get from SolveODE class)
        :param int order: optional. The order of the finite difference method.
        :param int degree: optional. The degree of the polynomials in the feature library.
        :param float threshold: optional. Minimum magnitude for coefficients in the weight vector. Coeffs below threshold are set to zero.
        """
        self.matrix = np.stack(matrix,axis=-1)
        self.t = t
        self.differentiation_method = ps.FiniteDifference(order=order)
        self.feature_library = ps.PolynomialLibrary(degree=degree)
        self.optimizer = ps.STLSQ(threshold=threshold)
        self.cr = CreateSymbols(len(matrix))

    def create_model(self):
        """
        Create a model for the equation.
        """
        model = ps.SINDy(differentiation_method=self.differentiation_method,
                         feature_library=self.feature_library,
                         optimizer=self.optimizer,
                         feature_names=self.cr.create_variables())
        return model

    def model_fit(self):
        """
        Fit the model
        """
        #mtx = np.stack(self.matrix,axis=-1)
        model = self.create_model()
        model.fit(self.matrix,t=self.t)
        return model

    def get_model_equations(self,number_of_decimals:int=3):
        """
        Get the right hand side of the SINDy model equations for each feature.
        :param int number_of_decimals: The number of decimals in the equation.
        """
        model = self.model_fit()
        return model.equations(number_of_decimals)

    def print_model_equations(self):
        """
        Print the model's equation we fitted on the data.
        """
        model = self.model_fit()
        return model.print()

    def get_feature_names(self):
        """
        Get a list of names of features used in the model.
        """
        model = self.model_fit()
        return model.get_feature_names()

    def get_coefficients(self):
        """
        Get the coefficients learned by the model.
        """
        model = self.model_fit()
        coef = model.coefficients()
        # rounded_coef = np.round(coef,4)
        return coef

    def process_feature(self,feature):
        """
        In the features changing the symbols so that simpy can understand the notation.
        """
        feature = feature.replace('^', '**')
        feature = feature.replace(' ', '*')
        return feature

    def sympify_feature(self):
        """
        Generate feature vector with symbols.
        """
        dc = self.cr.create_dict()
        fn = self.get_feature_names()
        fv = [sp.sympify(self.process_feature(feature),locals=dc) for feature in fn]
        return fv

