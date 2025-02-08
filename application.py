import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from tabulate import tabulate

from pysindy_methods import *
from solve_methods import *

class Application:
    def __init__(self,params:dict):
        """
        Main class for running tests, creating features, dataframes of the solutions.
        :param dict params: A dictionary of parameters:
        - diff_eq: The differential equation we observe
        - init: list of the initial values
        - time: list of time period
        - step_size: float, steps between time points
        - methodSy: string, the numerical method for the SINDy model
        - methodNM: string, the numerical method for the numerical solution
        - be_noise: bool, False if no noise, True otherwise
        - degree: degree of the feature vector in the SINDy model
        - threshold: threshold for the optimizer in the SINDy model
        """
        self.params = params
        self.so = SolveODE(self.params["diff_eq"], self.params["time"], self.params["init"], self.params["step_size"])
        self.mtx = self.so.get_matrix_with_noise(self.params["methodSy"], be_noise=self.params["be_noise"])
        self.t = self.so.create_time_points()
        self.pm = PysindyFunctions(self.mtx, self.t, threshold=self.params["threshold"])

    def fit_sindy_model(self):
        """
        The fitted model
        """
        model = self.pm.model_fit()
        return model

    def model_feature_vector(self,model) -> list:
        """
        Feature vector of the SINDy model.
        """
        fv = self.pm.simpify_feature(model)
        return fv

    def model_coefficients(self,model) -> np.array:
        """
        The coefficients in the SINDy model.
        """
        coef = self.pm.get_coefficients(model)
        return coef

    def model_solution(self,model) -> np.array:
        """
        The product of SINDy feature vector and coefficient matrix.
        """
        fv = self.model_feature_vector(model)
        coef = self.model_coefficients(model)
        sol = coef * fv
        return sol

    def num_method_with_symbols(self) -> list:
        """
        Solve the diff equation with the chosen numerical method symbolically
        """
        symb_init = self.pm.cr.create_symbols()
        nm = NumericalMethods(self.params["diff_eq"], 0, symb_init, self.params["step_size"])
        arr = getattr(nm, self.params["methodNM"])().flatten()
        lst = [sp.expand(expr) for expr in arr]
        return lst

    def num_method_coefficients(self,model) -> np.array:
        lst = self.num_method_with_symbols()
        fv = self.model_feature_vector(model)

        new = []
        for l in lst:
            dct = l.as_coefficients_dict()
            row = []
            for f in fv:
                coeff = dct.get(f, 0)
                row.append(coeff)
            new.append(row)
        coeff = np.array(new)
        return lst,fv,coeff

    def num_method_solution(self,model) -> np.array:
        fv = self.model_feature_vector(model)
        coeff = self.num_method_coefficients(model)
        return coeff * fv

    def create_index_for_df(self,method) -> list:
        var = [" dx", " dy", " dz"]
        indx = []
        for i in range(len(self.params["init"])):
          indx.append(f"{method}{var[i]}")
        return indx

    def create_header_for_df(self,model) -> list:
        fv = self.model_feature_vector(model)
        header = [str(expr) for expr in fv]
        return header

    def create_dataframe(self,model,coeff,method) -> pd.DataFrame:
        header = self.create_header_for_df(model)
        indx = self.create_index_for_df(method)
        df = pd.DataFrame(coeff,index=indx,columns=header)
        return df

    def create_table_of_solutions(self,model) -> None:
        df_sindy = self.create_dataframe(model,self.model_coefficients(model),"sindy")
        df_nm = self.create_dataframe(model,self.num_method_coefficients(model),"nm")
        df = pd.concat([df_sindy, df_nm])
        print(tabulate(df, headers=self.create_header_for_df(model)))

    def squared_deviation(self,model) -> float:
        coef_model = self.model_coefficients(model)
        coefNM = self.num_method_coefficients(model)

        sq_dev = (coef_model.reshape(1, -1)[0] - coefNM.reshape(1, -1)[0]) ** 2
        length = len(coefNM.reshape(1, -1)[0])
        summa = 0
        for i in sq_dev:
            summa += i
        return summa / length
