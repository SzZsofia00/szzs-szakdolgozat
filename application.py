import pandas as pd
from tabulate import tabulate
from sympy. polys. orderings import monomial_key

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

    ## Model function

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

    ## Additional function

    def get_degree_of_list(self, lst:list) -> int:
        """
        Get what is the highest degree in the numerical solution.
        """
        dgr = 0
        for expr in lst:
            for term in expr.as_ordered_terms():
                for var, exp in term.as_powers_dict().items():
                    dgr = max(dgr, exp)
        return dgr

    def create_new_coeff_mtx(self,lst,fv):
        new = []
        for l in lst:
            dct = l.as_coefficients_dict()
            row = []
            for f in fv:
                coeff = dct.get(f, 0)
                row.append(coeff)
            new.append(row)
        coeff = np.array(new)
        return coeff

    def length_of_longer_fv(self, model):
        s = self.model_feature_vector(model)
        nm = self.num_method_feature_vector()

        return max(len(s),len(nm))

    def how_many_extra_feature(self,model):
        s = len(self.model_feature_vector(model))
        nm = len(self.num_method_feature_vector())
        return abs(nm - s)

    ## Num method function

    def num_method_with_symbols(self) -> list:
        """
        Solve the diff equation with the chosen numerical method symbolically
        """
        symb_init = self.pm.cr.create_symbols()
        nm = NumericalMethods(self.params["diff_eq"], 0, symb_init, self.params["step_size"])
        arr = getattr(nm, self.params["methodNM"])().flatten()
        lst = [sp.expand(expr) for expr in arr]
        return lst

    def num_method_feature_vector(self) -> list:
        """Create a feature vector depending on the numerical solution"""
        lst = self.num_method_with_symbols()

        _set = set()
        for i in lst:
            _set = _set | i.free_symbols

        sorted_lst = sorted(list(_set),key=lambda s: s.name,reverse=True)

        degree = self.get_degree_of_list(lst)
        fv = sorted(sp.itermonomials(sorted_lst,degree), key=monomial_key('grevlex', sorted_lst))
        return fv

    def num_method_coefficients(self) -> np.array:
        """
        Get the coefficient matrix for the numerical solution.
        """
        lst = self.num_method_with_symbols()
        fv = self.num_method_feature_vector()
        coeff = self.create_new_coeff_mtx(lst,fv)
        return coeff

    def num_method_solution(self) -> np.array:
        """
        The product of the numerical solution's feature vector and coefficient matrix.
        """
        fv = self.num_method_feature_vector()
        coeff = self.num_method_coefficients()
        return coeff * fv

    ## Dataframe function

    def create_index_for_df(self,method:str) -> list:
        """
        Create the index for the dataframe.
        """
        var = [" dx", " dy", " dz"]
        indx = []
        for i in range(len(self.params["init"])):
          indx.append(f"{method}{var[i]}")
        return indx

    def create_header_for_df(self,model) -> list:
        """
        Create the header for the dataframe.
        """
        nm_dgr = self.get_degree_of_list(self.num_method_with_symbols())
        s_dgr = self.get_degree_of_list(self.model_feature_vector(model))

        if s_dgr >= nm_dgr:
            fv = self.model_feature_vector(model)
        else:
            fv = self.num_method_feature_vector()
        header = [str(expr) for expr in fv]
        return header

    def create_dataframe(self,model,coeff,method:str) -> pd.DataFrame:
        """
        Create a pandas dataframe from the given data
        :param model: The fitted SINDy model
        :param coeff: Coefficients of the model
        :param str method: sindy or nm (for numerical method)
        :return: pd.Dataframe
        """
        header = self.create_header_for_df(model)
        indx = self.create_index_for_df(method)
        df = pd.DataFrame(coeff,index=indx,columns=header)
        return df

    def reshape_coeff_matrix(self,model):
        sindy_coeff = self.model_coefficients(model)
        nm_coeff = self.num_method_coefficients()
        length = self.length_of_longer_fv(model)
        extra = self.how_many_extra_feature(model)
        dim = sindy_coeff.shape[0]

        if sindy_coeff.shape[1] != length:
            sindy_coeff = np.hstack((sindy_coeff, np.zeros((dim, extra))))
        elif nm_coeff.shape[1] != length:
            nm_coeff = np.hstack((nm_coeff, np.zeros((dim, extra))))

        return sindy_coeff,nm_coeff

    def create_table_of_solutions(self, model) -> None:
        sindy_coeff, nm_coeff = self.reshape_coeff_matrix(model)

        df_sindy = self.create_dataframe(model, sindy_coeff, "sindy")
        df_nm = self.create_dataframe(model, nm_coeff, "nm")
        df = pd.concat([df_sindy, df_nm])
        print(tabulate(df, headers=self.create_header_for_df(model)))

    def squared_deviation(self,model) -> float:
        """
        Calculate the squared deviation between the two methods
        """
        sindy_coeff, nm_coeff = self.reshape_coeff_matrix(model)

        sq_dev = (sindy_coeff.reshape(1, -1)[0] - nm_coeff.reshape(1, -1)[0]) ** 2
        length = self.length_of_longer_fv(model)
        summa = 0
        for i in sq_dev:
            summa += i
        return summa / length
