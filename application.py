from optimizers import *
from model_numerically import *
import pandas as pd
from tabulate import tabulate

class DataframeForCoefficients:
    def __init__(self,params):
        self.params = params
        self.opt = Optimizers(self.params)
        self.num = NumericalModel(self.params)

    def length_of_longer_fv(self) -> int:
        """
        Gives back the length of the longer feature vector between the model's and the one solved simbolically.
        """
        # s = self.num.pm.simpify_feature()
        s = self.opt.stlsq()
        nm = self.num.symbolic_features()
        return max(s.shape[1],len(nm))

    def how_many_extra_feature(self) -> int:
        """
        Gives back the absolute difference of the length of the SINDY's feature vector and symbolically solved one.
        """
        # s = len(self.num.pm.simpify_feature())
        s = self.opt.stlsq().shape[1]
        nm = len(self.num.symbolic_features())
        return abs(nm - s)

    def reshape_coeff_matrix(self,optimizer:str = None):
        """
        Reshape coefficients matrices so they have the same shape
        """
        if optimizer is None:
            coeff = self.num.symbolic_solution_coefficients()
        else:
            coeff = getattr(self.opt,optimizer)()

        coeff = np.atleast_2d(coeff) #ez azért h ha nem 2d-s shape
        if coeff.shape[0] == 1 and len(self.params["init"]) != 1:
            coeff = coeff.T

        length = self.length_of_longer_fv()
        extra = self.how_many_extra_feature()
        dim = len(self.params["init"])

        if coeff.shape[1] != length:
            coeff = np.hstack((coeff, np.zeros((dim, extra))))

        return coeff

    def squared_deviation(self,optimizer:str) -> float:
        """
        Calculate the squared deviation between the two methods
        """
        opt_coeff = self.reshape_coeff_matrix(optimizer)
        nm_coeff = self.reshape_coeff_matrix()

        sq_dev = (opt_coeff.reshape(1, -1)[0] - nm_coeff.reshape(1, -1)[0]) ** 2
        length = self.length_of_longer_fv()
        summa = 0
        for i in sq_dev:
            summa += i
        return summa / length

    # dataframe
    def get_multiplication_length(self,term) -> int:
        """
        Giving back how many factors does the term consists of.
        :param term: A symbolic expression
        """
        if isinstance(term, sp.Mul):
            return sum(self.get_multiplication_length(arg) for arg in term.args)
        elif isinstance(term, sp.Pow):
            return term.exp if term.base.is_Symbol else 1
        elif term.is_Symbol:
            return 1
        return 0

    def get_degree_of_list(self, lst:list) -> int:
        """
        Get what is the highest degree in a list.
        """
        dgr = max(max(self.get_multiplication_length(term) for term in expr.as_ordered_terms()) for expr in lst)
        return dgr

    def create_header_for_df(self) -> list:
        """
        Create the header for the dataframe.
        :param model: The fitted SINDy model
        """
        nm_dgr = self.get_degree_of_list(self.num.symbolic_features())
        s_dgr = self.get_degree_of_list(self.opt.simpify_feature())

        if s_dgr >= nm_dgr:
            fv = self.opt.simpify_feature()
        else:
            fv = self.num.symbolic_features()
        header = [str(expr) for expr in fv]
        return header

    def create_index_for_df(self,method:str) -> list:
        """
        Create the index for the dataframe.
        :param str method: sindy, nm (for numerical method), lls, ridge, lasso, gsls
        """
        var = [" dx", " dy", " dz"]
        indx = []
        for i in range(len(self.params["init"])):
          indx.append(f"{method}{var[i]}")
        return indx

    def create_dataframe(self,coeff:np.ndarray,method:str) -> pd.DataFrame:
        """
        Create a pandas dataframe from the given data
        :param model: The fitted SINDy model
        :param coeff: Coefficients of the model
        :param str method: sindy or nm (for numerical method)
        :return: pd.Dataframe
        """
        header = self.create_header_for_df()
        indx = self.create_index_for_df(method)
        df = pd.DataFrame(coeff,index=indx,columns=header)
        return df

    def create_table_of_solutions(self,optimizer:str) -> None:
        opt_coeff = self.reshape_coeff_matrix(optimizer)
        nm_coeff = self.reshape_coeff_matrix()

        df_sindy = self.create_dataframe(opt_coeff, optimizer)
        df_nm = self.create_dataframe(nm_coeff, "nm")
        df = pd.concat([df_sindy, df_nm])
        print(tabulate(df, headers=self.create_header_for_df()))

