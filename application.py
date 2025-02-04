import pandas as pd
from tabulate import tabulate

from pysindy_methods import *
from solve_methods import *

class Application:
    def __init__(self,params):
        self.params = params
        self.so = SolveODE(self.params["diff_eq"], self.params["time"], self.params["init"], self.params["step_size"])

    def sindy_model(self):
        mtx = self.so.get_matrix_with_noise(self.params["methodSy"], be_noise=self.params["be_noise"])
        t = self.so.create_time_points()
        pm = PysindyFunctions(mtx, t, threshold=self.params["threshold"])
        return pm

    def model_feature_vector(self,model):
        fv = model.simpify_feature()
        return fv

    def model_coefficients(self,model):
        coef = model.get_coefficients()
        return coef

    def model_solution(self,model):
        fv = self.model_feature_vector(model)
        coef = self.model_coefficients(model)
        sol = coef * fv
        return sol

    def num_method_with_symbols(self,model):
        if len(self.params["init"]) == 1:
            symb_init = [sp.Symbol('x')]
            nm = NumericalMethods(self.params["diff_eq"], 0, symb_init, self.params["step_size"])
            lst = [sp.expand(expr) for expr in getattr(nm, self.params["methodNM"])()[0]]
        else:
            symb_init = model.cr.create_symbols()
            nm = NumericalMethods(self.params["diff_eq"], 0, symb_init, self.params["step_size"])
            lst = [sp.expand(expr) for expr in getattr(nm, self.params["methodNM"])()]
        return lst

    def num_method_coefficients(self,model):
        lst = self.num_method_with_symbols(model)
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
        return coeff

    def num_method_solution(self,model):
        fv = self.model_feature_vector(model)
        coeff = self.num_method_coefficients(model)
        return coeff * fv

    def create_index_for_df(self,method):
        var = [" dx", " dy", " dz"]
        indx = []
        for i in range(len(self.params["init"])):
          indx.append(f"{method}{var[i]}")
        return indx

    def create_header_for_df(self,model):
        fv = self.model_feature_vector(model)
        header = [str(expr) for expr in fv]
        return header

    def create_dataframe(self,model,coeff,method):
        header = self.create_header_for_df(model)
        indx = self.create_index_for_df(method)
        df = pd.DataFrame(coeff,index=indx,columns=header)
        return df

    def create_table_of_solutions(self,model):
        df_sindy = self.create_dataframe(model,self.model_coefficients(model),"sindy")
        df_nm = self.create_dataframe(model,self.num_method_coefficients(model),"nm")
        df = pd.concat([df_sindy, df_nm])
        print(tabulate(df, headers=self.create_header_for_df(model)))

    def squared_deviation(self,model):
        coef_model = self.model_coefficients(model)
        coefNM = self.num_method_coefficients(model)

        sq_dev = (coef_model.reshape(1, -1)[0] - coefNM.reshape(1, -1)[0]) ** 2
        length = len(coefNM.reshape(1, -1)[0])
        summa = 0
        for i in sq_dev:
            summa += i
        print("Squared deviation: ", summa / length)
