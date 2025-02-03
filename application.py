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

    def model_feature_vector(self):
        pm = self.sindy_model()
        fv = pm.sympify_feature()
        return fv

    def model_coefficients(self):
        pm = self.sindy_model()
        coef = pm.get_coefficients()
        return coef

    def model_solution(self):
        fv = self.model_feature_vector()
        coef = self.model_coefficients()
        sol = coef * fv
        return sol

    def num_method_with_symbols(self):
        pm = self.sindy_model()
        symb_init = pm.cr.create_symbols()
        nm = NumericalMethods(self.params["diff_eq"], 0, symb_init, self.params["step_size"])
        lst = [sp.expand(expr) for expr in getattr(nm, self.params["methodNM"])()]
        return lst

    def num_method_coefficients(self):
        lst = self.num_method_with_symbols()
        fv = self.model_feature_vector()

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

    def num_method_solution(self):
        fv = self.model_feature_vector()
        coeff = self.num_method_coefficients()
        return coeff * fv

    def create_table_of_solutions(self):
        fv = self.model_feature_vector()
        header = [str(expr) for expr in fv]
        df_sindy = pd.DataFrame(self.model_coefficients(),
                                index=["sindy-vel dx", "sindy-vel dy", "sindy-vel dz"],
                                columns=header)
        df_nm = pd.DataFrame(self.num_method_coefficients(),
                             index=["nm-vel dx", "nm-vel dy", "nm-vel dz"],
                             columns=header)
        df = pd.concat([df_sindy, df_nm])
        print(tabulate(df, headers=header))

    def squared_deviation(self):
        coef_model = self.model_coefficients()
        coefNM = self.num_method_coefficients()

        sq_dev = (coef_model.reshape(1, -1)[0] - coefNM.reshape(1, -1)[0]) ** 2
        length = len(coefNM.reshape(1, -1)[0])
        summa = 0
        for i in sq_dev:
            summa += i
        print("Squared deviation: ", summa / length)
