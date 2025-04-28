import itertools
from solve_differential_equation import *
from pysindy_methods import *
from symbol_creation import *

from differential_equations import *

class NumericalModel:
    def __init__(self,params):
        self.params = params
        self.so = SolveODE(self.params["diff_eq"], self.params["time"], self.params["init"], self.params["step_size"])
        self.mtx = self.so.get_matrix_with_noise(self.params["methodSy"], be_noise=self.params["be_noise"]).T
        self.t = self.so.create_time_points()
        self.pm = PysindyFunctions(self.mtx.T, self.t, degree=self.params["degree"], threshold=self.params["threshold"])

    def differential_eq_symbolic(self) -> list:
        """
        Solve the diff equation with the chosen numerical method symbolically
        """
        symb_init = self.pm.cr.create_symbols()
        nm = NumericalMethods(self.params["diff_eq"], 0, symb_init, self.params["step_size"])
        arr = getattr(nm, self.params["methodNM"])().flatten()
        lst = [sp.expand(expr) for expr in arr]
        return lst

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

    def symbolic_features(self) -> list:
        """Create a feature vector depending on the numerical solution"""
        lst = self.differential_eq_symbolic()
        symb = CreateSymbols(len(self.params["init"])).create_symbols()

        dgr = self.get_degree_of_list(lst)

        fv = []
        for d in range(dgr + 1):
            for i in itertools.combinations_with_replacement(symb,d):
                fv.append(sp.Mul(*i))

        return fv

    def create_new_coeff_mtx(self,lst:list,fv:list) -> np.ndarray:
        """
        Create a coefficient matrix from a list and feature vector.
        """
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

    def symbolic_solution_coefficients(self) -> np.ndarray:
        """
        Get the coefficient matrix for the numerical solution.
        """
        lst = self.differential_eq_symbolic()
        fv = self.symbolic_features()
        coeff = self.create_new_coeff_mtx(lst,fv)
        return coeff

    def symbolic_solution(self) -> np.ndarray:
        """
        The product of the numerical solution's feature vector and coefficient matrix.
        """
        fv = self.symbolic_features()
        coeff = self.symbolic_solution_coefficients()
        return coeff * fv
