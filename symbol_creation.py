import sympy as sp
import numpy as np

class CreateSymbols:
    def __init__(self,dimension:int):
        """
        Create variables and sympy symbols.
        :param dimension: How many variables we want to create. Can only handle maximum 3.
        """
        self.dimension = dimension

    def create_variables(self) -> list:
        """
        Create an array of different variables.
        :return: list: String of variables in a list
        """
        variables = []
        for i in range(self.dimension):
          variables.append(chr(120+i))
        return variables

    def create_symbols(self) -> list:
        """
        Create an array of symbols from variables.
        :return: list
        """
        variables = self.create_variables()
        symbols = []
        for i in variables:
          new_symb = sp.symbols(f'{i}')
          symbols.append(new_symb)
        return symbols

    def create_xyz_symb(self) -> list:
        """
        Create an array of symbols depending on the dimension.
        """
        symb = []
        if self.dimension == 1:
            x = self.create_symbols()
            symb = x
        elif self.dimension == 2:
            x,y = self.create_symbols()
            symb = [x,y]
        elif self.dimension == 3:
            x,y,z = self.create_symbols()
            symb = [x,y,z]
        return symb

    def create_dict(self) -> dict:
        """
        Create a dictionary with the variables and there symbolic form.
        :return: dict
        """
        var = self.create_variables()
        symb = self.create_xyz_symb()
        return dict(zip(var,symb))

    def create_var_for_dde(self,prev):
        var = [[f'{chr(120 + i)}_{j}' for j in range(prev)] for i in range(self.dimension)]
        turned = np.fliplr(var)
        return np.array(var).flatten()

    def create_symbold_for_dde(self,prev):
        symb = [[sp.symbols(f'{chr(120 + i)}_{j}') for j in range(prev)] for i in range(self.dimension)]
        turned = np.fliplr(symb)
        return turned.flatten()