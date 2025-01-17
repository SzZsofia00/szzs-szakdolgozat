import sympy as sp
import numpy as np

def create_variables(dimension:int) -> np.array:
  """
  Create an array of different variables
  :param int dimension: the number of different variables.
  :return: np.array
  """
  variables = []
  for i in range(dimension):
    variables.append(chr(120+i))
  return variables

def create_symbols(dimension:int) -> np.array:
  """
  Create an array of symbols from variables
  :param int dimension: the number of different variables.
  :return: np.array
  """
  variables = create_variables(dimension)
  symbols = []
  for i in variables:
    new_symb = sp.symbols(f'{i}')
    symbols.append(new_symb)
  return symbols
