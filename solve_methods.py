from scipy.integrate import solve_ivp
from numerical_method import *
from exceptions import *
import sys

class SolveODE:
    def __init__(self,func,time:list,init:list,step_size:float):
        """
        Class that contains methods to solve a differential equation with initial values.
        :param func: Differential equation
        :param list time: Time period
        :param list init: Initial values
        :param step_size: Step size between time points
        """
        self.func = func
        self.time = time
        self.init = init
        self.step_size = step_size
        self.number_of_samples = int((self.time[1] - self.time[0]) / self.step_size) + 1

    def create_zero_matrix(self) -> np.array:
        """
        This class creates a zero matrix with given dimension.
        :return: np.array
        """
        empty_matrix = np.zeros((len(self.init), self.number_of_samples))
        return empty_matrix

    def create_time_points(self) -> np.array:
        """
        This class creates an array with the time points with given step_size.
        :return: np.array
        """
        t = np.linspace(self.time[0], self.time[1], self.number_of_samples)
        return t

    def fill_matrix_with_init(self) -> np.array:
        """
        Fills the zero_matrix with init values.
        :return: np.array
        """
        zero_matrix = self.create_zero_matrix()
        zero_matrix[:, 0] = self.init
        return zero_matrix

    def solve_with_numerical_method(self,numerical_method:str) -> np.array:
        """
        Solves the given differential equation with given numerical method.
        :param str numerical_method: Numerical method.
        :return: np.array
        """
        try:
            matrix = self.fill_matrix_with_init()
            t = self.create_time_points()

            for i in range(len(t) - 1):
                nm = NumericalMethods(self.func, t[i], matrix[:, i], self.step_size)
                matrix[:, i + 1] = getattr(nm, numerical_method)()
            return matrix

        except DimensionError as e:
            print(f"Error with the function:\n {e}")
            sys.exit("Stopping the program due to an error.")

    def solve_with_RK45(self) -> np.array:
        """
        Solves the given differential equation with Runge Kutta 45 method.
        :return: np.array
        """
        try:
            matrix = self.fill_matrix_with_init()
            t = self.create_time_points()

            sol = solve_ivp(self.func, self.time, matrix[:, 0], t_eval=t)
            matrix[:, :] = sol.y
            return matrix

        except DimensionError as e:
            print(f"Error with the function:\n {e}")
            sys.exit("Stopping the program due to an error.")

    def generate_data(self,numerical_method:str='RK45') -> np.array:
        """
        Method to generate a data matrix. Using the chosen numerical method
        if given. Otherwise, RK45 with solve_ivp.
        :param numerical_method: The given numerical method.
        :return: np.array
        """
        if numerical_method == 'RK45':
            matrix = self.solve_with_RK45()
        else:
            matrix = self.solve_with_numerical_method(numerical_method)
        return matrix

    def generate_noise(self,be_noise:bool=False,scale:float=1.0) -> np.array:
        """
        Generate noise for the data if it's true. If false generate zero_matrix so no noise.
        :param bool be_noise: True if we want noise.
        :param float scale: Standard deviation of the distribution.
        :return: np.array
        """
        if be_noise:
            noise = np.random.normal(loc=0.0, scale=scale, size=self.number_of_samples)
        else:
            noise = np.zeros(self.number_of_samples)
        return noise

    def get_matrix_with_noise(self,numerical_method:str='RK45',
                              be_noise:bool=False,scale:float=1.0) -> np.array:
        """
        Create matrix with noise.
        :param str numerical_method: Numerical method.
        :param bool be_noise: True if we want noise.
        :param float scale: Standard deviation of the distribution.
        :return: np.array
        """
        matrix_clean = self.generate_data(numerical_method)
        noise = self.generate_noise(be_noise,scale)
        matrix = matrix_clean + noise
        return matrix

    def get_matrix_rows(self,numerical_method:str='RK45',
                        be_noise:bool=False,scale:float=1.0) -> np.array:
        """
        Gets the rows of the matrix.
        :param str numerical_method: Numerical method.
        :param bool be_noise: True if we want noise.
        :param float scale: Standard deviation of the distribution.
        :return: np.array
        """
        matrix = self.get_matrix_with_noise(numerical_method,be_noise,scale)

        if len(matrix) == 1:
            x = matrix
            return x
        elif len(matrix) == 2:
            x, y = matrix
            return x, y
        elif len(matrix) == 3:
            x, y, z = matrix
            return x, y, z
