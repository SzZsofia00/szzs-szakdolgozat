import matplotlib.pyplot as plt

from differential_equations import *
from solve_methods import *
from pysindy_methods import *

# cases
# og butterfly: 10, 28, 8/3
# stable fixed points: 10 5 8/3
# low sigma: 2 28 8/3
# high rho: 10 100 8/3

# Differential equation
sigma = 10
rho = 28
beta = 8/3
de = ExampleDifferentialEquations()
diff_eq = lambda t, xyz: de.lorenz(t,xyz,sigma=sigma,rho=rho,beta=beta)

# System
init = [0,1,1]
time = [0,20]
step_size = 0.001

#-------------------------------------------------------------------#

# generate data
so = SolveODE(diff_eq,time,init,step_size)
num_method = 'euler'
noise = False
data_mtx = so.get_matrix_with_noise(numerical_method=num_method,be_noise=noise)

# pysindy model
t = so.create_time_points()
pf = PysindyFunctions(data_mtx, t)
model = pf.model_fit()
pf.print_model_equations(model)

# simulate
init_test = [8,7,15]
time_test = [0,30]
so_test = SolveODE(diff_eq,time_test,init_test,step_size)
data_test = so_test.get_matrix_with_noise(numerical_method=num_method,be_noise=noise)
t_test = so_test.create_time_points()

data_test_sim = model.simulate(init_test,t_test)

#-----------------------------------------------------#

def plot_lorenz_3d(data,name):
    x_data, y_data, z_data = data[0], data[1], data[2]
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot(x_data,y_data,z_data)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Lorenz Model")
    # plt.savefig(name)
    plt.show()

def plot_true_and_sim():
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 9))
    for i in range(3):
        axs[i].plot(t_test, data_test[i], "k", label="true simulation")
        axs[i].plot(t_test, data_test_sim[:, i], "r--", label="model simulation")
        axs[i].legend()

    axs[0].set(ylabel="$x$")
    axs[1].set(ylabel="$y$")
    axs[2].set(xlabel="$t$", ylabel="$z$")
    plt.savefig('high-rho-lorenz-xyz.pdf')
    fig.show()

#plot true function
plot_lorenz_3d(data_mtx,'high-rho-lorenz-true.pdf')
#plot simulated function
plot_lorenz_3d(data_test_sim.T, 'high-rho-lorenz-sim.pdf')
#plot axis from true and sim
plot_true_and_sim()
