import matplotlib.pyplot as plt

from differential_equations import *
from solve_differential_equation import *
from pysindy_methods import *

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
# init_test = [8,7,15]
init_test = data_mtx[:, -1].tolist()
time_test = [20,40]
so_test = SolveODE(diff_eq,time_test,init_test,step_size)
data_test = so_test.get_matrix_with_noise(numerical_method=num_method,be_noise=noise)
t_test = so_test.create_time_points()

data_test_sim = model.simulate(init_test,t_test).T

#-----------------------------------------------------#

def plot_lorenz_3d(data,sim):
    x_data, y_data, z_data = data[0], data[1], data[2]
    x2_data, y2_data, z2_data = sim[0], sim[1], sim[2]
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot(x_data,y_data,z_data,color='blue',label='true')
    ax.plot(x2_data,y2_data,z2_data,color='red',label='simulation')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Lorenz Model")
    ax.legend()
    plt.show()

def plot_lorenz_3d_true_sim(data,sim):
    x_data, y_data, z_data = data[0], data[1], data[2]
    x2_data, y2_data, z2_data = sim[0], sim[1], sim[2]
    fig, ax = plt.subplots(1, 2,subplot_kw={'projection': '3d'})
    ax[0].plot(x_data, y_data, z_data, color='blue')
    ax[1].plot(x2_data, y2_data, z2_data, color='red')
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_zlabel("Z")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].set_zlabel("Z")
    plt.show()

def plot_true_and_sim():
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 9))
    for i in range(3):
        axs[i].plot(t, data_mtx[i], "k", label="true simulation")
        axs[i].plot(t_test, data_test_sim[i], "r", label="model simulation")
        axs[i].legend()

    axs[0].set(ylabel="$x$")
    axs[1].set(ylabel="$y$")
    axs[2].set(xlabel="$t$", ylabel="$z$")
    # plt.savefig('high-rho-lorenz-xyz.pdf')
    fig.show()

plot_lorenz_3d(data_mtx,data_test_sim)
plot_lorenz_3d_true_sim(data_mtx,data_test_sim)
plot_true_and_sim()