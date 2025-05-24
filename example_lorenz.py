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
init = [1,1,1]
time = [0,40]
step_size = 0.001

optimizer = 'stlsq'

# generate data
so = SolveODE(diff_eq,time,init,step_size)
num_method = 'midpoint_euler'
noise = True
scale = 0.1
data_mtx = so.get_matrix_with_noise(numerical_method=num_method,be_noise=noise,scale=scale)
data_mtx_without_noise = so.get_matrix_with_noise(numerical_method=num_method,be_noise=False,scale=scale)
t = so.create_time_points()

#slice
train_end_time = 20
end_time_index = int(train_end_time/step_size)
data_train = data_mtx[:, :end_time_index]
t_train = t[:end_time_index]

# pysindy model
pf = PysindyFunctions(data_train, t_train,optimizer)
model = pf.model_fit()
pf.print_model_equations(model)

# simulate
init_test = data_train[:, -1].tolist()
time_test = [train_end_time,40]
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
    # plt.savefig('lorenz.pdf')
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
    # plt.savefig('lorenz2.pdf')
    plt.show()

def plot_true_and_sim():
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 9))

    end_time_index = int(train_end_time / step_size)
    sim_end_index = int(10/step_size)

    for i in range(3):
        axs[i].plot(t[:end_time_index], data_mtx[i,:end_time_index], "blue", label="noisy data")
        axs[i].plot(t_test, data_mtx_without_noise[i,end_time_index:], "k", label="true")
        axs[i].plot(t_test, data_test_sim[i], "r", label="model simulation")
        # axs[i].plot(t_test[:sim_end_index], data_mtx_without_noise[i, end_time_index:int(30/step_size)], "k", label="true")
        # axs[i].plot(t_test[:sim_end_index], data_test_sim[i,:sim_end_index], "r", label="model simulation")
        axs[i].legend()

    axs[0].set(ylabel="$x$")
    axs[1].set(ylabel="$y$")
    axs[2].set(xlabel="$t$", ylabel="$z$")
    # plt.savefig('lorenz4.pdf')
    fig.show()

# plot_lorenz_3d(data_train,data_test_sim)
# plot_lorenz_3d_true_sim(data_train,data_test_sim)
plot_true_and_sim()