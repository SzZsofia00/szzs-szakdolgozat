import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

from solve_methods import *
import numpy as np

#differential equation dxdt = -x
def diff_eq(t,x):
    return -x

def exact_sol(t):
    return np.exp(-t)

def derivative(t):
    return -np.exp(-t)

#exact solution
t_exact = np.linspace(0,5,11)
sol = exact_sol(t_exact)

#-------------------------#

#initial conditions
time = [0,5]
init = [1]
h = 0.5

#-------------------------#
so = SolveODE(diff_eq,time,init,h)
t_nm = so.create_time_points()

#solve euler
euler = (so.solve_with_numerical_method('euler')).flatten()

#draw tangent: y = slope * (t-t0) + y0
t0 = 0
t_tangent = np.linspace(t0-0.25,t0+0.75,50)
y0 = exact_sol(t0)
slope = derivative(t0)
line = slope * (t_tangent - t0) + y0

t1 = t_nm[1]
t1_tangent = np.linspace(t1-0.25,t1+0.75,50)
y1 = euler[1]
slope1 = diff_eq(t1,y1)
line1 = slope1 * (t1_tangent - t1) + y1

def plot_euler():
    plt.figure(figsize=(10,6))
    plt.plot(t_exact,sol,color='grey',linestyle='-',label="$x(t)=e^{-t}$")
    plt.plot(t_nm, euler, 'go-', label="Euler módszer")

    plt.plot(t_tangent,line,'r--')
    plt.scatter([t0],[y0],color='red')
    plt.vlines(t0+h,ymin=0.3,ymax=0.75,linestyle="--",color="red")

    plt.plot(t1_tangent, line1, 'blue', linestyle="--")
    plt.scatter([t1], [y1], color='blue')
    plt.vlines(t1 + h, ymin=0, ymax=0.5, linestyle="--", color="blue")

    plt.xticks(t_exact,["$x_0$","$x_0+h$","$x_0+2h$","","","","","","","","$x_0+Nh$"])
    plt.yticks([])

    plt.text(-0.15,0.95,"$P$",fontsize=16,color="black")
    plt.text(0.32, 0.45, "$P'$", fontsize=16, color="black")
    plt.text(0.79, 0.2, "$P''$", fontsize=16, color="black")

    plt.xlabel("$t$",fontsize=16)
    plt.ylabel("$x$",fontsize=16)
    plt.legend()
    plt.savefig("Euler_method.png")
    # plt.show()

plot_euler()

#-------------------------#

#solve midpoint
midpoint = so.solve_with_numerical_method('midpoint_euler').flatten()
print(midpoint)

t_mid = t_nm[0] + h/2
x_mid = 1 + (h/2) * diff_eq(t_nm[0],1)
slope = diff_eq(t_mid,x_mid)

t_tangent = np.linspace(t_mid-0.25,t_mid+0.5,50)
# y0 = midpoint[0]
# slope = diff_eq(t_mid,y0)
# line = slope * (t_tangent - t_mid) + y0
line = x_mid + slope * (t_tangent - t_mid)


def plot_midpoint():
    plt.figure(figsize=(10, 6))
    plt.plot(t_exact, sol, color='grey', linestyle='-', label="Exact Solution")
    plt.plot(t_nm, midpoint, 'go-', label="Midpoint Euler method")

    plt.plot(t_tangent, line, 'r--')
    plt.scatter([t_mid], [x_mid], color='red')
    plt.vlines(t_mid + h/2, ymin=0.3, ymax=0.75, linestyle="--", color="red")


    plt.xlabel("$t$", fontsize=16)
    plt.ylabel("$x$", fontsize=16)
    plt.legend()
    plt.show()

# plot_midpoint()