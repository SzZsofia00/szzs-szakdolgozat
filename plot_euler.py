import matplotlib.pyplot as plt
from solve_differential_equation import *
import numpy as np

def diff_eq(t,x):
    return 9 * t**2 - 22*t + 10

def exact_sol(t):
    return 3*t**3 - 11*t**2 + 10*t + 1

def derivative(t):
    return 9 * t**2 - 22*t + 10

#exact solution
t_exact = np.linspace(0,3,100)
sol = exact_sol(t_exact)

#initial conditions
time = [0,3]
init = [1]
h = 0.25

#-------------------------#
so = SolveODE(diff_eq,time,init,h)
t_nm = so.create_time_points()

#solve euler
euler = (so.solve_with_numerical_method('euler')).flatten()

#draw tangent: y = slope * (t-t0) + y0
t0 = 0
t_tangent = np.linspace(t0-0.25,t0+0.5,50)
y0 = exact_sol(t0)
slope = derivative(t0)
line = slope * (t_tangent - t0) + y0

t1 = t_nm[1]
t1_tangent = np.linspace(t1-0.25,t1+0.5,50)
y1 = euler[1]
slope1 = diff_eq(t1,y1)
line1 = slope1 * (t1_tangent - t1) + y1

def plot_euler():
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 20
    # plt.rcParams["mathtext.default"] = 18

    plt.figure(figsize=(10,6))
    plt.plot(t_exact,sol,color='black',linestyle='-',label="$x(t)=3t^3-11t^2+10t+1$")
    plt.plot(t_nm, euler, color='#C32148',marker='o', linestyle='-', label=r"$\text{Euler-módszer}$")

    plt.plot(t_tangent,line,color='#00468C',linestyle='--')
    # plt.scatter([t0],[y0],color='black')
    plt.vlines(t0+h,ymin=-0.5,ymax=y1+1,linestyle="--",color="#00468C")

    plt.plot(t1_tangent, line1, '#00468C', linestyle="--")
    # plt.scatter([t1], [y1], color='#00468C')
    plt.vlines(t1 + h, ymin=-0.5, ymax=5.5, linestyle="--", color="#00468C")

    # plt.xticks(t_nm,["$x_0$","$x_0+h$","","$x_0+3h$","","","","","","","","","$x_0+Nh$"])
    plt.xticks(t_nm, ["$t_0$","", "$t_0+2h$", "", "", "", "", "", "", "", "", "", "$t_0+T$"])
    plt.yticks([])

    plt.text(-0.15,0.95,r"$\text{P}$",fontsize=16,color="black")
    plt.text(0.1, y1, r"$\text{P'}$", fontsize=16, color="black")
    plt.text(0.5, 5.3, r"$\text{P''}$", fontsize=16, color="black")

    plt.ylim(-0.5)
    plt.xlim(-0.5)
    # plt.xlabel("$t$",fontsize=20)
    plt.ylabel("$x$",fontsize=20)
    plt.legend()
    # plt.savefig("euler.pdf")
    plt.show()

plot_euler()