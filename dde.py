import ddeint
import itertools

import numpy as np

from symbol_creation import *


def mackey_glass(x,t,mu=1,p=2,n=2):
    return -mu * x(t) + (p * x(t-1)) / (1 + (x(t-1))**n)

def history(t):
    return 0.5

# ezt itt majd dinamizálni num = 10 -t külső változóban és akkor data-ban is 9-t átírni
ts = np.linspace(1,10,10)
ys = ddeint.ddeint(mackey_glass,history,ts).flatten()
data = np.vstack([ys[-9:], ys[:9]])

symb = CreateSymbols(1).create_symbold_for_dde(2)
dgr = 2

fv = []
for d in range(dgr + 1):
    for i in itertools.combinations_with_replacement(symb,d):
        fv.append(sp.Mul(*i))

x_current = data[0][0]
x_delayed = data[1][0]

print(x_current)
print(x_delayed)

def AB2(func,tn1,xn1,tn0,xn0,h): #xn1,tn1 aktuális és xn0,tn1 megelőző
  return xn1 + h/2 * (3 * func(tn1,xn1) - func(tn0,xn0))

h = 1/2
num_of_samples = int((ts[-1] - ts[0]) / h) + 1
time = np.linspace(ts[0],ts[-1],num_of_samples)

zero_mtx = np.zeros(num_of_samples)
zero_mtx[0] = x_delayed
zero_mtx[1] = x_current

print(zero_mtx)

# for i in range(time - 1)
#     matrix[i+1] = AB2(mackey_glass,)

