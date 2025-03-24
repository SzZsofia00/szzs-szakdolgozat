import ddeint
import itertools
from symbol_creation import *


def mackey_glass(x,t,mu=1,p=2,n=2):
    return -mu * x(t) + (p * x(t-1)) / (1 + (x(t-1))**n)

def history(t):
    return 0.5

# ezt itt majd dinamizálni num = 10 -t külső változóban és akkor data-ban is 9-t átírni
ts = np.linspace(0,10,10)
ys = ddeint.ddeint(mackey_glass,history,ts).flatten()
data = np.vstack([ys[-9:], ys[:9]])

symb = CreateSymbols(1).create_symbold_for_dde(2)
dgr = 2

fv = []
for d in range(dgr + 1):
    for i in itertools.combinations_with_replacement(symb,d):
        fv.append(sp.Mul(*i))

X_current = data[0].reshape(-1, 1)
X_delayed = data[1].reshape(-1, 1)

def AB2(func,tn,xn,h):
  return xn + h/2 * (3 * func(tn,xn) - func(tn-h,xn-h))
