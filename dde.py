import ddeint
import pysindy as ps
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
import matplotlib.pyplot as plt

def mackey_glass(x,t,tau=7,mu=1.2,p=1.6,n=2, alpha=0.75):
    #return -mu * x(t) + (p * x(t-tau)) / (1 + (x(t-tau))**n)
    #return - p * x(t - tau)
    return x(t) - x(t)**3 - alpha * x(t - tau)

def history(t):
    return np.sin(t)


dgr = 3
order = 2

h = 0.1
T = 1000

S = np.arange(1, 30)*10



ts = np.linspace(0,T,int(T/h))
ys = ddeint.ddeint(mackey_glass,history,ts).flatten() # generált adat
plt.plot(ts, ys)
plt.show()
num_of_points = int(T/h)

info = []
Xis = []

for s in S:
    print(s)

    X = np.vstack([ys[s:], ys[:num_of_points-s]]).T # X

    feature_library = ps.PolynomialLibrary(degree=dgr)
    feature_library.fit(X)

    theta = feature_library.transform(X)
    theta = theta[:,[0,1,2,3,5,6,9]]

    differentiation_method = ps.FiniteDifference(order=order)

    print(X.shape)
    print(ts.shape)
    print(ts[s+1:].shape)

    X_dot = differentiation_method._differentiate(X[:,0],ts[s:])

    lasso = Lasso(alpha=0.1,fit_intercept=False)
    lasso.fit(theta,X_dot)

    Xi = np.array(lasso.coef_).T
    print(Xi)

    celfuggveny = np.linalg.norm(X_dot - theta @ Xi) / X.shape[0]
    print(celfuggveny)

    info.append([s, celfuggveny])
    Xis.append(Xi)
info = np.array(info)
print(info)

