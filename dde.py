import ddeint
import pysindy as ps
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression

def mackey_glass(x,t,tau,mu=1,p=2,n=2):
    return -mu * x(t) + (p * x(t-tau)) / (1 + (x(t-tau))**n)

def history(t):
    return 0.5

num_of_points = 50
dgr = 3
order = 2

S = np.arange(1, 30)
info = []

for s in S:
    print(s)
    def mackey_with_tau(x, t):
        return mackey_glass(x, t, tau=s)

    ts = np.linspace(0,s+num_of_points-1,s+num_of_points)
    ys = ddeint.ddeint(mackey_with_tau,history,ts).flatten() # generált adat

    X = np.vstack([ys[-num_of_points+s:], ys[:num_of_points-s]]).T # X

    feature_library = ps.PolynomialLibrary(degree=dgr)
    feature_library.fit(X)
    theta = feature_library.transform(X)

    differentiation_method = ps.FiniteDifference(order=order)
    print(X.shape)
    print(ts.shape)
    print(ts[s+1:].shape)
    X_dot = differentiation_method._differentiate(X,ts[s+1:])

    lasso = Lasso(alpha=0.2,fit_intercept=False)
    lasso.fit(theta,X_dot)

    Xi = np.array(lasso.coef_).T
    print(Xi)

    celfuggveny = np.linalg.norm(X_dot - theta @ Xi)
    print(celfuggveny)

    info.append([s, Xi, celfuggveny])

print(info)

