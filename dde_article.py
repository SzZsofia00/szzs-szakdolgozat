import ddeint
import pysindy as ps
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
import matplotlib.pyplot as plt


def dde_eq(x, t, tau=7, alpha=0.75):
    return x(t) - x(t)**3 - alpha * x(t - tau)

def history(t): #periodic function for past
    return np.sin(t)


dgr = 3
order = 1

h = 0.5 #delta t = 0.025 and we need 4000 samples
N = 4000
T = h * (N-1)
num_of_points = N

s = np.linspace(1,int(8.5/h),int(8.5/h)) #just the index for the running tau
# tau_s = s * h

#minta data.. ehhez hasonlitották az adatukat
ts = np.linspace(0,T,N)
ys = ddeint.ddeint(dde_eq, history, ts).flatten() # generált adat

# plt.plot(ts, ys)
# plt.show()


info = []
Xis = []


for ind in s:
    print(ind)
    tau = ind * h
    ind = int(ind)

    def dde_with_tau(x,t):
        return dde_eq(x,t,tau=tau)

    ys = ddeint.ddeint(dde_with_tau, history, ts).flatten()
    X = np.vstack([ys[ind:], ys[:num_of_points-ind]]).T # X

    feature_library = ps.PolynomialLibrary(degree=dgr)
    feature_library.fit(X)

    theta = feature_library.transform(X)
    theta = theta[:,[0,1,2,3,5,6,9]]

    differentiation_method = ps.FiniteDifference(order=order)

    # print(X.shape)
    # print(ts.shape)
    # print(ts[s+1:].shape)

    X_dot = differentiation_method._differentiate(X[:,0],ts[ind:])

    lasso = Lasso(alpha=0.1,fit_intercept=False)
    lasso.fit(theta,X_dot)

    Xi = np.array(lasso.coef_).T
    # print(Xi)

    celfuggveny = np.linalg.norm(X_dot - theta @ Xi) / X.shape[0]
    # print(celfuggveny)

    info.append([ind, celfuggveny])
    Xis.append(Xi)

info = np.array(info)
print(info)

