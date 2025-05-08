import ddeint
import pysindy as ps
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
import matplotlib.pyplot as plt

def dde_optimizer(theta,X_dot):
    Q = []

    lls = LinearRegression(fit_intercept=False)

    ridge = Ridge(alpha=0.5,fit_intercept=False)
    ridge.fit(theta,X_dot)
    Xi = np.array(ridge.coef_)

    C = np.linalg.norm(X_dot - theta @ Xi)

    while True:
        mask = [i for i in range(theta.shape[1]) if i not in Q]

        theta_msk = theta[:,mask]
        Xi_msk = Xi[mask]

        C = np.linalg.norm(X_dot - theta_msk @ Xi_msk)

        candidate_error = []

        for i,idx in enumerate(mask): #kinda a map with i (sorszam) and corresponding index in the original Xi

            if theta_msk.shape[1] > 1:
                theta_tmp = theta_msk.copy()
                theta_tmp = np.delete(theta_tmp,i,axis=1)
                lls.fit(theta_tmp,X_dot)
                Xi_tmp = np.array(lls.coef_)

                err = np.linalg.norm(X_dot - theta_tmp @ Xi_tmp)
                candidate_error.append((err,idx))
            else:
                candidate_error.append((C * 2, 0))

        min_error,min_index = min(candidate_error)

        if min_error < C * 1.1:
            Xi[min_index] = 0
            Q.append(min_index)
            continue

        ridge.fit(theta_msk, X_dot)
        Xi_tmp = np.array(ridge.coef_)
        i = 0
        for j in range(len(Xi)):
            if j not in Q:
                Xi[j] = Xi_tmp[i]
                i = i + 1
            else:
                Xi[j] = 0.

        break
    return Xi

def dde_eq3(x,t,tau=1,mu=1,p=2,n=2):
    return -mu * x(t) + (p * x(t-tau)) / (1 + (x(t-tau))**n)

def dde_eq(x, t, tau=7, alpha=0.75):
    return x(t) - x(t)**3 - alpha * x(t - tau)

def history(t): #periodic function for past
    return np.sin(t)

def dde_eq_generic(x, t, tau, params):
    xx = x(t)
    xd = x(t - tau)
    return np.dot(params, [1, xx, xd, xx**2, xd**2, xx**3, xd**3])


dgr = 3
order = 1

h = 0.025 #delta t = 0.025 and we need 4000 samples
N = 4000
T = h * (N-1)
num_of_points = N

# s = np.linspace(1,int(8.5/h),int(8.5/h))
s = np.linspace(int(3/h),int(8.5/h),int(5.5/h)) #just the index for the running tau
# tau_s = s * h

#minta data.. ehhez hasonlitották az adatukat
ts = np.linspace(0,T,N)
data = ddeint.ddeint(dde_eq, history, ts).flatten() # generált adat

#noise
# noise = np.random.normal(loc=0.0, scale=0.5, size=data.shape) #scale = standard deviation
noise = np.zeros(data.shape)
ys = data + noise
# ys = ys + 0.011  * np.random.randn(*ys.shape)

plt.plot(ts, ys)
plt.plot(ts, data,color='red')
plt.show()

info = []
Xis = []

length = N - len(s)

for ind in s:
    print(int(ind))
    tau = ind * h
    ind = int(ind)

    X = np.vstack([ys[ind:], ys[:num_of_points-ind]]).T # X
    X = X[0:length]

    feature_library = ps.PolynomialLibrary(degree=dgr)
    feature_library.fit(X)

    theta = feature_library.transform(X)
    theta = theta[:,[0,1,2,3,5,6,9]]

    differentiation_method = ps.FiniteDifference(order=order)
    X_dot = differentiation_method._differentiate(X[:, 0], ts[ind:ind + len(X)])

    lasso = Lasso(alpha=0.01,fit_intercept=False)
    lasso.fit(theta,X_dot)

    # Xi = np.array(lasso.coef_).T #ha Xi lassoval
    Xi = dde_optimizer(theta,X_dot) #ha Xi a cikk szerinti modon

    # visszafejtjük X-t a prediktált Xi-ből
    X_reconstructed = ddeint.ddeint(
        lambda x, t: dde_eq_generic(x, t, ind * h, Xi),
        history, ts).flatten()

    # Calculate reconstruction error
    Z = np.sum(X[:, 0] ** 2)
    # prediktált X - valós X
    E = np.sum((X_reconstructed - ys) ** 2) / Z

    info.append([ind*h, E])

info = np.array(info)

plt.plot(info[:,0],info[:,1],marker='*',markerfacecolor='r',markersize=3,label='DDE')
plt.show()


# data + error
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(ts, ys)
ax[0].plot(ts, data,color='red')
ax[1].plot(info[:,0],info[:,1],marker='*',markerfacecolor='r',markersize=3,label='DDE')
# plt.savefig("dde-data-error-noise-0.5.pdf")
plt.show()

