import numpy as np


def Tablet(X):
    if len(X.shape) < 2:
        X = np.expand_dims(X,0)
    Xsqure = X**2
    fx = 10e+6*Xsqure[:,0]
    fx += np.sum(Xsqure[:,1:], axis=-1)
    return fx

def Quadric(X):
    if len(X.shape) < 2:
        X = np.expand_dims(X,0)
    Xsqure = X**2
    x_dim = X.shape[1]
    fx = np.zeros(X.shape[0])
    for i in range(x_dim):
        tfx = np.zeros(X.shape[0])
        for j in range(i+1):
            tfx += Xsqure[:,j]
        fx += tfx**2
    return fx

def Rosenbrock(X):
    if len(X.shape) < 2:
        X = np.expand_dims(X,0)
    x_dim = X.shape[1]
    fx = np.zeros(X.shape[0])
    for i in range(x_dim-1):
        fx += 100.0*(X[:,i+1] -X[:,i]**2)**2 + (X[:,i]-1.0)**2
    return fx

def Griewank(X):
    if len(X.shape) < 2:
        X = np.expand_dims(X,0)
    x_dim = X.shape[1]
    sumfx = np.zeros(X.shape[0])
    mulfx = np.ones(X.shape[0])
    for i in range(x_dim):
        sumfx += X[:,i]**2
        mulfx *= np.cos(X[:,i]/np.sqrt(i+1.0))
    fx = sumfx/4000.0 - mulfx + 1
    return fx

def Rastrigrin(X):
    if len(X.shape) < 2:
        X = np.expand_dims(X,0)
    x_dim = X.shape[1]
    fx = np.zeros(X.shape[0])
    for i in range(x_dim):
        fx += (X[:,i]**2 - 10.0*np.cos(2*np.pi*X[:,i]) + 10)
    return fx

def SchafferF7(X):
    if len(X.shape) < 2:
        X = np.expand_dims(X,0)
    x_dim = X.shape[1]
    fx = np.zeros(X.shape[0])
    Xsqure = X**2
    for i in range(x_dim-1):
        fx += ((Xsqure[:,i] + Xsqure[:,i+1])**0.25)*(np.sin(50*(Xsqure[:,i]+Xsqure[:,i+1])**0.1)+1.0)
    return fx 