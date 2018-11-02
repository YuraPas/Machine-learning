
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


def normalize(X):
    n = X.shape[0]
    mean = np.mean(X, axis=1).reshape((n, 1))
    std = np.std(X, axis=1).reshape((n, 1))
    X_new = (X - mean) / std**2
    return X_new, mean, std


# In[8]:


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


# In[9]:


def plot_data(X, y):
    ax = plt.gca()
    ax.scatter(X[:,0], X[:,1], c=(y == 1), cmap=cm_bright)


# In[10]:


def fwd_pass(X, params):
    W = params["W"]
    b = params["b"]
    
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    
    cache = (W, b, Z, A)
    
    return A, cache


# In[33]:


def cost(A, Y):
    m = Y.shape[1]
    
    L = - Y * np.log(A) - (1 - Y) * np.log(1 - A)
    J = np.sum(L) / m
    
    return J


# In[53]:


def bwd_pass(X, Y, cache):
    n, m = X.shape
    (W, b, Z, A) = cache
    
    dZ = A - Y
    dW = 1. / m * np.dot(X, dZ.T).reshape((1, n))
    db = 1. / m * np.sum(dZ)
    
    grads = {"dW" : dW, "db" : db, "dZ" : dZ}
    return grads


# In[13]:


def init_params(n, m):
    W = np.random.randn(1, n) * 0.01
    b = 0
    
    params = {"W" : W, "b" : b}
    return params


# In[14]:


def update_params(params, grads, learning_rate):
    W = params["W"]
    b = params["b"]
    
    dW = grads["dW"]
    db = grads["db"]
    
    params["W"] = W - learning_rate * dW
    params["b"] = b - learning_rate * db
    
    return params


# In[15]:


def fit(X, Y, learning_rate = 0.01, num_iter = 30000, debug = False):
    n, m = X.shape 
    params = init_params(n, m)
    costs = []
    for i in range(num_iter):
        A, cache = fwd_pass(X, params)
        curr_cost = cost(A, Y)
        grads = bwd_pass(X, Y, cache)
        
        params = update_params(params, grads, learning_rate)
        
        if debug and i % 1000 == 0:
            print("{}-th iteration: {}".format(i, curr_cost))
            costs.append(curr_cost)
    
    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("Iteration, *1000")
    plt.show()
    
    return params    


# In[54]:


data_columns = ["exam1", "exam2"]
target_column = "submitted"
df = pd.read_csv("sats.csv")
X, Y = df[data_columns].values, df[target_column]
print('Training set: X={}, y={}'.format(X.shape, Y.shape))


# In[55]:


plot_data(X, Y)


# In[56]:


Y = Y.values.reshape((df.shape[0], 1))


# In[57]:


X, Y = X.T, Y.T


# In[58]:


X, mean, std = normalize(X)


# In[59]:


X[:,:5]


# In[60]:


Y.shape


# In[61]:


mean


# In[62]:


std


# In[72]:


params = fit(X, Y, learning_rate = 0.1, num_iter = 200000, debug = True)


# In[73]:


print(params)

