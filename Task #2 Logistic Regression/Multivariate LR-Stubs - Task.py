
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def load_datasets(filename, data_columns, target_column):
    df = pd.read_csv(filename)
    X, y = df[data_columns], df[target_column]
    return X, y


# In[3]:


def plot_data(A, b, test = False):
    positive_indices = np.where(b == 1)[0]
    negative_indices = np.where(b == 0)[0]
    
    plt.scatter(A.iloc[positive_indices, 0], A.iloc[positive_indices, 1], marker='x', c= 'yellow' if test else 'green')
    plt.scatter(A.iloc[negative_indices, 0], A.iloc[negative_indices, 1], marker='+', c= 'blue' if test else 'red')
    
   # plt.show()


# In[4]:


def decision_boundary(X, y, mean, std, theta):
    pass


# In[5]:


def normalize(X):
    mean = None
    std = None
    X_new = None
    return X_new, mean, std


# In[6]:


def prepare_X(X):
    m = X.shape[0]
    ones = np.ones((m, 1))
    X_new = np.column_stack((ones, X))
    return X_new


# In[7]:


def sigmoid(z):
    g = None
    return g


# In[8]:


def h(X, theta):
    z = None
    return sigmoid(z)


# In[9]:


def cost_function(X, y, theta):
    m = X.shape[0]
    if m ==0:
        return None
    
    J = None
    return J


# In[10]:


def derivative_theta(X, y, theta):
    m = X.shape[0]
    if m == 0:
        return None

    d_theta = None
    
    return d_theta


# In[11]:


def gradient_descent(X, y, theta, alpha, epsilon, num_iters, print_J = True):
    m = X.shape[0]
    J_history = []
    
    J = None
    
    if print_J == True:
        print(J)
    J_history.append(J)
    for i in range(num_iters):
        
        theta = None
        
        J = None
        
        J_history.append(J)
        
        if i % 1000 == 0 and print_J == True:
            print(J)
        
        if abs(J-J_history[-2]) < epsilon:
            break
            
    return theta, J_history


# In[12]:


def predict(X, mean, std, theta):
    predictions = None
    return predictions


# In[13]:


data_columns = ["exam1", "exam2"]
target_column = "submitted"
X, y = load_datasets('sats.csv', data_columns, target_column)
print('Training set: X={}, y={}'.format(X.shape, y.shape))


# In[14]:


# Спробувати на цьому датасеті після написання всіх функцій і відлагодження на попередньому
#data_columns = ["test1", "test2"]
#target_column = "passed"
#X, y = load_datasets('tests.csv', data_columns, target_column)
#print('Training set: X={}, y={}'.format(X.shape, y.shape))


# In[15]:


plot_data(X, y)


# In[16]:


X_new, mean, std = normalize(X)


# In[17]:


X_new = prepare_X(X_new)
y_new = y.reshape((X.shape[0], 1))


# In[ ]:


theta = np.zeros((X_new.shape[1], 1))


# In[ ]:


cost_function(X_new, y_new, theta)


# In[ ]:


new_theta, Js = gradient_descent(X_new, y_new, theta, 0.01, 1e-7, 1000000, False)
print(new_theta, len(Js))


# In[ ]:


cost_function(X_new, y_new, new_theta)


# In[ ]:


decision_boundary(X, y, mean, std, new_theta)


# In[ ]:


T = np.array([[40, 40], [60, 65]])
print(predict(T, mean, std, new_theta))

