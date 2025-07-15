import numpy as np
from scipy.optimize import minimize
from functools import partial
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Question 21
def f_objective(theta, X, Y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    num_features=X.shape[1]
    num_instances=X.shape[0]
    loss=0
    for i in range(num_instances):
        loss+=np.logaddexp(0,-Y[i]*np.dot(theta,X[i]))
    loss=loss/num_instances
    reg_term=l2_param*(np.dot(theta,theta))
    return (loss+reg_term)
#Question 22
def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    num_features=X.shape[1]
    objective=partial(objective_function, X=X, Y=y, l2_param=l2_param)
    w_0=np.ones(num_features)
    w=minimize(objective,w_0).x
    return w
#The paths here are adjusted for my desktop however it would need to be changed if running it on another 
#desktop according to the correct path where the files are 
with open("Downloads/hw2/X_train.txt") as textFile:
    x_train = [line.split() for line in textFile]
x_train=[[float(train) for train in r[0].split(',')] for r in x_train]
with open("Downloads/hw2/X_val.txt") as textFile:
    x_val = [line.split() for line in textFile]
x_val=[[float(v) for v in r[0].split(',')] for r in x_val]
with open("Downloads/hw2/y_train.txt") as textFile:
    y_train = [line.split() for line in textFile]
y_train=[[float(v) for v in r[0].split(',')] for r in y_train]
with open("Downloads/hw2/y_val.txt") as textFile:
    y_val = [line.split() for line in textFile]
y_val=[[float(v) for v in r[0].split(',')] for r in y_val]
x_train, x_val, y_train, y_val = np.array(x_train),np.array(x_val),np.array(y_train),np.array(y_val)
SS = StandardScaler()
X_train = SS.fit_transform(x_train)
X_val = SS.fit_transform(x_val)
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))
y_train[y_train == 0] = -1
y_val[y_val == 0] = -1
theta=fit_logistic_reg(X_train, y_train, f_objective, l2_param=1)
#test for 22: 
print(theta)

#Question 23
def log_likelihood(theta,X,y): 
    num_features=X.shape[1]
    num_instances=X.shape[0]
    loss=0
    for i in range(num_instances):
        loss += np.logaddexp(0,-y[i]*np.dot(theta, X[i]))
    return(-loss)
L2=list([10.0**i for i in np.arange(-4,2,0.5)])
l=[]
for l2reg in L2:
    theta = fit_logistic_reg(X_train, y_train, f_objective, l2_param=l2reg)
    result = log_likelihood(theta,X_val,y_val)
    l.append(result)   
print(l)
log_L2 = [np.log10(i) for i in L2]
plt.plot(log_L2,l)
plt.show()

#Question 24 Optional
from sklearn import datasets
from sklearn.calibration import calibration_curve

plt.plot([0, 1], [0, 1])
num_instances, num_features = X_val.shape
theta_optimize=fit_logistic_reg(X_train, y_train, f_objective, l2_param=10**(-2))
prob_pos = np.dot(X_val,theta_optimize)
for i in range(num_instances):
    prob_pos[i] = 1/(1+np.exp(-prob_pos[i]))
prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
fraction_of_positives, mean_predicted_value = calibration_curve(y_val, prob_pos, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s" % ('Logistic'))
plt.show()
print(fraction_of_positives)