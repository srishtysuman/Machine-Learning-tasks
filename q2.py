# -*- coding: utf-8 -*-
"""q2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fb6PGDLjZeEBm4zIcVy3BicyPmsSxBB6

#Question 2: SVM
"""

import numpy as np
from sklearn.svm import SVC
from cvxopt import matrix, solvers
from sklearn import metrics

n = 100
x = np.random.uniform(-3,3,(100,2))
for i in range(100):
  x[i][0]=np.random.uniform(-3,3)
  x[i][1]=np.random.uniform(-3,3)
y = np.random.randint(-1,1,(n,1))
a = np.array([[1,0], [0,1]])
s = np.std(a)
w = np.random.normal(0,s,(2,1))
b = np.random.normal(0,1)

def predict(x,b,w):
  if ((np.dot(w.T,x)+b)) > 0:
    return 1
  else:
    return -1

for i in range(n):
  y[i]=predict(x[i],b,w)
print(w)

import matplotlib.pyplot as plt
import pandas as pd

plt.scatter(x[:,0], x[:,1], c=['red' if y_i == -1 else 'blue' for y_i in y], facecolors='none',s=10 )
plt.ylabel('x[1]')
plt.xlabel('x[0]')

plt.show()

def train_svm(xq,yq,c):
    P=np.zeros(shape=(103,103))
    P[0][0]=0.5
    P[1][1]=0.5
    q=np.zeros(shape=(103,1))
    for i in range(2,102):
      q[i]=c
    G=np.zeros(shape=(200,103))
    for i in range(100):
      G[i][102]=-1*yq[i]
      for j in range(2):
        G[i][j]= -(yq[i]*xq[i][j])
    for i1 in range(100):
      for j1 in range(2,102):
        if j1-i1==2:
          G[i1][j1]=-1
    for i in range(100,200):
      for j in range(2,102):
        if i-j == 98:
          G[i][j]=-1
    h=np.zeros(shape=(200,1))
    for i in range(100):
      h[i]=-1.0
    
    P1=matrix(P,tc='d')
    q1=matrix(q,tc='d')
    G1=matrix(G,tc='d')
    h1=matrix(h,tc='d')
    #A=matrix(y)
    
    sol = solvers.qp(P1, q1, G1, h1)
    return sol

def acc(xin,yin,bo,wo,t):
  t1=t
  y_pred= np.random.randint(-1,1,(t1,1))
  for i in range(t1):
    y_pred[i] = predict(xin[i],bo,wo)
  print("Accuracy:",metrics.accuracy_score(yin, y_pred))
  return metrics.accuracy_score(yin,y_pred)

def call_fun(x_vec,y_vec):
  margin=np.zeros((500,1))
  accuracy=np.zeros((500,1))
  w_vec=np.zeros((500,2))
  b_vec=np.zeros((500,1))
  w_opt=np.zeros((2,1))
  c_vec=np.zeros((500,1))
  b_opt=0
  C=0.0001
  iteration=500
  for it in range(iteration):
      C+=0.1
      x1=x_vec
      y1=y_vec
      svm_parameters = train_svm(x1,y1,C)
      alphas = np.array(svm_parameters['x'])
      w_opt[0]=alphas[0]
      w_opt[1]=alphas[1]
      b_opt=alphas[102]
      epsilon=acc(x_vec,y_vec,b_opt,w_opt,len(x_vec))
      margin_t = 1 / np.sqrt(np.sum(np.power(w_opt,2)))
      margin[it]=margin_t
      accuracy[it]=epsilon
      w_vec[it][0]=w_opt[0]
      w_vec[it][1]=w_opt[1]
      b_vec[it]=b_opt
      c_vec[it]=C
  return margin,accuracy,w_vec,b_vec,c_vec
m1,a1,w1,b1,c1=call_fun(x,y)
max_a=0.0
max_m=np.zeros((2,1))
max_w=0.0
max_b=0.0
max_c=0.0
for i in range(len(a)):
  if a1[i]>max_a:
    max_a,max_m,max_w,max_b,max_c=a1[i],m1[i],w1[i],b1[i],c1[i]
  elif a1[i]==max_a and m1[i]>max_m:
    max_a,max_m,max_w,max_b,max_c=a1[i],m1[i],w1[i],b1[i],c1[i]
print('optimal accuracy is:', max_a)
print('optimal margin is:', max_m)
print('optimal weight is:', max_w)
print('optimal b is:', max_b)
print('optimal c is:', max_c)

x_test_1=np.zeros((50,2))
for i in range(50):
  x_test_1[i][0]=np.random.uniform(-3,3)
  x_test_1[i][1]=np.random.uniform(-3,3)
y_test_1=np.zeros((50,1))
for i in range(50):
  y_test_1[i]=predict(x_test_1[i],b,w)
print(acc(x_test_1,y_test_1,max_b,max_w,50))

def new_predict(x):
  if(np.power(x[0],2) + np.power(x[1],2)/2) <= 2:
    return 1.0
  else:
    return -1.0
x_2_train = np.random.uniform(-3,3,(100,2))
for i in range(100):
  x_2_train[i][0]=np.random.uniform(-3,3)
  x_2_train[i][1]=np.random.uniform(-3,3)
y_2_train = np.zeros(shape=(100,1))
for i in range(100):
  y_2_train[i] = new_predict(x_2_train[i])
#plt.scatter(x_2_train[:,[0]], x_2_train[:,1], c=['red' if y_i == -1.0 else 'blue' for y_i in y_2_train], facecolors='none')
for i in range(100):
  if y_2_train[i]==-1.0:
    plt.scatter(x_2_train[i][0],x_2_train[i][1],color='red',s=10)
  else:
        plt.scatter(x_2_train[i][0],x_2_train[i][1],color='blue',s=10)
plt.xlabel('x_2_train[0]')
plt.ylabel('x_2_train[1]')

m2c,a2c,w2c,b2c,c2c=call_fun(x_2_train,y_2_train)
max_a2c=0.0
max_m2c=np.zeros((2,1))
max_w2c=0.0
max_b2c=0.0
max_c2c=0.0
for iq in range(len(a2c)):
  if a2c[iq]>max_a2c:
    max_a2c,max_m2c,max_w2c,max_b2c,max_c2c=a2c[iq],m2c[iq],w2c[iq],b2c[iq],c2c[iq]
  elif a2c[i]==max_a2c and m2c[i]>max_m2c:
    max_a2c,max_m2c,max_w2c,max_b2c,max_c2c=a2c[iq],m2c[iq],w2c[iq],b2c[iq],c2c[iq]
print('optimal accuracy is:', max_a2c)
print('optimal margin is:', max_m2c)
print('optimal weight is:', max_w2c)
print('optimal b is:', max_b2c)
print('optimal c is:', max_c2c)

x_test_2c=np.zeros((50,2))
for i in range(50):
  x_test_2c[i][0]=np.random.uniform(-3,3)
  x_test_2c[i][1]=np.random.uniform(-3,3)
y_test_2c=np.zeros((50,1))
for i in range(50):
  y_test_2c[i]=new_predict(x_test_2c[i])
print(acc(x_test_2c,y_test_2c,max_b2c,max_w2c,50))

x2d=x_2_train
y2d=y_2_train
for i in range(100):
  x2d[i][0] = x2d[i][0]*x2d[i][0]
  x2d[i][1] = x2d[i][1]*x2d[i][1]

plt.scatter(x2d[:,0], x2d[:,1], c=['red' if y_i == -1 else 'blue' for y_i in y2d], facecolors='none',s=10)
plt.xlabel('x2d[0]')
plt.ylabel('x2d[1]')

m2d,a2d,w2d,b2d,c2d=call_fun(x2d,y2d)
max_a2d=0.0
max_m2d=np.zeros((2,1))
max_w2d=0.0
max_b2d=0.0
max_c2d=0.0
for iq in range(len(a2d)):
  if a2d[iq]>max_a2d:
    max_a2d,max_m2d,max_w2d,max_b2d,max_c2d=a2d[iq],m2d[iq],w2d[iq],b2d[iq],c2d[iq]
  elif a2d[i]==max_a2d and m2d[i]>max_m2d:
    max_a2d,max_m2d,max_w2d,max_b2d,max_c2d=a2d[iq],m2d[iq],w2d[iq],b2d[iq],c2d[iq]
print('optimal accuracy is:', max_a2d)
print('optimal margin is:', max_m2d)
print('optimal weight is:', max_w2d)
print('optimal b is:', max_b2d)
print('optimal c is:', max_c2d)

x_test_2d=np.zeros((50,2))
for i in range(50):
  a=np.random.uniform(-3,3)
  x_test_2d[i][0]=a*a
  b=np.random.uniform(-3,3)
  x_test_2d[i][1]=b*b
y_test_2d=np.zeros((50,1))
for i in range(50):
  y_test_2d[i]=new_predict(x_test_2d[i])
print(acc(x_test_2d,y_test_2d,max_b2d,max_w2d,50))

parameters = {}
KERNEL_RBF=2
KERNEL_LINEAR=1
def gram_matrix(X, Y, kernel_type, gamma=0.5):
    K = np.zeros((X.shape[0], Y.shape[0]))   
    if kernel_type == KERNEL_LINEAR:
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = np.dot(x.T, y)               
    elif kernel_type == KERNEL_RBF:
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = np.exp(-gamma * np.linalg.norm(x - y) ** 2)      
    return K

def get_parameters(alphas,xl,yl):
    threshold = 1e-5 # Values greater than zero (some floating point tolerance)
    S = (alphas > threshold).reshape(-1, )
    w = np.dot(xl.T, alphas * yl)
    b = yl[S] - np.dot(xl[S], w) # b calculation
    b = np.mean(b)
    return w, b, S

def train_svm2(C,xt,yt):
    n, k = xt.shape
    y_matrix = yt.reshape(1, -1)
    H = np.dot(y_matrix.T, y_matrix) * gram_matrix(xt, xt, KERNEL_RBF )
    P = matrix(H)
    q = matrix(-np.ones((n, 1)))
    G = matrix(np.vstack((-np.eye((n)), np.eye(n))))
    h = matrix(np.vstack((np.zeros((n,1)), np.ones((n,1)) * C)))
    A = matrix(y_matrix)
    A = matrix(A, (1, n), 'd')
    b = matrix(np.zeros(1))   
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10
    return solvers.qp(P, q, G, h, A, b)

def call_fun2(x_vec,y_vec):
  margin=np.zeros((500,1))
  accuracy=np.zeros((500,1))
  w_vec=np.zeros((500,2))
  b_vec=np.zeros((500,1))
  w_opt=np.zeros((2,1))
  c_vec=np.zeros((500,1))
  b_opt=0
  C=0.000006
  iteration=200
  for it in range(iteration):
      print('iteration number is :', it)
      print('c is', C)
      x1=x_vec
      y1=y_vec
      svm_parameters = train_svm2(C,x1,y1)
      alphas = np.array(svm_parameters['x'])
      w, b, S = get_parameters(alphas,x1,y1)
      w_opt[0]=w[0]
      w_opt[1]=w[1]
      b_opt=b
      epsilon=acc(x_vec,y_vec,b_opt,w_opt,len(x_vec))
      margin_t = 1 / np.sqrt(np.sum(np.power(w_opt,2)))
      margin[it]=margin_t
      accuracy[it]=epsilon
      w_vec[it][0]=w_opt[0]
      w_vec[it][1]=w_opt[1]
      b_vec[it]=b_opt
      c_vec[it]=C
      C+=0.00001
  return margin,accuracy,w_vec,b_vec,c_vec

m2e,a2e,w2e,b2e,c2e=call_fun2(x_2_train,y_2_train)
max_a2e=0.0
max_m2e=np.zeros((2,1))
max_w2e=0.0
max_b2e=0.0
max_c2e=0.0
for i in range(len(a2e)):
  if a2e[i]>max_a2e:
    max_a2e,max_m2e,max_w2e,max_b2e,max_c2e=a2e[i],m2e[i],w2e[i],b2e[i],c2e[i]
  elif a2e[i]==max_a2e and m2e[i]>max_m2e:
    max_a2e,max_m2e,max_w2e,max_b2e,max_c2e=a2e[i],m2e[i],w2e[i],b2e[i],c2e[i]
print('optimal accuracy is:', max_a2e)
print('optimal margin is:', max_m2e)
print('optimal weight is:', max_w2e)
print('optimal b is:', max_b2e)
print('optimal c is:', max_c2e)

x_test_2e=np.zeros((50,2))
for i in range(50):
  a=np.random.uniform(-3,3)
  x_test_2e[i][0]=a*a
  b=np.random.uniform(-3,3)
  x_test_2e[i][1]=b*b
y_test_2e=np.zeros((50,1))
for i in range(50):
  y_test_2e[i]=new_predict(x_test_2e[i])
print(acc(x_test_2e,y_test_2e,max_b2e,max_w2e,50))