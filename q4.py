import numpy as np
import pandas as pd
import dill
import random
import matplotlib.pyplot as plt
with open('kernel_4a.pkl', 'rb') as in_strm:
	a = dill.load(in_strm)

bs=b'\x80\x03cdill._dill\n_create_function\nq\x00(cdill._dill\n_load_type\nq\x01X\x08\x00\x00\x00CodeTypeq\x02\x85q\x03Rq\x04(K\x02K\x00K\x02K\x03KCC |\x00d\x01\x13\x00\xa0\x00\xa1\x00d\x01\x18\x00|\x01d\x01\x13\x00\xa0\x00\xa1\x00d\x01\x18\x00\x14\x00S\x00q\x05Xi\x00\x00\x00\n    :param x: (d, 1) feature vector\n    :param y: (d, 1) feature vector\n\n    :return value: k(x, y)\n    q\x06K\x02\x86q\x07X\x03\x00\x00\x00sumq\x08\x85q\tX\x01\x00\x00\x00xq\nX\x01\x00\x00\x00yq\x0b\x86q\x0cX\x10\x00\x00\x00kernel_source.pyq\rX\t\x00\x00\x00kernel_4aq\x0eK\x06C\x02\x00\x07q\x0f))tq\x10Rq\x11c__builtin__\n__main__\nh\x0eNN}q\x12Ntq\x13Rq\x14.'
import pickle
load = dill.loads(bs)

def dist(x,y,k):
	return load(x,x) -2*load(x,y) +load(y,y)
d=10
D = np.zeros(shape=(d, d))
e = np.zeros(shape=(d,d))
for i in range(d):
	e[i][i]=1

'''for i in range(d):
  print(e[:i])'''
for i in range(d):
	for j in range(d):
    #print(e[:i])
		D[i][j]=dist(e[:,[i]],e[:,[i]],load)
dsum=0
for i in range(10):
	for j in range(10):
		dsum+=D[i,j]
print(dsum)
ans=np.zeros(d)
sum3=0
for i in range(d):
	for j in range(d):
		sum3+=load(e[:,[i]],e[:,[j]])
#print(sum3)
sum2=np.zeros(d)
for i in range(d):
	for j in range(d):
		sum2[i]+=load(e[:,[i]],e[:,[j]])
#print(sum2)
sum1=np.zeros(d)
for i in range(d):
	sum1[i]=load(e[:,[i]],e[:,[i]])
#print(sum1)
sum_final=0
for i in range(d):
	a=sum1[i]-2*sum2[i]/d + sum3/np.power(d,2)
	sum_final+=a  
print(sum_final)


data = np.load('data.npy')
X=data




def k_distance(x1, x2):
    return np.sqrt(dist(x1,x2,load))


class KernelKMeans():

    def __init__(self, K=2, iterations=100):
        self.K = K
        self.iterations = iterations
        self.centers = []
        self.clusters = [[] for _ in range(self.K)]
        

    def train_cluster_fun(self, X):
        self.X = X
        self.n, self.f = X.shape        
        rsi = np.random.choice(self.n, self.K)
        self.centers = [self.X[idx] for idx in rsi]

        for _ in range(self.iterations):
            self.clusters = self.create_cluster_fun(self.centers)
            centers_old = self.centers
            self.centers = self.find_cluster_fun(self.clusters)
            if self.converge_fun(centers_old, self.centers):
                break
        return self.label_cluster_fun(self.clusters)

    def create_cluster_fun(self, ct):
        cls = [[] for _ in range(self.K)]
        for idx, s in enumerate(self.X):
            c_idx = self.nearest_center_fun(s, ct)
            cls[c_idx].append(idx)
        return cls

    def label_cluster_fun(self, clusters):
        l = np.zeros((self.n,1))
        for i, cl in enumerate(clusters):
            for k in cl:
                l[k] = i
        return l
    def find_cluster_fun(self, cs):
        ct = np.zeros((self.K, self.f))
        for c_idx, c in enumerate(cs):
            sum1=0
            if X[c].shape[0]>0:
                 for i in range(len(X[c])):
                      sum1+=np.sqrt(np.sqrt((load(X[c][i],X[c][i]))))
            c_mean=sum1/100
            ct[c_idx] = c_mean
        return ct
    def nearest_center_fun(self, s, ct):
        d = [k_distance(s, p) for p in ct]
        c_i = np.argmin(d)
        return c_i
    def converge_fun(self, c_old, c_new):
        d = [k_distance(c_old[i], c_new[i]) for i in range(self.K)]
        return sum(d) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(20, 10))
        for i, j in enumerate(self.clusters):
            p = self.X[j].T
            ax.scatter(*p)
        plt.title('scatter plot after applying kernelised k-mean clustering')
        plt.show()

obj=KernelKMeans()
obj.train_cluster_fun(X)
obj.plot()

