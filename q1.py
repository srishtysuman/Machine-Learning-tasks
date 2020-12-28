import numpy as np
import pandas as pd
import dill
import random
import matplotlib.pyplot as plt
import pickle

def is_kernel(K):
	eigenvalues = np.linalg.eigvals(K)
	print(np.min(eigenvalues))
	symmetric=True
	psd=True
	for i in range(100):
		j=0
		for j in range(100):
			if K[i][j]!=K[j][i]:
				symmetric=False
				print('this kernel is not symmetric')
				break
		if j!=100:
			break
	i=0	
	for i in range(len(eigenvalues)):
		if eigenvalues[i]<0:
			psd=False
			print('this kernel is not psd')
			break
	if psd==True and symmetric==True:
		print('this kernel is symmetric and psd')
		return True
	elif symmetric==False and psd==True:
		print('this kernel is not symmetric but psd')
		return False
	elif psd==False and symmetric==True:
		print('this kernel is not psd but symmetric')
		return False
	else:
		print('this kernel is neither symmetric nor psd')
		return False

with open('function5.pkl', 'rb') as in_strm:
	sa = dill.load(in_strm)
with open('k5sampler.pkl', 'rb') as in_strm:
	sb = dill.load(in_strm)
bsa=b'\x80\x03cdill._dill\n_create_function\nq\x00(cdill._dill\n_load_type\nq\x01X\x08\x00\x00\x00CodeTypeq\x02\x85q\x03Rq\x04(K\x02K\x00K\x02K\x06KCC\x90|\x00j\x00d\x01\x19\x00|\x01j\x00d\x01\x19\x00k\x02rL|\x00j\x00d\x02\x19\x00d\x02k\x02rL|\x01j\x00d\x02\x19\x00d\x02k\x02rLt\x01|\x00j\x00\x83\x01d\x03k\x02rLt\x01|\x01j\x00\x83\x01d\x03k\x02sTt\x02d\x04\x83\x01\x82\x01t\x03\xa0\x04t\x03\xa0\x05|\x00|\x01\x18\x00\xa1\x01\xa1\x01d\x05k\x01r~t\x03\xa0\x06|\x00j\x07|\x01\xa1\x02d\x06\x18\x00S\x00t\x03\xa0\x06|\x00j\x07|\x01\xa1\x02S\x00d\x07S\x00q\x05(X\x87\x00\x00\x00\n    :param x: A vector of size (n, 1)\n    :param y: A vector of size (n, 1)\n\n    :return value: kernel evaluated at points (x, y)\n    q\x06K\x00K\x01K\x02X,\x00\x00\x00Vectors x and y must have shape (n, 1) each.q\x07G?\xb9\x99\x99\x99\x99\x99\x9aG?\xe0\x00\x00\x00\x00\x00\x00Ntq\x08(X\x05\x00\x00\x00shapeq\tX\x03\x00\x00\x00lenq\nX\x0e\x00\x00\x00AssertionErrorq\x0bX\x02\x00\x00\x00npq\x0cX\x03\x00\x00\x00sumq\rX\x03\x00\x00\x00absq\x0eX\x06\x00\x00\x00matmulq\x0fX\x01\x00\x00\x00Tq\x10tq\x11X\x01\x00\x00\x00xq\x12X\x01\x00\x00\x00yq\x13\x86q\x14X\x13\x00\x00\x00functions_source.pyq\x15X\t\x00\x00\x00function5q\x16KMC\x12\x00\x08\x14\x01\x0e\x01\x0e\x01\x0e\x01\x10\x01\x06\x02\x18\x01\x12\x02q\x17))tq\x18Rq\x19c__builtin__\n__main__\nh\x16NN}q\x1aNtq\x1bRq\x1c.'

bsb=b'\x80\x03cdill._dill\n_create_function\nq\x00(cdill._dill\n_load_type\nq\x01X\x08\x00\x00\x00CodeTypeq\x02\x85q\x03Rq\x04(K\x00K\x00K\x02K\x06KCCLd\x01}\x00x0|\x00s4t\x00j\x01j\x02d\x02d\x03\x8d\x01}\x01t\x00j\x03\xa0\x04|\x01\xa0\x05d\x04\xa1\x01\xa1\x01d\x05k\x01r\x06d\x06}\x00q\x06W\x00|\x01t\x00j\x03\xa0\x04|\x01\xa0\x05d\x04\xa1\x01\xa1\x01\x1b\x00S\x00q\x05(Xs\x00\x00\x00\n    :return sample: (3, 1) A sample from positive orthant on the surface\n                    of a unit sphere\n    q\x06\x89K\x03K\x01\x86q\x07X\x04\x00\x00\x00sizeq\x08\x85q\tJ\xff\xff\xff\xff\x85q\nK\x01\x88tq\x0b(X\x02\x00\x00\x00npq\x0cX\x06\x00\x00\x00randomq\rX\x07\x00\x00\x00uniformq\x0eX\x06\x00\x00\x00linalgq\x0fX\x04\x00\x00\x00normq\x10X\x07\x00\x00\x00reshapeq\x11tq\x12X\x04\x00\x00\x00doneq\x13X\x06\x00\x00\x00sampleq\x14\x86q\x15X\x13\x00\x00\x00functions_source.pyq\x16X\t\x00\x00\x00k5samplerq\x17KbC\x0c\x00\x06\x04\x01\x06\x01\x0e\x01\x16\x01\x08\x01q\x18))tq\x19Rq\x1ac__builtin__\n__main__\nh\x17NN}q\x1bNtq\x1cRq\x1d.'

load_sampler = dill.loads(bsb)
load_function = dill.loads(bsa)
x5=np.zeros(shape=(100,3,1))
for i in range(100):
	x5[i]=load_sampler()
K5=np.zeros((100,100))
for i in range(100):
	for j in range(100):
		K5[i,j]=load_function(x5[i],x5[j])
print(K5)
ret5=is_kernel(K5)
if ret5==True:
	print('K5 is a kernel')
else:
	print('K5 is not a kernel')
print("--------------------------------------------------------------------------------------------------")
with open('function3.pkl', 'rb') as in_strm:
	a4 = dill.load(in_strm)

bs4=b'\x80\x03cdill._dill\n_create_function\nq\x00(cdill._dill\n_load_type\nq\x01X\x08\x00\x00\x00CodeTypeq\x02\x85q\x03Rq\x04(K\x02K\x00K\x02K\x06KCCv|\x00j\x00d\x01\x19\x00|\x01j\x00d\x01\x19\x00k\x02rL|\x00j\x00d\x02\x19\x00d\x02k\x02rL|\x01j\x00d\x02\x19\x00d\x02k\x02rLt\x01|\x00j\x00\x83\x01d\x03k\x02rLt\x01|\x01j\x00\x83\x01d\x03k\x02sTt\x02d\x04\x83\x01\x82\x01t\x03t\x04j\x05\xa0\x06|\x00|\x01\x18\x00\xa0\x07d\x05\xa1\x01\xa1\x01d\x03\x13\x00d\x06k\x01\x83\x01S\x00q\x05(X\x87\x00\x00\x00\n    :param x: A vector of size (n, 1)\n    :param y: A vector of size (n, 1)\n\n    :return value: kernel evaluated at points (x, y)\n    q\x06K\x00K\x01K\x02X,\x00\x00\x00Vectors x and y must have shape (n, 1) each.q\x07J\xff\xff\xff\xff\x85q\x08K\x06tq\t(X\x05\x00\x00\x00shapeq\nX\x03\x00\x00\x00lenq\x0bX\x0e\x00\x00\x00AssertionErrorq\x0cX\x05\x00\x00\x00floatq\rX\x02\x00\x00\x00npq\x0eX\x06\x00\x00\x00linalgq\x0fX\x04\x00\x00\x00normq\x10X\x07\x00\x00\x00reshapeq\x11tq\x12X\x01\x00\x00\x00xq\x13X\x01\x00\x00\x00yq\x14\x86q\x15X\x13\x00\x00\x00functions_source.pyq\x16X\t\x00\x00\x00function4q\x17K;C\x0e\x00\x08\x14\x01\x0e\x01\x0e\x01\x0e\x01\x10\x01\x06\x02q\x18))tq\x19Rq\x1ac__builtin__\n__main__\nh\x17NN}q\x1bNtq\x1cRq\x1d.'

load4 = dill.loads(bs4)
x4=np.zeros(shape=(100,3))
for i in range(100):
	x4[i][0]=np.random.uniform(-5,5)
	x4[i][1]=np.random.uniform(-5,5)
	x4[i][2]=np.random.uniform(-5,5)
K4=np.zeros(shape=(100,100))
for i1 in range(100):
	for j1 in range(100):
		x41=x4[i1].reshape(3,1)
		x42=x4[j1].reshape(3,1)
		K4[i1,j1]=load4(x41,x42)
print("--------------------------------------------------------------------------------------------------")
print(K4)
ret=is_kernel(K4)
if ret==True:
	print('K4 a kernel')
else:
	print('K4 not a kernel')
print("--------------------------------------------------------------------------------------------------")

with open('function3.pkl', 'rb') as in_strm:
	a3 = dill.load(in_strm)

bs3=b'\x80\x03cdill._dill\n_create_function\nq\x00(cdill._dill\n_load_type\nq\x01X\x08\x00\x00\x00CodeTypeq\x02\x85q\x03Rq\x04(K\x02K\x00K\x02K\x05KCCl|\x00j\x00d\x01\x19\x00|\x01j\x00d\x01\x19\x00k\x02rL|\x00j\x00d\x02\x19\x00d\x02k\x02rL|\x01j\x00d\x02\x19\x00d\x02k\x02rLt\x01|\x00j\x00\x83\x01d\x03k\x02rLt\x01|\x01j\x00\x83\x01d\x03k\x02sTt\x02d\x04\x83\x01\x82\x01t\x03\xa0\x04|\x00j\x05d\x05\x13\x00|\x01d\x05\x13\x00\x18\x00\xa1\x01S\x00q\x05(X\x87\x00\x00\x00\n    :param x: A vector of size (n, 1)\n    :param y: A vector of size (n, 1)\n\n    :return value: kernel evaluated at points (x, y)\n    q\x06K\x00K\x01K\x02X,\x00\x00\x00Vectors x and y must have shape (n, 1) each.q\x07K\x06tq\x08(X\x05\x00\x00\x00shapeq\tX\x03\x00\x00\x00lenq\nX\x0e\x00\x00\x00AssertionErrorq\x0bX\x02\x00\x00\x00npq\x0cX\x03\x00\x00\x00sumq\rX\x01\x00\x00\x00Tq\x0etq\x0fX\x01\x00\x00\x00xq\x10X\x01\x00\x00\x00yq\x11\x86q\x12X\x13\x00\x00\x00functions_source.pyq\x13X\t\x00\x00\x00function3q\x14K)C\x0e\x00\x08\x14\x01\x0e\x01\x0e\x01\x0e\x01\x10\x01\x06\x02q\x15))tq\x16Rq\x17c__builtin__\n__main__\nh\x14NN}q\x18Ntq\x19Rq\x1a.'

load3 = dill.loads(bs3)
x3=np.zeros(shape=(100,3))
for i in range(100):
	x3[i][0]=np.random.uniform(-5,5)
	x3[i][1]=np.random.uniform(-5,5)
	x3[i][2]=np.random.uniform(-5,5)
K3=np.zeros(shape=(100,100))
for i1 in range(100):
	for j1 in range(100):
		x31=x3[i1].reshape(3,1)
		x32=x3[j1].reshape(3,1)
		K3[i1,j1]=load3(x31,x32)
print("--------------------------------------------------------------------------------------------------")
print(K3)
ret3=is_kernel(K3)
if ret3==True:
	print('K3 a kernel')
else:
	print('K3 not a kernel')
print("--------------------------------------------------------------------------------------------------")


with open('function2.pkl', 'rb') as in_strm:
	a2 = dill.load(in_strm)

bs2=b'\x80\x03cdill._dill\n_create_function\nq\x00(cdill._dill\n_load_type\nq\x01X\x08\x00\x00\x00CodeTypeq\x02\x85q\x03Rq\x04(K\x02K\x00K\x02K\x05KCCj|\x00j\x00d\x01\x19\x00|\x01j\x00d\x01\x19\x00k\x02rL|\x00j\x00d\x02\x19\x00d\x02k\x02rL|\x01j\x00d\x02\x19\x00d\x02k\x02rLt\x01|\x00j\x00\x83\x01d\x03k\x02rLt\x01|\x01j\x00\x83\x01d\x03k\x02sTt\x02d\x04\x83\x01\x82\x01t\x03\xa0\x04|\x00j\x05d\x05\x13\x00|\x01d\x05\x13\x00\xa1\x02S\x00q\x05(X\x87\x00\x00\x00\n    :param x: A vector of size (n, 1)\n    :param y: A vector of size (n, 1)\n\n    :return value: kernel evaluated at points (x, y)\n    q\x06K\x00K\x01K\x02X,\x00\x00\x00Vectors x and y must have shape (n, 1) each.q\x07K\x06tq\x08(X\x05\x00\x00\x00shapeq\tX\x03\x00\x00\x00lenq\nX\x0e\x00\x00\x00AssertionErrorq\x0bX\x02\x00\x00\x00npq\x0cX\x06\x00\x00\x00matmulq\rX\x01\x00\x00\x00Tq\x0etq\x0fX\x01\x00\x00\x00xq\x10X\x01\x00\x00\x00yq\x11\x86q\x12X\x13\x00\x00\x00functions_source.pyq\x13X\t\x00\x00\x00function2q\x14K\x17C\x0e\x00\x08\x14\x01\x0e\x01\x0e\x01\x0e\x01\x10\x01\x06\x02q\x15))tq\x16Rq\x17c__builtin__\n__main__\nh\x14NN}q\x18Ntq\x19Rq\x1a.'

load2 = dill.loads(bs2)
x2=np.zeros(shape=(100,3))
for i in range(100):
	x2[i][0]=np.random.uniform(-5,5)
	x2[i][1]=np.random.uniform(-5,5)
	x2[i][2]=np.random.uniform(-5,5)
K2=np.zeros(shape=(100,100))
for i1 in range(100):
	for j1 in range(100):
		x21=x2[i1].reshape(3,1)
		x22=x2[j1].reshape(3,1)
		K2[i1,j1]=load2(x21,x22)
print("--------------------------------------------------------------------------------------------------")
print(K2)
ret2=is_kernel(K2)
if ret2==True:
	print('K2 a kernel')
else:
	print('K2 not a kernel')
print("--------------------------------------------------------------------------------------------------")
with open('function1.pkl', 'rb') as in_strm:
	a = dill.load(in_strm)

bs=b'\x80\x03cdill._dill\n_create_function\nq\x00(cdill._dill\n_load_type\nq\x01X\x08\x00\x00\x00CodeTypeq\x02\x85q\x03Rq\x04(K\x02K\x00K\x02K\x04KCCd|\x00j\x00d\x01\x19\x00|\x01j\x00d\x01\x19\x00k\x02rL|\x00j\x00d\x02\x19\x00d\x02k\x02rL|\x01j\x00d\x02\x19\x00d\x02k\x02rLt\x01|\x00j\x00\x83\x01d\x03k\x02rLt\x01|\x01j\x00\x83\x01d\x03k\x02sTt\x02d\x04\x83\x01\x82\x01t\x03\xa0\x04|\x00j\x05|\x01\xa1\x02\x0b\x00S\x00q\x05(X\x87\x00\x00\x00\n    :param x: A vector of size (n, 1)\n    :param y: A vector of size (n, 1)\n\n    :return value: kernel evaluated at points (x, y)\n    q\x06K\x00K\x01K\x02X,\x00\x00\x00Vectors x and y must have shape (n, 1) each.q\x07tq\x08(X\x05\x00\x00\x00shapeq\tX\x03\x00\x00\x00lenq\nX\x0e\x00\x00\x00AssertionErrorq\x0bX\x02\x00\x00\x00npq\x0cX\x06\x00\x00\x00matmulq\rX\x01\x00\x00\x00Tq\x0etq\x0fX\x01\x00\x00\x00xq\x10X\x01\x00\x00\x00yq\x11\x86q\x12X\x13\x00\x00\x00functions_source.pyq\x13X\t\x00\x00\x00function1q\x14K\x05C\x0e\x00\x08\x14\x01\x0e\x01\x0e\x01\x0e\x01\x10\x01\x06\x02q\x15))tq\x16Rq\x17c__builtin__\n__main__\nh\x14NN}q\x18Ntq\x19Rq\x1a.'

load = dill.loads(bs)
x1=np.zeros(shape=(100,3))
for i in range(100):
	x1[i][0]=np.random.uniform(-5,5)
	x1[i][1]=np.random.uniform(-5,5)
	x1[i][2]=np.random.uniform(-5,5)
K1=np.zeros(shape=(100,100))
for i1 in range(100):
	for j1 in range(100):
		x11=x1[i1].reshape(3,1)
		x12=x1[j1].reshape(3,1)
		K1[i1,j1]=load(x11,x12)

print("--------------------------------------------------------------------------------------------------")
print(K1)
ret1=is_kernel(K1)
if ret1==True:
	print('K1 a kernel')
else:
	print('K1 not a kernel')
print("--------------------------------------------------------------------------------------------------")



