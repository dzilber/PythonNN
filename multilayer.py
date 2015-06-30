import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt


'''


'''

dim1r = 100
dim1c = 28**2

dim2r = 10
dim2c = 100

W1 = np.matrix(np.random.rand(dim1r * dim1c)*2-1).reshape(dim1r, dim1c)
W2 = np.matrix(np.random.rand(dim2r * dim2c)*2-1).reshape(dim2r, dim2c)
b1 = np.matrix(np.random.rand(dim1r)*2-1).reshape(dim1r, 1)
b2 = np.matrix(np.random.rand(dim2r)*2 -1 ).reshape(dim2r, 1)
a1 = 8
a2 = 4

# class multilayer():
# 	def __init__(self, shape):
# 		self.weights = [np.random.randn(row, col) for row, col in zip(shape[1:],shape(:-1))
# 		self.bias = 

def dsig2(x):
	return np.multiply(sig(x),(1-sig(x)))

def sig2(x):
	return 1./(1.+np.exp(-x))

sig = np.vectorize(sig2)
dsig = np.vectorize(dsig2)

def train(dataList, labelList):
	global W1, W2, b1, b2, a1, a2

	dw2 = np.zeros(W2.shape)
	dw1 = np.zeros(W1.shape)
	db2 = np.zeros(b2.shape)
	db1 = np.zeros(b1.shape)
	errorSet = []
	n = len(dataList)

	for data,label in zip(dataList, labelList):
		z1 = W1 * data
		y1 = sig(z1 + b1)
		z2 = W2 * y1
		y2 = sig(z2 + b2)

		# print(y2 - label)
		predict = np.argmax(y2)
		actual = np.argmax(label)

		error = 1/2 * (y2 - label).T * (y2 - label)
		# print(y2-label)
		errorSet.append(error)
		derror_dy2 =  y2 - label   #[10 x 1]
		dy2_dz2 = dsig(z2 + b2) # [10 x 1]
		dz2_dw2 = y1.T # [1000 x 1]
		dz2_dy1 = W2.T # [1000x10] 
		dy1_dz1 = dsig(z1 + b1) # [1000 x 1]
		dz1_dw1 = data.T # [784 x 1]

		derror_dz2 = np.multiply(dy2_dz2, derror_dy2) # [10 x 1]
		# print(derror_dz2.T)
		derror_dy1 = dz2_dy1 * derror_dz2 # [1000 x 10] * [10 x 1] = [1000 x 1]
		# print(derror_dy1.T)
		derror_dz1 = np.multiply(dy1_dz1, derror_dy1)
		derror_dw2 = derror_dz2 * dz2_dw2 # [10x1] * [1*1000] = [10x1000] = W2
		derror_dw1 = derror_dz1 * dz1_dw1  # [1000x1]*[1x784] = [1000x784] = W1

		dw2 += derror_dw2
		dw1 += derror_dw1
		db2 += derror_dz2
		db1 += derror_dz1

	W2 += -a2/n*dw2
	b2 += -a2/n*db2

	W1 += -a1/n*dw1
	b1 += -a1/n*db1



	return np.average(errorSet)


def feedForward(data):
	global W1, W2, b1, b2, a1, a2
	z1 = W1 * data
	y1 = sig(z1 + b1)
	z2 = W2 * y1
	y2 = sig(z2 + b2)
	return np.argmax(y2)

def encodeLabel(lab):
	enc = np.zeros(10)
	enc[int(lab)]=1
	return enc

def encodeData(dat):
	intv = np.vectorize(int)
	data = intv(dat).reshape(784,1)/255
	return data

def test(testSet):
	
	count = 0

	for row in testSet:
		label = int(row[0])
		data = encodeData(row[1:])
		predict = feedForward(data)
		# print("Predict: ",predict, "Actual:",label)
		if label==predict:
			count+=1
	return count/len(testSet)









inputData = csv.reader(open('mnist/train.csv'))
epochs = 8
batch = 10
trainingSamples = 500
testSize = 1000



header = next(inputData)
testSet=[next(inputData) for i in range(testSize)]
errorList = []
wList = []


for e in range(epochs):

	for j in range(trainingSamples):
		dataList = []
		labelList = []
		for i in range(batch):

			# read pixel data
			row =[int(i) for i in next(inputData)]

			trueValue = row[0]
			data = np.array(row[1:]).reshape(784,1)/255


			#convert true value to vector
			label = [0]*10
			label[trueValue] = 1
			label = np.array(label).reshape(10,1)
			labelList.append(label)
			dataList.append(data)
			
			# errorList.append(error[0,0])
		error = train(dataList, labelList)
		errorList.append(error)
		# print("TrainingSet",j,"   ERROR:",error)
		wList.append((W2[5,1],W2[0,6]))
	pcorrect = test(testSet)
	
	print("Epoch:",e,"  PERCENT CORRECT:",pcorrect)

	




plt.plot(errorList)
plt.show()

x,y = zip(*wList)
plt.subplot(3,1,1)
plt.plot(x)
plt.subplot(3,1,2)
plt.plot(y)
plt.subplot(3,1,3)
plt.plot(x,y)
plt.show()






