from network import Network
import csv
import numpy as np

# readData

def encodeLabel(label):
	label = float(label)
	lab = np.zeros((3,1))
	if label==1.:
		lab[0]=1.0
	elif label == 0.:
		lab[1] = 1.0
	elif label == -1.0:
		lab[2] = 1.0
	else:
		print("No Match, error")
	return lab

def encodeData(x):
	floatv = np.vectorize(float)
	return floatv(x).reshape(22,1)

filename = 'stock/jnjdata4.csv'


data = [row for row in csv.reader(open(filename))]
header = data.pop(0)
data = [(encodeData(row[1:]),encodeLabel(row[0])) for row in data]

inputDim = len(header)-1

shape = [22, 30, 3]
nn = Network(shape)

testStartIndex = len(data)-1000
trainSet = data[:10000]
testSet = data[testStartIndex:]

nn.TMB(trainSet, 10, 10, 5, testSet)
