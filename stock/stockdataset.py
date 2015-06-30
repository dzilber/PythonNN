import numpy as np
import csv


def convertSign(x):
	return [np.sign(i-j) for i,j in zip(x[1:], x[:-1])]

def sigm(x):

	ratio = [-1*i/j for i,j in zip(x[1:], x[:-1])]
	return 1/(1+np.exp(ratio))

# Scale the 
def maxDiv(x):
	mx = np.max(x)
	mn = np.min(x)
	return list((x-mn)/(mx-mn))



red = csv.reader(open('JNJ.csv'))

hed = next(red)
data = [float(i[6]) for i in red]

print(data[1])
span = 22


data = [data[i:i+span] for i in range(len(data)-span)]
data2 = [convertSign(v) for v in data]
data3 = [sigm(v) for v in data]
data4 = [maxDiv(v) for v in data]


print(data3[4])
label = [np.sign(i[-1]-j[-1]) for i,j in zip(data[1:],data[:-1])]

labeled = [[i]+list(j) for i,j in zip(label,data4)]

print(labeled[2])

writ = csv.writer(open('jnjdata4.csv','w'))

header = ['Label']+['Day '+str(i) for i in range(span)]
print(header)
writ.writerow(header)

for row in labeled:
	writ.writerow(row)
