import numpy as np
import matplotlib.pyplot as plt
from pySOT import *

def read_measured_data(filename):
    file = filename
    with open(file) as f:
        reader = f.readlines()
        data = []
        for i, line in enumerate(reader):
            if i >= 1:
                str1 = line.split('\t')
                str2 = str1[3].replace('@[', '').replace('\n', '').replace(']', '').split(', ')
                str3 = str1[0:3] + str2
                if 'None' in str3:
                    str3 = str1[0:2] + ['0'] + str2
                    str3 = [float(item) for item in str3]
                    data.append(str3)
                else:
                    str3 = [float(item) for item in str3]
                    data.append(str3)
        data = np.asarray(data)
        return data

# Step 1 - Read the monitored calibration progress file
filename = "pysot_result_9_24.txt"
data = read_measured_data(filename)

# Step 2 - Norimize the parameter value

xlow = np.array([0.1, 0.1, 0.1, 0, 0, 0, 0.001, 0.001, 0.02])  
xup = np.array([2.0, 1.0, 1.0, 0.005, 0.005, 0.05, 0.002, 0.002, 0.03])

nsamples = len(data)
dim = 9
for i in range(dim):
    for j in range(nsamples):
       data[j][i + 3] = (data[j][ i + 3] - xlow[i]) / (xup[i] - xlow[i])


# Step 2 - Fit Surrogate Model
fs = np.zeros(nsamples)  # This array will store actual function evaluations of the test sample
fhat = RBFInterpolant(kernel=CubicKernel, tail=LinearTail)  # fhat corresponds to the instance of the cubic surrogate model (overwritten by use of Mixture model in this example)

fm = [] # value from surrogate model
fo = [] # value from the original model 
best = []
parallel_p = 24
j = 1
for i in range(nsamples):  # The Surrogate Model is fitted in this loop where a point from the training test sample is added to the model in each loop iteration
    x = data[i][3:]
    f = data[i][2] # Actual (supposedly expensive) evaluation
    fhat.add_point(x, f)  # Add all training points iteratively to fit surrogate model iteratively
    fs[i] = f  # Storing actual evaluations of points in training set for future reference
    if (i + 1) / parallel_p == j:
        best.append(data[: i + 1, 2].min())
        print i
        new_data = np.copy(data[i + 1: i + 1 + parallel_p])
        idx = np.argsort(new_data[:, 1])
        new_data = new_data[idx, :]
        if (i + parallel_p) < nsamples:
            value = fhat.evals(new_data[:, 3:])
            value = np.array(value.ravel())
            fm.append(value[0])
            fo.append(new_data[:, 2])
        else:
            value = fhat.evals(new_data[:, 3:])
            value = np.array(value.ravel())
            fm.append(value[0])
            fo.append(new_data[:, 2])
        j = j + 1
fm = np.asarray(fm)
fo = np.asarray(fo)

colors = ['r', 'blue', 'cyan', 'orange', 'purple', 'black', 'green']
s = 121
fig = plt.figure()
for i in range(len(fm)):
    if i == 0:
        ax = fig.add_subplot(7, 1, i + 1)
        plt.scatter(np.arange(1, len(fm[i]) * (i + 1) + 1), fm[i],s=2*s,  color = colors[i], marker='^', alpha= 0.4)
        plt.scatter(np.arange(1, len(fo[i]) * (i + 1) + 1), fo[i], color = colors [i],s = 2*s, marker='+')
        plt.plot(np.arange(0, 26), best[i] * np.ones(26), 'r-')
        plt.xlim([0, 25])
    else:
        ax = fig.add_subplot(7, 1, i + 1)
        plt.scatter(np.arange(1, len(fm[i])+ 1), fm[i],
                    s=s, color=colors[i], marker='^', alpha=0.4)
        plt.scatter(np.arange(1, len(fo[i])+ 1), fo[i],s = 2*s,  color = colors[i], marker='+')
        plt.plot(np.arange(0, 26), best[i] * np.ones(26), 'r-')
        plt.xlim([0, 25])
    print ((fm[i] - fo[i])**2).mean() # mean squared error measuring the quality of the surrogated model
    print np.corrcoef(fm[i], fo[i]) # correlation coeffication measuring the correct selection of the surrograte model 
    
plt.show()



