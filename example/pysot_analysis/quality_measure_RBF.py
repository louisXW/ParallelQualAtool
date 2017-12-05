import numpy as np
import matplotlib.pyplot as plt
from pySOT import *
# import pySOT.gp_regression as GPRegression
# import pySOT.mars_interpolant as MARSInterpolant
# from pySOT import MARSInterpolant
import pandas as pd
from pandas.tools.plotting import parallel_coordinates


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
                    str3 = str1[0:2] + ['5'] + str2
                    str3 = [float(item) for item in str3]
                    data.append(str3)
                else:
                    str3 = [float(item) for item in str3]
                    data.append(str3)
        data = np.asarray(data)
        return data
# filename = "pysot_result_9_extraall.txt"
# data1 = read_measured_data(filename)
# filename = "pysot_result_9_extra24.txt"
# data2 = read_measured_data(filename)
# filename = "pysot_result_9_PSO.txt"
# data3 = read_measured_data(filename)
filename = "pysot_result_9_24.txt"
data4 = read_measured_data(filename)
# filename = "pysot_result_9_dynamic_extrall.txt"
# data5 = read_measured_data(filename)
# filename = "pysot_result_9_dynamic.txt"
# data6 = read_measured_data(filename)
# data = np.concatenate((data1, data2), axis=0)
# data = np.concatenate((data, data3), axis=0)
# data = np.concatenate((data, data4), axis=0)
# data = np.concatenate((data, data5), axis=0)
# data = np.concatenate((data, data6), axis=0)
data = data4
xlow = np.array([0.1, 0.1, 0.1, 0, 0, 0, 0.001, 0.001, 0.02])
xup = np.array([2.0, 1.0, 1.0, 0.005, 0.005, 0.05, 0.002, 0.002, 0.03])

nsamples = len(data)
dim = 9
for i in range(dim):
    for j in range(nsamples):
       data[j][i + 3] = (data[j][ i + 3] - xlow[i]) / (xup[i] - xlow[i])
print data.shape
# In this example, test samples are being generated randomly whereas in the HW you are asked
# load them from a text file. Use the numpy function "loadtext" to load the samples (sample code provided in next commented line):
# xs = np.loadtxt('small_samples.out',ndmin=2)

# Step 2 - Fit Surrogate Model
fs = np.zeros(nsamples)  # This array will store actual function evaluations of the test sample
fhat = RBFInterpolant(kernel=CubicKernel, tail=LinearTail)  # fhat corresponds to the instance of the cubic surrogate model (overwritten by use of Mixture model in this example)
# fhat = RBFInterpolant(kernel=TPSKernel, tail=LinearTail)
# fhat = GPRegression(maxp=500)
# fhat = MARSInterpolant()
# Use RBFInterpolant(kernel=TPSKernel, tail=LinearTail, maxp=500) for RBF surrogate with thin plate spline kernel

fm = []
fo = []
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
            # value = fhat.evals(data[i + 1: nsamples, 3:])
            # value = np.array(value.ravel())
            # fm.append(value[0])
            # fo.append(data[i + 1: i + 1 + parallel_p, 2])
            value = fhat.evals(new_data[:, 3:])
            value = np.array(value.ravel())
            fm.append(value[0])
            fo.append(new_data[:, 2])
        j = j + 1

fm = np.asarray(fm)
fo = np.asarray(fo)
print best
correlation = []
print fo
print fm
colors = ['r', 'blue', 'cyan', 'orange', 'purple', 'black', 'green']
s = 121
fig = plt.figure()

for i in range(len(fm)):
    if i == 0:
        ax = fig.add_subplot(7, 1, i + 1)
        plt.scatter(np.arange(1, len(fm[i]) * (i + 1) + 1), fm[i],s=2*s,  color = colors[i], marker='^', alpha= 0.4)
        plt.scatter(np.arange(1, len(fo[i]) * (i + 1) + 1), fo[i], color = colors [i],s = 2*s, marker='+')
        # plt.plot(np.arange(0, 26), -1.5 * np.ones(26), 'b-' )
        plt.plot(np.arange(0, 26), best[i] * np.ones(26), 'r-')
        plt.xlim([0, 25])
    else:
        # print len(fm[i -1]) * (i) + 1
        # print len(fm[i]) * (i + 1) + 1
        # plt.scatter(np.arange(max(1, len(fm[i -1]) * (i) + 1), max(1, len(fm[i -1]) * (i) + 1) + len(fm[i])), fm[i],s=s,  color = colors[i], marker='^', alpha = 0.4)
        # plt.scatter(np.arange(max(1, len(fo[i -1]) * (i) + 1), max(1, len(fm[i -1]) * (i) + 1) + len(fm[i])), fo[i],s = 2*s,  color = colors[i], marker='+')
        ax = fig.add_subplot(7, 1, i + 1)
        plt.scatter(np.arange(1, len(fm[i])+ 1), fm[i],
                    s=s, color=colors[i], marker='^', alpha=0.4)
        plt.scatter(np.arange(1, len(fo[i])+ 1), fo[i],s = 2*s,  color = colors[i], marker='+')
        # plt.plot(np.arange(0, 26), -1.5 * np.ones(26), 'b-' )
        plt.plot(np.arange(0, 26), best[i] * np.ones(26), 'r-')
        plt.xlim([0, 25])

    print ((fm[i] - fo[i])**2).mean()
    print np.corrcoef(fm[i], fo[i])

# How to evaluate any point of domain via surrogate model (You can evaluate validation samples via the following code using a loop to subsequently compute validation statistics)

par_names = ['Secchi', 'Vicouv', 'Dicouv', 'Vicoww', 'Dicoww', 'OzmidovXlo', 'Stantn', 'Dalton', 'Ccofu&v']
    # plot_approximate_sensitivity(rbf1,dim,xpub,xmin,xmax,par_names)
plt.show()



