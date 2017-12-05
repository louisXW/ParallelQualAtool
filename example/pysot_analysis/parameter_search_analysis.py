import numpy as np
# from matplotlib import pyplot
from pySOT import *
# import pySOT.gp_regression as GPRegression
# import pySOT.mars_interpolant as MARSInterpolant
# from pySOT import MARSInterpolant
import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


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
                    str3 = str1[0:2] + ['999'] + str2
                    str3 = [float(item) for item in str3]
                    data.append(str3)
                else:
                    str3 = [float(item) for item in str3]
                    data.append(str3)
        data = np.asarray(data)
        return data

filename = "pysot_result_9_PSO.txt"
data4 = read_measured_data(filename)

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
# fs = np.zeros(nsamples)  # This array will store actual function evaluations of the test sample
# fhat = RBFInterpolant(kernel=CubicKernel, tail=LinearTail)  # fhat corresponds to the instance of the cubic surrogate model (overwritten by use of Mixture model in this example)
#
#
# for i in range(nsamples):  # The Surrogate Model is fitted in this loop where a point from the training test sample is added to the model in each loop iteration
#     x = data[i][3:]
#     f = data[i][2] # Actual (supposedly expensive) evaluation
#     fhat.add_point(x, f)  # Add all training points iteratively to fit surrogate model iteratively
#     fs[i] = f  # Storing actual evaluations of points in training set for future reference

# # How to evaluate any point of domain via surrogate model (You can evaluate validation samples via the following code using a loop to subsequently compute validation statistics)

par_names = ['Secchi', 'Vicouv', 'Dicouv', 'Vicoww', 'Dicoww', 'OzmidovXlo', 'Stantn', 'Dalton', 'Ccofu&v']
    # plot_approximate_sensitivity(rbf1,dim,xpub,xmin,xmax,par_names)

print data[24:48]
prebest = data [0]
newbest = data [0]
i = 1
for index, item in enumerate(data):
    if index / 24 < 1:
        data [index] = item
        if item [2] < newbest [2]:
            newbest = np.copy(item)
        if index % 24 == 23:
            prebest = newbest
    else :
        if item [2] < newbest [2]:
            newbest = np.copy(item)
        data[index][2:] = data[index][2:] - prebest[2:]
        if index % 24 == 23:
            prebest = newbest


np.set_printoptions(precision=3, suppress=True)
print data[24:48]
print data[144:168][:, (2, 10)]


def parameter_changing_analysis(data):

    fig = plt.figure()
    ax1 = fig.add_subplot(161)
    data1 = data[24:48][:, 2:]
    cmap = colors.ListedColormap(['red','gray', 'blue'])
    bounds = [-10, -0.00000001, +0.0000001, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax1.imshow(data1, cmap=cmap, norm=norm, alpha =1, interpolation = 'nearest')
    # draw gridlines
    ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax1.set_xticks(np.arange(-0.5, 10.5, 1));
    ax1.set_yticks(np.arange(-0.50, 24.5, 1));

    ax2 = fig.add_subplot(162)
    data2 = data[48:72][:, 2:]
    cmap = colors.ListedColormap(['red','gray', 'blue'])
    bounds = [-10, -0.00000001, +0.0000001, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax2.imshow(data2, cmap=cmap, norm=norm, alpha =1, interpolation = 'nearest')
    # draw gridlines
    ax2.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax2.set_xticks(np.arange(-0.5, 10.5, 1));
    ax2.set_yticks(np.arange(-0.50, 24.5, 1));

    ax4 = fig.add_subplot(163)
    data4 = data[72:96][:, 2:]
    cmap = colors.ListedColormap(['red','gray', 'blue'])
    bounds = [-10, -0.00000001, +0.0000001, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax4.imshow(data4, cmap=cmap, norm=norm, alpha =1, interpolation = 'nearest')
    # draw gridlines
    ax4.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax4.set_xticks(np.arange(-0.5, 10.5, 1));
    ax4.set_yticks(np.arange(-0.50, 24.5, 1));

    ax5 = fig.add_subplot(164)
    data5 = data[96:120][:, 2:]
    cmap = colors.ListedColormap(['red','gray', 'blue'])
    bounds = [-10, -0.00000001, +0.0000001, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax5.imshow(data5, cmap=cmap, norm=norm, alpha =1, interpolation = 'nearest')
    # draw gridlines
    ax5.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax5.set_xticks(np.arange(-0.5, 10.5, 1));
    ax5.set_yticks(np.arange(-0.50, 24.5, 1));

    ax6 = fig.add_subplot(165)
    data6 = data[120:144][:, 2:]
    cmap = colors.ListedColormap(['red', 'gray', 'blue'])
    bounds = [-10, -0.00000001, +0.0000001, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax6.imshow(data6, cmap=cmap, norm=norm, alpha=1, interpolation='nearest')
    # draw gridlines
    ax6.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax6.set_xticks(np.arange(-0.5, 10.5, 1));
    ax6.set_yticks(np.arange(-0.50, 24.5, 1));

    ax3 = fig.add_subplot(166)
    data3 = data[144:168][:, 2:]
    cmap = colors.ListedColormap(['red','gray', 'blue'])
    bounds = [-10, -0.00000001, +0.0000001, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax3.imshow(data3, cmap=cmap, norm=norm, alpha =1, interpolation = 'nearest')
    # draw gridlines
    ax3.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax3.set_xticks(np.arange(-0.5, 10.5, 1));
    ax3.set_yticks(np.arange(-0.50, 24.5, 1));


    plt.show()
# data_i2 = data[144:168]

parameter_changing_analysis(data)
