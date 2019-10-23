# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:04:11 2019

@author: Jorge C. Lucero

This script aligns a set of lip position curves using nonlinear time warping.
It uses the package of R routines for functional data analysis developed by
James O. Ramsay, available at
ftp://ego.psych.mcgill.ca/pub/ramsay/FDAfuns/Matlab/
"""

import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri
import numpy as np
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt

# Constants

FS = 200  # Sampling frequency
NBASIS_DATA = 32  # Number of basis functions for lip data
NBASIS_WARP = 10  # Number of basis functions for warping functions
NORDER = 4  # Order
RANGEVAL = np.array([0, 1])  # Normalized time range
LFDOBJ = 2  # Penalize size of the second derivative
LAMBDA = .0001  # Roughness penalty coefficient

# Close all figures

plt.close('all')

# Prepare fda package

utils = rpackages.importr('utils')
fda = rpackages.importr('fda')
numpy2ri.activate()

# Read data

# The data consist of a 2D array ('data'), which contains records of lip
# position in columns, padded with zeros to a common length. The records
# represent the position of the lower lip during the repeated production
# of the sentence "Buy Bobby a puppy", collected at 200 Hz sampling rate.
# The first element of each column is the number of samples for that record,
# and the remaining elements are the lip position measurements.

data = np.loadtxt('data.txt')

nrecords = data.shape[1]
rlength = data[0, :].astype(int)
records = data[1:, :]

# Interpolate records to common length and center vertically

clength = max(rlength)
lipmat = np.zeros((clength, nrecords))

for i in range(nrecords):
    tck = splrep(np.linspace(0, 1, rlength[i]), records[:rlength[i], i], s=0)
    lipmat[:, i] = splev(np.linspace(0, 1, clength), tck, der=0)

lipmat = lipmat - lipmat.mean()
liptime = np.linspace(0, 1, clength)[:, np.newaxis]

# Create a B-splines basis for the data

# It is not clear how to determine the appropiate number of splines in the
# basis. One technique is just visual inspection. Another one might be:
# the basis expansion might be consider as a low-pass filtering process,
# see, e.g., Unser et al., "B-spline signal processing: Part I - Theory",
# IEEE Trans. Signal Processing 41, 821-833 (1993), and "B-spline signal
# processing: Part II - Efficient design and applications", IEEE Trans.
# Signal Processing 41, 834-848 (1993). Letting ni be the number of smaples
# of record i, and nknots the number of knots or time breaks for the
# B-splines, then the frequency cutoff of such filtering process is
# f = (fsampling/2)*(nbreaks/ni). In case of a base of 32 B-splines of
# order 4, nbreaks = nbasis + 2 - norder = 30. Letting also fsampling = 200
# Hz and ni = 200 points, we get f = 15 Hz which is ok for lip motion.

lipbasis = fda.create_bspline_basis(RANGEVAL, NBASIS_DATA, NORDER)

# Create functional data object

lipfdLst = fda.smooth_basis(liptime, lipmat, lipbasis)
lipfd = lipfdLst.rx2('fd')  # Extract the fd object

# Create basis for the warping functions

# A basis of 10 splines (cutoff frequency of about 4 Hz) seems to work fine,
# according to visual inspection of the results.

wbasis = fda.create_bspline_basis(RANGEVAL, NBASIS_WARP, NORDER)

# Create functional parameter object

wfdPar = fda.fdPar(wbasis, LFDOBJ, LAMBDA)

# Alignment (registration)

target = fda.mean(lipfd)  # Target for the alignment
lipregfdLst = fda.register_fd(target, lipfd, wfdPar)

lipregfd = lipregfdLst.rx2('regfd')
warpfd = lipregfdLst.rx2('warpfd')
Wfd = lipregfdLst.rx2('Wfd')

# Refine the alignment

wfdPar = fda.fdPar(Wfd, LFDOBJ, LAMBDA)
target = fda.mean(lipregfd)
lipregfdLst = fda.register_fd(target, lipfd, wfdPar)

lipregfd = lipregfdLst.rx2('regfd')
warpfd = lipregfdLst.rx2('warpfd')
Wfd = lipregfdLst.rx2('Wfd')

# Evaluate warping functions

# The warping functions for each record are obtained as columns of a
# matrix (warpmat). They are evaluated over a time scale in the range
# [0,1], and with a number of samples equal to the largest data record
# length.

warpmat = np.array(fda.eval_monfd(liptime, Wfd))
warpmat = warpmat/(warpmat[-1, :])

# The deformation functions d(t) are defined as d(t) = h(t) - t, and
# measure the time shift of each data point from its position at the target
# curve.

defmat = warpmat - liptime

# Compute mean and variability of aligned curves

lipmeanfd = fda.mean_fd(lipregfd)
lipstdfd = fda.std_fd(lipregfd)
wstd = np.std(defmat, axis=1)

# Plot results

plt.figure(1)

# Plot data

plt.subplot(3, 1, 1)
plt.title('Lip data')
for i in range(nrecords):
    plt.plot(np.arange(rlength[i])/FS, records[:rlength[i], i])
plt.ylabel('Position (cm)')
plt.xlabel('Time (s)')

# Plot functional data

plt.subplot(3, 1, 2)
plt.plot(liptime, fda.eval_fd(liptime, lipfd))
plt.title('Functional data')
plt.xlabel('Norm time')
plt.ylabel('Position (mm)')

# Plot aligned data

plt.subplot(3, 1, 3)
plt.plot(liptime, fda.eval_fd(liptime, lipregfd))
plt.title('Aligned data')
plt.xlabel('Norm time')
plt.ylabel('Position (mm)')

plt.tight_layout()

plt.figure(2)

# Plot mean of aligned curves

plt.subplot(2, 1, 1)
plt.plot(liptime, fda.eval_fd(liptime, lipmeanfd))
plt.title('Mean of aligned curves')
plt.xlabel('Norm time')
plt.ylabel('Position (mm)')

# Plot standard deviation of aligned curves

plt.subplot(2, 1, 2)
plt.plot(liptime, fda.eval_fd(liptime, lipstdfd))
plt.title('Standard deviation of aligned curves')
plt.xlabel('Norm time')
plt.ylabel('Position (mm)')

plt.tight_layout()

plt.figure(3)

# Plot deformation functions

plt.subplot(2, 1, 1)
plt.plot(liptime, defmat)
plt.title('Deformation functions')
plt.xlabel('Norm time')
plt.ylabel('Norm time')

# Plot standard deviation of deformation functions

plt.subplot(2, 1, 2)
plt.plot(liptime, wstd)
plt.title('Standard deviation of deformation functions')
plt.xlabel('Norm time')
plt.ylabel('Norm time')

plt.tight_layout()

plt.show()
