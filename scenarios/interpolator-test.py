import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.trajectory_interpolators import polynomial5_interpolation
from robolearn.utils.trajectory_interpolators import spline_interpolation

N = 100
xf = np.array([2, 3, 4, 1])
x0 = np.array([2, 2, 2, 2])
dxf = np.array([0, 0, 0, 0])*N
dx0 = np.array([0, 0, 0, 0])*N
ddxf = np.array([2, 0, 0, 0])*N**2
ddx0 = np.array([0, 0, 0, 0])*N**2
#x, dx, ddx = polynomial5_interpolation(N, xf, x0, dxf, dx0, ddxf, ddx0)
#
#for ii in range(xf.size):
#    plt.plot(ddx[:, ii])
#plt.show()

N = 100
time_points = np.array([0, 5, 7, 10])
via_points = np.array([[2, 7, 8, 10],
                       [7, 1, 3, 2],
                       [1, 2, 4, 9],
                       [4, 1, 4, 4]])

x = spline_interpolation(N, time_points, via_points)

for ii in range(via_points.shape[1]):
    plt.plot(x[:, ii])
plt.show()

