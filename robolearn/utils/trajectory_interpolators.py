import numpy as np
from scipy import interpolate
from pyquaternion import Quaternion

def polynomial5_interpolation(N, xf, x0=None, dxf=None, dx0=None, ddxf=None, ddx0=None):
    # Polynomial Hermite 5th order interpolation

    n_array = np.array(range(N+1))
    x_n = np.array([[n**5, n**4, n**3, n**2, n, 1] for n in n_array])
    dx_n = np.array([[5*n**4, 4*n**3, 3*n**2, 2*n, 1] for n in n_array])
    ddx_n = np.array([[20*n**3, 12*n**2, 6*n, 2] for n in n_array])

    xf = np.array(xf)#.reshape(1, -1)
    dim = xf.size#.shape[1]

    if x0 is None:
        x0 = np.zeros_like(xf)
    else:
        x0 = np.array(x0)#.reshape(1, -1)
    if dxf is None:
        dxf = np.zeros_like(xf)
    else:
        dxf = np.array(dxf)#.reshape(1, -1)
    if dx0 is None:
        dx0 = np.zeros_like(x0)
    else:
        dx0 = np.array(dx0)#.reshape(1, -1)
    if ddxf is None:
        ddxf = np.zeros_like(xf)
    else:
        ddxf = np.array(ddxf)#.reshape(1, -1)
    if ddx0 is None:
        ddx0 = np.zeros_like(x0)
    else:
        ddx0 = np.array(ddx0)#.reshape(1, -1)

    x = np.empty([N+1, dim])
    dx = np.empty([N+1, dim])
    ddx = np.empty([N+1, dim])

    for ii in range(dim):
        A = np.array([[1,  1,  1, 1, 1, 1],
                      [5,  4,  3, 2, 1, 0],
                      [20, 12, 6, 2, 0, 0],
                      [0,  0,  0, 1, 0, 0],
                      [0,  0,  0, 0, 1, 0],
                      [0,  0,  0, 0, 0, 1]])

        A *= x_n[-1, :]
        b = np.array([xf[ii], dxf[ii], ddxf[ii], ddx0[ii], dx0[ii], x0[ii]])
        coeffs = np.linalg.solve(A, b)

        x[:, ii] = np.sum(coeffs * x_n, axis=1)
        dx[:, ii] = np.sum(coeffs[:-1] * dx_n, axis=1)
        ddx[:, ii] = np.sum(coeffs[:-2] * ddx_n, axis=1)

    return x, dx, ddx


def spline_interpolation(N, time_points, via_points):
    n_points = via_points.shape[0]
    dim = via_points.shape[0]

    if n_points != len(time_points):
        raise ValueError("Via points and time_points do not have the same size!")

    #total_points = int(np.ceil(time_points[-1]-time_points[0])/N)
    #tcks = [None for _ in range(n_points)]

    x = np.empty([N, dim])

    time_new = np.linspace(time_points[0], time_points[-1], N)

    spline_degree = min(n_points-1, 3)

    for ii in range(dim):
        tck = interpolate.splrep(time_points, via_points[:, ii], s=0, k=spline_degree)

        x[:, ii] = interpolate.splev(time_new, tck, der=0)

    return x


def quaternion_interpolation(N, q_end, q_init=None):
    if q_init is None:
        q_init = np.array([0, 0, 0, 1])

    #q_end = Quaternion(q_end[3], q_end[0], q_end[1], q_end[2])
    q_end = Quaternion(q_end[[3, 0, 1, 2]])
    #q_init = Quaternion(q_init[3], q_init[0], q_init[1], q_init[2])
    q_init = Quaternion(q_init[[3, 0, 1, 2]])

    quat_traj = np.empty([N+1, 4])

    for ii in range(N+1):
        temp_quat = Quaternion.slerp(q_init, q_end, amount=float(ii)/N)
        quat_traj[ii, 3] = temp_quat.scalar
        quat_traj[ii, :3] = temp_quat.vector

    return quat_traj