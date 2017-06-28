import math
import numpy as np
import tf
from scipy import interpolate


def polynomial5_interpolation(N, xf, x0=None, dxf=None, dx0=None, ddxf=None, ddx0=None):
    # Polynomial Hermite 5th order interpolation

    # n_array = np.array(range(N+1))
    n_array = np.array(range(N))
    # n_array = np.linspace(0, 1, N)
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

    # x = np.empty([N+1, dim])
    # dx = np.empty([N+1, dim])
    # ddx = np.empty([N+1, dim])
    x = np.empty([N, dim])
    dx = np.empty([N, dim])
    ddx = np.empty([N, dim])

    for ii in range(dim):
        A = np.array([[1,  1,  1, 1, 1, 1],
                      [5,  4,  3, 2, 1, 0],
                      [20, 12, 6, 2, 0, 0],
                      [0,  0,  0, 1, 0, 0],
                      [0,  0,  0, 0, 1, 0],
                      [0,  0,  0, 0, 0, 1]], dtype=np.float64)

        A *= x_n[-1, :]
        b = np.array([xf[ii], dxf[ii], ddxf[ii], ddx0[ii], dx0[ii], x0[ii]])
        coeffs = np.linalg.solve(A, b)

        x[:, ii] = np.sum(coeffs * x_n, axis=1)
        dx[:, ii] = np.sum(coeffs[:-1] * dx_n, axis=1)
        ddx[:, ii] = np.sum(coeffs[:-2] * ddx_n, axis=1)

    return x, dx, ddx


def lspv_interpolation(N, xf, x0=None, v=None):
    """
    Linear Segment with parabolic blend.
    Based on Robotics Toolbox for MATLAB (release 10.1): (c) Peter Corke 1992-2011 http://www.petercorke.com
    :param N: Steps
    :param xf: final point
    :param x0: initial point
    :param v: velocity of the linear segment
    :return: 
    """
    dim = xf.size
    tend = N

    if x0 is None:
        x0 = np.zeros_like(xf)

    if v is None:
        v = (xf-x0)/N * 1.5
    else:
        v = np.abs(v) * np.sign(xf-x0)

        for ii in range(dim):
            if np.abs(v[ii]) < np.abs(xf-x0)[ii]/N:
                raise AttributeError('Velocity %d too small: %f' % (ii, v[ii]))
            elif np.abs(v[ii]) > 2*np.abs(xf-x0)[ii]/N:
                raise AttributeError('Velocity %d too big: %f' % (ii, v[ii]))

    if np.array_equal(x0, xf):
        return np.tile(xf, (N, 1)), np.zeros((N, dim)), np.zeros((N, xf.size))

    tb = np.divide((x0 - xf + v*tend), v)
    a = np.divide(v, tb)

    xs = np.zeros((N, dim))
    xds = np.zeros((N, dim))
    xdds = np.zeros((N, dim))

    for ii in range(dim):
        for tt in range(N):
            if tt < tb[ii]:
                # Initial blend
                xs[tt, ii] = x0[ii] + a[ii]/2*tt**2
                xds[tt, ii] = a[ii]*tt
                xdds[tt, ii] = a[ii]
            elif tt <= (tend[ii] - tb[ii]):
                # Linear motion
                xs[tt, ii] = (xf[ii] + x0[ii] - v[ii]*tend)/2 + v[ii]*tt
                xds[tt, ii] = v[ii]
                xdds[tt, ii] = 0
            else:
                # Final blend
                xs[tt, ii] = xf[ii] - a[ii]/2*tend**2 + a[ii]*tend*tt + a[ii]/2*tt**2
                xds[tt, ii] = a[ii]*tend - a[ii]*tt
                xdds[tt, ii] = -a[ii]

    return xs, xds, xdds


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


def ang_vel_from_quaternions(q0, q1):
    # From https://www.gamedev.net/forums/topic/347752-quaternion-and-angular-velocity/
    q = tf.transformations.quaternion_multiply(q1, tf.transformations.quaternion_conjugate(q0))

    len = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
    angle = 2 * math.atan2(len, q[3])
    axis = q[:3] / len if len > 0 else np.array([1, 0, 0])
    #print(q)
    #print(axis)
    #print(angle)
    #print(axis*angle)
    #print("---")
    return axis*angle


def quaternion_slerp_interpolation(N, q_end, q_init=None):
    if q_init is None:
        q_init = np.array([0, 0, 0, 1])

    quat_traj = np.empty((N, 4))
    ang_vel_traj = np.empty((N, 3))
    linspace_interp = np.linspace(0, 1, N)
    for ii in range(N):
        quat_traj[ii, :] = tf.transformations.quaternion_slerp(q_init, q_end, linspace_interp[ii])

    # Angular velocity
    for ii in range(N-1):
        ang_vel_traj[ii, :] = ang_vel_from_quaternions(quat_traj[ii], quat_traj[ii+1])
    ang_vel_traj[-1, :] = [0, 0, 0]

    # Angular acceleration
    ang_acc_traj = np.vstack((np.diff(ang_vel_traj, axis=0), np.zeros((1, 3))))

    return quat_traj, ang_vel_traj, ang_acc_traj
