import numpy as np


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

