import numpy as np
import scipy as sp
from robolearn.algos.gps.policies.lin_gauss_policy import LinearGaussianPolicy


def init_pd(dU, dX, T,
            state_idx=None,
            dstate_idx=None,
            pos_gains=0.001,
            Kp=1,
            Kv=0.001,
            init_var=0.01,
            ):
    """
    This function initializes the linear-Gaussian controller as a
    proportional-derivative (PD) controller with Gaussian noise. The
    position gains are controlled by the variable pos_gains, velocity
    gains are controlled by pos_gains*vel_gans_mult.
    """

    if not issubclass(type(pos_gains), list) \
            and not issubclass(type(pos_gains), np.ndarray):
        pos_gains = np.tile(pos_gains, dU)
    elif len(pos_gains) == dU:
        pos_gains = pos_gains
    else:
        raise TypeError("noise_var_scale size (%d) does not match dU (%d)"
                        % (len(pos_gains), dU))

    # Choose initialization mode.
    Jac = np.zeros((dU, dX))

    Jac[:, state_idx] = np.eye(dU)*Kp
    if dstate_idx is not None:
        Jac[:, dstate_idx] = np.eye(dU)*Kv

    K = -np.diag(pos_gains).dot(Jac)
    K = np.tile(K, [T, 1, 1])

    # if state_to_pd == 'distance':
    #     k = np.tile(-K[0, :, :].dot(x0), [T, 1])
    # else:
    #     k = np.tile(2*K[0, :, :].dot(x0), [T, 1])
    k = np.tile(np.zeros(dU), [T, 1])

    #k = np.tile(K[0, :, :].dot(x0), [T, 1])
    PSig = init_var * np.tile(np.eye(dU), [T, 1, 1])
    cholPSig = np.sqrt(init_var) * np.tile(np.eye(dU), [T, 1, 1])
    invPSig = (1.0 / init_var) * np.tile(np.eye(dU), [T, 1, 1])

    max_std = np.sqrt(init_var)

    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig, max_std=max_std)


# Original code
def init_lqr(dU, dX, T, dt, x0, stiffness=1.0, stiffness_vel=0.5,
             init_var=0.01, final_weight=1.0, init_acc=None, init_gains=None):
    """
    Return initial gains for a time-varying linear Gaussian controller that
    tries to hold the initial position.
    """
    #TODO: Use packing instead of assuming which indices are the joint angles.

    # Notation notes:
    # L = loss, Q = q-function (dX+dU dimensional),
    # V = value function (dX dimensional), F = dynamics
    # Vectors are lower-case, matrices are upper case.
    # Derivatives: x = state, u = action, t = state+action (trajectory).
    # The time index is denoted by _t after the above.
    # Ex. Ltt_t = Loss, 2nd derivative (w.r.t. trajectory), indexed by time t.

    # Constants.
    idx_x = slice(dX)  # Slices out state.
    idx_u = slice(dX, dX+dU)  # Slices out actions.

    if init_acc is None:
        init_acc = np.zeros(dU)

    if init_gains is None:
        init_gains = np.ones(dU)

    # Set up simple linear dynamics model.
    Fd, fc = guess_dynamics(init_gains, init_acc, dX, dU, dt)

    # Setup a cost function based on stiffness.
    # Ltt = (dX+dU) by (dX+dU) - Hessian of loss with respect to trajectory at
    # a single timestep.
    Ltt = np.diag(np.hstack([stiffness * np.ones(dU),
                             stiffness * stiffness_vel * np.ones(dU),
                             np.zeros(dX - dU*2),
                             np.ones(dU)
                             ]))
    Ltt = Ltt / init_var  # Cost function - quadratic term.
    lt = -Ltt.dot(np.r_[x0, np.zeros(dU)])  # Cost function - linear term.

    # Perform dynamic programming.
    K = np.zeros((T, dU, dX))  # Controller gains matrix.
    k = np.zeros((T, dU))  # Controller bias term.
    PSig = np.zeros((T, dU, dU))  # Covariance of noise.
    cholPSig = np.zeros((T, dU, dU))  # Cholesky decomposition.
    invPSig = np.zeros((T, dU, dU))  # Inverse of covariance.
    vx_t = np.zeros(dX)  # Vx = dV/dX. Derivative of value function wrt to X at time t.
    Vxx_t = np.zeros((dX, dX))  # Vxx = ddV/dXdX at time t.

    # LQR backward pass.
    for t in range(T - 1, -1, -1):
        # Compute Q function at this step.
        if t == (T - 1):
            Ltt_t = final_weight * Ltt
            lt_t = final_weight * lt
        else:
            Ltt_t = Ltt
            lt_t = lt
        # Qtt = (dX+dU) by (dX+dU) 2nd Derivative of Q-function with respect to trajectory (dX+dU).
        Qtt_t = Ltt_t + Fd.T.dot(Vxx_t).dot(Fd)
        # Qt = (dX+dU) 1st Derivative of Q-function with respect to trajectory (dX+dU).
        qt_t = lt_t + Fd.T.dot(vx_t + Vxx_t.dot(fc))

        # Compute preceding value function.
        U = sp.linalg.cholesky(Qtt_t[idx_u, idx_u])
        L = U.T

        invPSig[t, :, :] = Qtt_t[idx_u, idx_u]
        PSig[t, :, :] = sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
        )
        cholPSig[t, :, :] = sp.linalg.cholesky(PSig[t, :, :])
        K[t, :, :] = -sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, Qtt_t[idx_u, idx_x], lower=True)
        )
        k[t, :] = -sp.linalg.solve_triangular(
            U, sp.linalg.solve_triangular(L, qt_t[idx_u], lower=True)
        )
        Vxx_t = Qtt_t[idx_x, idx_x] + Qtt_t[idx_x, idx_u].dot(K[t, :, :])
        vx_t = qt_t[idx_x] + Qtt_t[idx_x, idx_u].dot(k[t, :])
        Vxx_t = 0.5 * (Vxx_t + Vxx_t.T)

    max_std = np.sqrt(init_var)

    return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig, max_std=max_std)


def guess_dynamics(gains, acc, dX, dU, dt):
    """
    Initial guess at the model using position-velocity assumption.
    Note: This code assumes joint positions occupy the first dU state
          indices and joint velocities occupy the next dU.
    Args:
        gains: dU dimensional joint gains.
        acc: dU dimensional joint acceleration.
        dX: Dimensionality of the state.
        dU: Dimensionality of the action.
        dt: Length of a time step.
    Returns:
        Fd: A dX by dX+dU transition matrix.
        fc: A dX bias vector.
    """
    #TODO: Use packing instead of assuming which indices are the joint
    #      angles.
    Fd = np.vstack([
        np.hstack([
            np.eye(dU), dt * np.eye(dU), np.zeros((dU, dX - dU*2)),
                        dt ** 2 * np.diag(gains)
        ]),
        np.hstack([
            np.zeros((dU, dU)), np.eye(dU), np.zeros((dU, dX - dU*2)),
            dt * np.diag(gains)
        ]),
        np.zeros((dX - dU*2, dX+dU))
    ])
    fc = np.hstack([acc * dt ** 2, acc * dt, np.zeros((dX - dU*2))])
    return Fd, fc
