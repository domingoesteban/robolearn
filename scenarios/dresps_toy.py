import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize



# seed = 5

# dt = 0.1
# Tend = 5
# x_des = np.array([3, 0])
# x0 = np.array([0, 0])
# init_var = 0.000010
#
# # Learning Algo
# M = 5  # Number of Samples
# N = 10  # Number of iterations
# epsilon =
# beta0 = 0.1
#
# np.random.seed(seed)
#
#
# def transition(x, u, noisy=True):
#     """ Double integrator """
#     if noisy:
#         noise = np.random.randn(x.shape[0])*0.001
#     else:
#         noise = np.random.randn(x.shape[0])*0
#
#     x_dot = np.array([[0, 1], [0, 0]]).dot(x) + np.array([[0], [1]]).dot(u)
#     x_new = x + dt*x_dot + noise
#     return x_new
#
#
# def cost(x, u):
#     """  """
#     c = 1./2 * np.sum((x - x_des)**2) + 1./2 * np.sum(u**2)
#     return c
#
#
# def eval_cost(xs, us):
#     if len(xs.shape) == 2:
#         T = xs.shape[0]
#         cs = np.zeros((T, 1))
#         for tt in range(T):
#             cs[tt] = cost(xs[tt, :], us[tt, :])
#     elif len(xs.shape) == 3:
#         M = xs.shape[0]
#         T = xs.shape[1]
#         cs = np.zeros((M, T, 1))
#         for mm in range(M):
#             for tt in range(T):
#                 cs[mm, tt] = cost(xs[mm, tt, :], us[mm, tt, :])
#     else:
#         raise ValueError
#
#     return cs
#
#
#
# class TVLGC(object):
#     def __init__(self, T, dU, dX, init_var=1):
#         self.K = np.zeros((T, dU, dX))
#         self.k = np.zeros((T, dU))
#         self.Sigma = np.tile(np.eye(dU)*init_var, (T, 1))
#         self.inv_Sigma = np.tile(np.linalg.inv(np.eye(dU)*init_var), (T, 1))
#         self.kol_Sigma = np.tile(np.linalg.cholesky(np.eye(dU)*init_var), (T, 1))
#
#     def eval(self, x, t, noise):
#         return self.K[t].dot(x) + self.k[t] + self.kol_Sigma[t].dot(noise)
#
#     def fit(self, u):
#         pass
#
#     def get_params(self):
#         return {'K': self.K.copy(), 'k': self.k.copy(), 'Sigma': self.Sigma.copy()}
#
#
# T = int(Tend / dt)
# dA = 1
# dS = 2
#
#
# policy = TVLGC(T, dA, dS, init_var=init_var)
#
# noises = np.random.randn(M, T, dA)
#
# # Initial u
# actions = np.zeros((M, T, dA))
# states = np.zeros((M, T, dS))
# costs = np.zeros((M, T, 1))
# for mm in range(M):
#     # Rollout
#     states[mm, 0, :] = x0
#     for tt in range(T-1):
#         actions[mm, tt, :] = policy.eval(states[mm, tt, :], tt, noises[mm, tt, :])
#         states[mm, tt + 1, :] = transition(states[mm, tt, :], actions[mm, tt, :])
#
# # Eval costs
# costs = eval_cost(states[0, :, :], actions[0, :, :])
#
# fig, ax = plt.subplots(3, 1)
#
# time = np.arange(0, T*dt, dt)
# ax[0].plot(time, states[0, :, 0])
# ax[1].plot(time, states[0, :, 1])
# ax[2].plot(time, actions[0, :, 0])
#
# plt.show()




def multimodal_cost_function(theta):
    coef1 = 10
    coef2 = 5
    psi1 = np.array([-0.5, 0.5])
    psi2 = np.array([-0.2, -0.2])
    psi3 = np.array([0.2, 0.2])

    psi = np.array([psi1, psi2, psi3])
    theta = np.array(theta)
    first_term = coef1*np.linalg.norm(np.sum(psi/3, axis=0) - theta)
    second_term = coef2*np.sum(np.linalg.norm(psi - theta, axis=1))
    return first_term - second_term


N = 50
M = 100  # Total Samples
max_prev_samples = 400

seed = 5

epsilon = 0.5
xi = 5
chi = 2 * epsilon

L2_reg_dual = 0.
L2_reg_loss = 0.

optimizer = scipy.optimize.fmin_l_bfgs_b


thetas = np.zeros((N, M, 2))
Js_reps = np.zeros((N, M))
learning_curve_reps = np.zeros(N)

np.random.seed(seed)

# Initial policy
omega = [0, 0, 1, 1]
mu = np.zeros(2)
Sigma = np.eye(2)
for nn in range(N):
    # Sample
    mu[:] = omega[:1]
    n = Sigma.shape[0]
    Sigma[range(n), range(n)] = omega[2:]
    thetas[nn, :, :] = np.random.multivariate_normal(mu, Sigma, M)

    # Policy Evaluation
    for mm in range(M):
        Js_reps[nn, mm] = multimodal_cost_function(thetas[nn, mm])
    learning_curve_reps[nn] = np.mean(Js_reps[nn, :])

    # Policy Optimization
    eta = 0






# Plot

thetas0 = np.linspace(-1, 1, 100)
thetas1 = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(thetas0, thetas1)
n_plot = len(X)

Jvals = np.zeros(shape=(thetas0.size, thetas1.size))

#Fill out J_vals
for xx in range(len(thetas0)):
    for yy in range(len(thetas1)):
        Jvals[xx, yy] = multimodal_cost_function([X[xx, yy], Y[xx, yy]])

plt.figure()
#contour = plt.contourf(X, Y, Jvals)
# plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)
# c = ('#ff0000', '#ffff00', '#0000FF', '0.01', 'c', 'm')
# contour_filled = plt.contourf(X, Y, Jvals, colors=c)
# plt.colorbar(contour)
cp = plt.contourf(X, Y, Jvals)
plt.colorbar(cp)
plt.title('Reward Function')
plt.xlabel('theta0')
plt.ylabel('theta1')


plt.scatter(thetas[-1, :, 0], thetas[0, :, 1], color='red')

axes = plt.gca()
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])


plt.figure()
plt.plot(learning_curve_reps)
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Return')

plt.show()

