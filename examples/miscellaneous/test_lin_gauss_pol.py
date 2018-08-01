import torch
from robolearn.torch.policies import LinearGaussianPolicy
import matplotlib.pyplot as plt

batch_size = 5
obs_dim = 3
action_dim = 3
n_policies = 2

Tend = 5
Ts = 0.01
T = int(Tend/Ts)

time = torch.linspace(0, Tend-Ts, T)
des_obs_x = torch.sin(time)
des_obs_y = torch.cos(time)
des_obs_z = torch.cos(time*2)
obs = torch.cat([
    des_obs_x.unsqueeze(-1),
    des_obs_y.unsqueeze(-1),
    des_obs_z.unsqueeze(-1),
    ],
    dim=-1,
)

nn_pol = LinearGaussianPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    T=T,
    )

nn_pol.K.data.uniform_(-1, 1)
nn_pol.k.data.uniform_(-1, 1)

print('##'*10)
print(nn_pol)
print('##'*10)
print('MODULE PARAMETERS:')
for name, p in nn_pol.named_parameters():
    print(name, p.shape, 'grad?', p.requires_grad)
print('##'*10)

for name, p in nn_pol.named_parameters():
    print(name, '\n', p)
print('##'*10)
# print('SHARED PARAMETERS:')
# for name, p in nn_pol.named_shared_parameters():
#     print(name, p.shape)
#     print(p.data)
#     print('.')
# print('##'*10)
# print('MIXING PARAMETERS:')
# for name, p in nn_pol.named_mixing_parameters():
#     print(name, p.shape)
#     print(p.data)
#     print('.')
# print('##'*10)
# print('ALL POLICIES PARAMETERS:')
# for name, p in nn_pol.named_policies_parameters():
#     print(name, p.shape)
#     print(p.data)
#     print('.')
# print('##'*10)
# print('SPECIFIC POLICY PARAMETERS:')
# for pol_idx in range(nn_pol.n_heads):
#     print('--- POLICY ', pol_idx, ' ---')
#     for name, p in nn_pol.named_policies_parameters(idx=pol_idx):
#         print(name, p.shape)
#         print(p.data)
#         print('.')
# print('##\n'*5)
# for param in nn_pol.parameters():
#     print(param.shape)

print('##\n'*5)
# input("Press a key to start training...")


# obs = torch.rand((T, obs_dim))


act_des = torch.rand((T, action_dim))
# act_des = torch.tensor([[0.1],
#                        [0.1],
#                        [0.3]])
act_des = act_des.uniform_(-1, 1)

# o = nn_pol(a, _val_idxs=[0], deterministic=True)
# error = torch.sum(b0 - o[0][0])

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-2
optimizer_pol = torch.optim.Adam(nn_pol.parameters(), learning_rate)

print('obs shape:', obs.shape)
print('action shape:', act_des.shape)


params_initial = list()
for param in nn_pol.parameters():
    params_initial.append(param.data.clone())

output_initial = nn_pol(obs)


Fdyn = torch.rand((T, obs_dim, action_dim))
fdyn = torch.rand((T, obs_dim))


def sim_dynamics(obs0, policy):
    # next_obs = torch.zeros((T, obs_dim), requires_grad=True)
    # next_obs.data[0, :] = obs0
    obs_list = []
    obs_list.append(obs0.unsqueeze(0))
    for t in range(T-1):
        obs_t = obs_list[-1]
        act = policy(obs_t, t=t)[0]
        # next_obs.data[t+1, :] = next_obs.data[t+1, :] + \
        #     (torch.sum(Fdyn[t, :, :]*act.unsqueeze(dim=-2),
        #                dim=-1) + fdyn[t, :])
        dobs_t = torch.sum(Fdyn[t, :, :]*act.unsqueeze(dim=-1), dim=-1) \
                 + fdyn[t, :]
        obs_tp1 = obs_t + dobs_t*Ts
        obs_list.append(obs_tp1)
        # next_obs[t+1, :] = next_obs[t, :] + act*Ts
        # print(next_obs.data[t, :])
        # print(act)
        # print(next_obs.data[t+1, :])
        # input('wuuu')
    next_obs = torch.cat(obs_list, dim=0)

    return next_obs


next_obs_initial = sim_dynamics(obs[0, :], nn_pol)

for tt in range(2500):
    # act_pred, policy_info = nn_pol(obs)

    next_obs = sim_dynamics(obs[0, :], nn_pol)

    # loss = loss_fn(act_pred, act_des)
    loss = loss_fn(next_obs[:-1, :], obs[1:, :])
    # loss = loss_fn(policy_mean, act_des)
    # loss = loss_fn(policy_log_std, act_des)
    # loss = loss_fn(pre_tanh_value, act_des)

    print('t=', tt, '| loss=', loss.item())

    optimizer_pol.zero_grad()
    loss.backward()

    if tt == 0:
        print('Showing the gradients')
        for name, param in nn_pol.named_parameters():
            print('----')
            print(name, '\n', param.grad)
        # input('Press a key to continue training...')

    optimizer_pol.step()

# error.backward()

next_obs = sim_dynamics(obs[0, :], nn_pol)

fig, axs = plt.subplots(obs_dim, 1)

for ii in range(obs_dim):
    axs[ii].plot(time[1:].data.numpy(), obs[1:, ii].data.numpy(), label='des')
    axs[ii].plot(time[1:].data.numpy(), next_obs[:-1, ii].data.numpy(), label='obtained')
    axs[ii].plot(time[1:].data.numpy(), next_obs_initial[:-1, ii].data.numpy(), label='initial')
    axs[ii].legend()
plt.show()

# print('='*10)
# print('='*10)
# output = nn_pol(obs)
# print('Initial output')
# for key, val in output_initial[1].items():
#     print(key, '\n', val)
# print('==')
# print('Final output')
# for key, val in output[1].items():
#     print(key, '\n', val)
# print('action_des', act_des)
# print('action_pred_initial', output_initial[0])
# print('action_pred', output[0])
# print('action_one_by_one')
# for ii in range(batch_size):
#     print(ii, '-->', nn_pol(obs[ii])[0])
#
# print('_______ DEBUG___')
# nn_pol(obs)


input('Show parameters...')

print('##\n'*2)

params_final = list()
for param in nn_pol.shared_parameters():
    params_final.append(param.data.clone())

print('##\n'*2)
print('LOSS', loss)
for name, param in nn_pol.named_parameters():
    print('--')
    print('NAME', name)
    print('DATA', param.data)
    print('GRAD', param.grad)

print('init_shared')
print(params_initial)
print('final_shared')
print(params_final)
input('wuuu')
