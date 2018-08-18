import torch
from robolearn.torch.policies import TanhGaussianComposedMultiPolicy

batch_size = 5
obs_dim = 4
action_dim = 3
n_policies = 2

nn_pol = TanhGaussianComposedMultiPolicy(
    obs_dim,
    action_dim,
    n_policies,
    shared_hidden_sizes=[8],
    embedding_hidden_sizes=[16, 16],
    composition_hidden_sizes=[32, 32],
    shared_layer_norm=True,
    policies_layer_norm=True,
    mixture_layer_norm=True,
    mixing_temperature=1,
)
input('wuuuXXXX023847')

print('##'*10)
print(nn_pol)
print('##'*10)
print('MODULE PARAMETERS:')
for name, p in nn_pol.named_parameters():
    print(name, p.shape)
print('##'*10)
print('SHARED PARAMETERS:')
for name, p in nn_pol.named_shared_parameters():
    print(name, p.shape)
    print(p.data)
    print('.')
print('##'*10)
print('MIXING PARAMETERS:')
for name, p in nn_pol.named_mixing_parameters():
    print(name, p.shape)
    print(p.data)
    print('.')
print('##'*10)
print('ALL POLICIES PARAMETERS:')
for name, p in nn_pol.named_policies_parameters():
    print(name, p.shape)
    print(p.data)
    print('.')
print('##'*10)
print('SPECIFIC POLICY PARAMETERS:')
for pol_idx in range(nn_pol.n_heads):
    print('--- POLICY ', pol_idx, ' ---')
    for name, p in nn_pol.named_policies_parameters(idx=pol_idx):
        print(name, p.shape)
        print(p.data)
        print('.')
print('##\n'*5)
for param in nn_pol.parameters():
    print(param.shape)

print('##\n'*5)
input("Press a key to start training...")


obs = torch.rand((batch_size, obs_dim))

act_des = torch.rand((batch_size, action_dim))
# act_des = torch.tensor([[0.1],
#                        [0.1],
#                        [0.3]])
act_des = act_des.uniform_(-1, 1)

# o = nn_pol(a, _val_idxs=[0], deterministic=True)
# error = torch.sum(b0 - o[0][0])

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-2
optimizer_pol = torch.optim.Adam([
    {'params': nn_pol.mixing_parameters(),
     'lr': learning_rate},
    {'params': nn_pol.policies_parameters(),
     'lr': learning_rate},
    {'params': nn_pol.shared_parameters(),
     'lr': learning_rate},
])

print('obs shape:', obs.shape)
print('action shape:', act_des.shape)


shared_params_initial = list()
for param in nn_pol.shared_parameters():
    shared_params_initial.append(param.data.clone())
policies_params_initial = list()
for param in nn_pol.policies_parameters():
    policies_params_initial.append(param.data.clone())
mixing_params_initial = list()
for param in nn_pol.mixing_parameters():
    mixing_params_initial.append(param.data.clone())

output_initial = nn_pol(obs, deterministic=True)

for tt in range(1000):
    act_pred, policy_info = nn_pol(obs, deterministic=False,
                                   optimize_policies=False,
                                   return_log_prob=True)

    log_pi = policy_info['log_prob']
    policy_mean = policy_info['mean']
    policy_log_std = policy_info['log_std']
    pre_tanh_value = policy_info['pre_tanh_value']
    print(log_pi)

    # loss = loss_fn(act_pred, act_des)
    loss = loss_fn(log_pi, act_des[:, 0].unsqueeze(dim=-1))
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
        input('Press a key to continue training...')

    optimizer_pol.step()

# error.backward()

print('='*10)
print('='*10)
output = nn_pol(obs, deterministic=True)
print('Initial output')
for key, val in output_initial[1].items():
    print(key, '\n', val)
print('==')
print('Final output')
for key, val in output[1].items():
    print(key, '\n', val)
print('action_des', act_des)
print('action_pred_initial', output_initial[0])
print('action_pred', output[0])
print('action_one_by_one')
for ii in range(batch_size):
    print(ii, '-->', nn_pol(obs[ii], deterministic=True)[0])

print('_______ DEBUG___')
nn_pol(obs, deterministic=True, print_debug=True)

print('_______ DEBUG___POL_IDX-0')
nn_pol(obs, deterministic=True, print_debug=True, pol_idx=0)

print('_______ DEBUG___POL_IDX-1')
nn_pol(obs, deterministic=True, print_debug=True, pol_idx=1)


input('Show parameters...')

print('##\n'*2)

shared_params_final = list()
for param in nn_pol.shared_parameters():
    shared_params_final.append(param.data.clone())
policies_params_final = list()
for param in nn_pol.policies_parameters():
    policies_params_final.append(param.data.clone())
mixing_params_final = list()
for param in nn_pol.mixing_parameters():
    mixing_params_final.append(param.data.clone())

print('##\n'*2)
print('LOSS', loss)
for name, param in nn_pol.named_parameters():
    print('--')
    print('NAME', name)
    print('DATA', param.data)
    print('GRAD', param.grad)

print('init_shared')
print(shared_params_initial)
print('final_shared')
print(shared_params_final)
input('wuuu')
