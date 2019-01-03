import torch
import robolearn.torch.utils.pytorch_util as ptu

from robolearn.torch.policies import TanhGaussianPolicy
from robolearn.torch.policies import TanhMlpPolicy
from robolearn.models.policies import ExplorationPolicy


obs_dim = 3
act_dim = 2

# nn_pol = TanhGaussianPolicy(hidden_sizes=[4],
#                             obs_dim=obs_dim,
#                             action_dim=act_dim,
#                             std=None,
#                             hidden_w_init=ptu.xavier_initOLD,
#                             hidden_b_init_val=0,
#                             output_w_init=ptu.xavier_initOLD,
#                             output_b_init_val=0)

nn_pol = TanhMlpPolicy(hidden_sizes=[4],
                       obs_dim=obs_dim,
                       action_dim=act_dim,
                       hidden_w_init=ptu.xavier_initOLD,
                       hidden_b_init_val=0,
                       output_w_init=ptu.xavier_initOLD,
                       output_b_init_val=0)
# nn_pol = MlpPolicy(hidden_sizes=[4],
#                    obs_dim=obs_dim,
#                    action_dim=act_dim,
#                    hidden_w_init=ptu.xavier_initOLD,
#                    hidden_b_init_val=0,
#                    output_w_init=ptu.xavier_initOLD,
#                    output_b_init_val=0)

print("Policy: '", TanhGaussianPolicy.__name__, "' parameters:")
for name, param in nn_pol.named_parameters():
    print('name: ', name, '| shape: ', param.data.shape)


obs = torch.rand(obs_dim)
print('\n')
print('Evaluate with one obs:')
output = nn_pol(obs)
print('action: ', output[0])

print('\n')
print('Evaluate with five obs:')
obs = torch.rand((5, obs_dim))
output = nn_pol(obs)
print('actions: ', output[0])

print('\n')
print('Evaluate with one np_obs:')
obs = torch.rand(obs_dim).data.numpy()
output = nn_pol.get_action(obs)
print('action: ', output[0], '| shape: ', output[0].shape)

print('\n')
print('Evaluate with five np_obs:')
obs = torch.rand((5, obs_dim)).data.numpy()
output = nn_pol.get_actions(obs)
print('action: ', output[0], '| shape: ', output[0].shape)


print('\n')
print('Dummy optimization:')
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-2
optimizer_pol = torch.optim.Adam(nn_pol.parameters(), lr=learning_rate)
des_action = torch.ones(act_dim) * 5
# des_action = torch.ones(act_dim) * 0.5

obs = torch.rand(obs_dim)

for i in range(500):
    if isinstance(nn_pol, ExplorationPolicy):
        a_pred = nn_pol(obs, deterministic=False)
    else:
        a_pred = nn_pol(obs)

    loss = loss_fn(a_pred[0], des_action)
    print('iter: ', i, 'loss=', loss.item())

    optimizer_pol.zero_grad()
    loss.backward()
    optimizer_pol.step()

    # for name, param in nn_pol.named_parameters():
    #     print('name: ', name, '| grad: ', param.grad)
    # input('PIPI')

if isinstance(nn_pol, ExplorationPolicy):
    print('desired:', des_action,
          ' | expected: ', nn_pol(obs, deterministic=True)[0])
else:
    print('desired:', des_action,
          ' | expected: ', nn_pol(obs)[0])

