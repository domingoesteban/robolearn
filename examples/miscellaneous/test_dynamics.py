import torch
from robolearn.torch.models import TVLGDynamics

horizon = 200
dO = 6
dA = 3
batch = 100

dynamics = TVLGDynamics(horizon=horizon, obs_dim=dO, action_dim=dA)

obs = torch.rand(batch, horizon, dO)
act = torch.rand(batch, horizon, dA)

print('Dynamics parameters:')
for name, parameter in dynamics.named_parameters():
    print(name, parameter.shape)


print('Forward dynamics:')
t = 0
# next_obs = dynamics(obs, act, time=t)

print('Fitting')
dynamics.fit(obs, act)

input('wuuu')
