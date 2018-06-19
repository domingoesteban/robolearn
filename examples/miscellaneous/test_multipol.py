import torch
from robolearn.torch.sac.policies import TanhGaussianMultiPolicy
from robolearn.torch.sac.value_functions import NNMultiQFunction

nn_pol = TanhGaussianMultiPolicy([5], 5, [2, 1], [3])

a = torch.Tensor([5, 0.2, 9, 0.4, 0.5])

b0 = torch.Tensor([0.3, -0.2])
b1 = torch.Tensor([-0.8])

# o = nn_pol(a, _val_idxs=[0], deterministic=True)
# error = torch.sum(b0 - o[0][0])

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-2
optimizer_pol = torch.optim.Adam(nn_pol.parameters(), lr=learning_rate)

# y0 = b0
y1 = b1
for tt in range(50):
    # y_pred = nn_pol(a, _val_idxs=[0, 1], deterministic=True)[0]
    y_pred = nn_pol(a, pol_idxs=[1], deterministic=True)[0]
    # loss = loss_fn(y_pred[0], y0) + loss_fn(y_pred[1], y1)
    loss = loss_fn(y_pred[0], y1)
    print(tt, loss.item())

    optimizer_pol.zero_grad()
    loss.backward()
    optimizer_pol.step()

# error.backward()

# for name, param in nn_pol.named_parameters():
#     print('+'*10)
#     print(name)
#     print('+'*10)
#     print('VALS', param)
#     print('GRAD', param.grad)
#     print('\n')
print('='*10)
print('='*10)

# print(a)
# print(o[0])
# print(b0)

# y_pred = nn_pol(a, _val_idxs=[0, 1], deterministic=True)[0]
y_pred = nn_pol(a, pol_idxs=[1], deterministic=True)[0]
print('Prediction', y_pred)

a_np = a.numpy()
y_pred_np = nn_pol.get_action(a_np, pol_idxs=[1], deterministic=True)[0]
print('Prediction_np', y_pred_np)


# input('Now train Value Fcn')

# ###
# ###
# ###
# ###
# ###

values = torch.Tensor([10.11, -120.56, 150.9923])
nn_val = NNMultiQFunction(5, 2, 3, [3], [2])

optimizer_val = torch.optim.Adam(nn_val.parameters(), lr=learning_rate)

obs = torch.Tensor([4, 5, -10, -100, 2])
acts = torch.Tensor([2.1, -100.2])

y0 = values[0]
y1 = values[1]
y2 = values[2]
for tt in range(20000):
    # y_pred = nn_pol(a, _val_idxs=[0, 1], deterministic=True)[0]
    y_pred = nn_val(obs, acts, val_idxs=[0, 1, 2])
    loss = loss_fn(y_pred[0], y0) + loss_fn(y_pred[1], y1) + loss_fn(y_pred[2], y2)
    # loss = loss_fn(y_pred[0], y1)
    print(tt, loss.item())

    optimizer_val.zero_grad()
    loss.backward()
    optimizer_val.step()

y_pred = nn_val(obs, acts, val_idxs=[0, 1, 2])
print('Prediction Value', y_pred, 'Desired:', values)

# for name, param in nn_val.named_parameters():
#     print('+'*10)
#     print(name)
#     print('+'*10)
#     print('VALS', param)
#     print('GRAD', param.grad)
#     print('\n')
