import time
import torch
from robolearn.torch.policies import TanhGaussianWeightedMultiPolicy
from robolearn.torch.models.values import NNMultiQFunction
from robolearn_gym_envs.pybullet import CentauroTrayEnv
from robolearn.utils.data_management import MultiGoalReplayBuffer
import robolearn.torch.pytorch_util as ptu
from torch.autograd import Variable


# SEED
SEED = 48
torch.manual_seed(SEED)

T = 5000
SIM_TIMESTEP = 0.01
FRAME_SKIP = 1
DT = SIM_TIMESTEP * FRAME_SKIP
env_params = dict(
    is_render=False,
    obs_with_img=False,
    active_joints='RA',
    control_type='tasktorque',
    # control_type='torque',
    # control_type='velocity',
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    obs_distances=False,
    balance_cost_weight=2.0,
    fall_cost_weight=0.5,
    tgt_cost_weight=20.0,
    # tgt_cost_weight=50.0,
    balance_done_cost=0.,  # 2.0*PATH_LENGTH,  # TODO: dont forget same balance weight
    tgt_done_reward=0.,  # 20.0,
    # tgt_cost_weight=5.0,
    # balance_cost_weight=0.0,
    # fall_cost_weight=0.0,
    # tgt_cost_weight=0.0,
    # balance_cost_weight=5.0,
    # fall_cost_weight=7.0,
    ctrl_cost_weight=1.0e-1,
    use_log_distances=True,
    log_alpha_pos=1e-4,
    log_alpha_ori=1e-4,
    goal_tolerance=0.05,
    min_obj_height=0.60,
    max_obj_height=1.20,
    max_obj_distance=0.20,
    max_time=None,
    subtask=None,
    # subtask=1,
    random_init=True,

)
env = CentauroTrayEnv(**env_params)

obs_dim = env.obs_dim
action_dim = env.action_dim
n_intentions = env.n_subgoals


# REPLAY BUFFER
BATCH_SIZE = 256
replay_buffer = MultiGoalReplayBuffer(
    max_replay_buffer_size=1e6,
    obs_dim=obs_dim,
    action_dim=action_dim,
    reward_vector_size=n_intentions,
)


# POLICY PARAMS
shared_hidden_sizes = (256, 256)
unshared_hidden_sizes = (256, 256)
unshared_mix_hidden_sizes = (256, 256)

nn_pol = TanhGaussianWeightedMultiPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    n_policies=n_intentions,
    shared_hidden_sizes=shared_hidden_sizes,
    unshared_hidden_sizes=unshared_hidden_sizes,
    unshared_mix_hidden_sizes=unshared_mix_hidden_sizes,
    hidden_activation='relu',
)

# Q_VAL
u_qf = NNMultiQFunction(obs_dim=obs_dim,
                        action_dim=action_dim,
                        n_qs=n_intentions,
                        # shared_hidden_sizes=[net_size, net_size],
                        shared_hidden_sizes=(256, 256),
                        unshared_hidden_sizes=(256, 256))

print('NN MULTI POLICY')
print(nn_pol)
print('**\n'*4)

print("ALL PARAMS")
for name, param in nn_pol.named_parameters():
    print(name, param.shape)
print('**\n'*4)

print("SHARED PARAMS")
for name, param in nn_pol.named_shared_parameters():
    print(name, param.shape)
print('**\n'*4)

print("POL PARAMS")
for name, param in nn_pol.named_policies_parameters():
    print(name, param.shape)
print('**\n'*4)

print("MIX PARAMS")
for name, param in nn_pol.named_mixing_parameters():
    print(name, param.shape)
print('**\n'*4)

# input('Press key to start training')

batch_size = 50
all_obs = torch.randn((batch_size, obs_dim))#*5**2 + 30
des_acts = torch.randn((batch_size, action_dim))
des_subacts = torch.randn((batch_size, action_dim, n_intentions))

# o = nn_pol(a, _val_idxs=[0], deterministic=True)
# error = torch.sum(b0 - o[0][0])

loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4
optimizer_pol = torch.optim.Adam(nn_pol.parameters(), lr=learning_rate)

# # y0 = b0
# for tt in range(100000):
#     # loss = loss_fn(y_pred[0], y0) + loss_fn(y_pred[1], y1)
#     loss = 0
#     for aa in range(n_policies):
#         a_preds = nn_pol(all_obs, pol_idx=aa, deterministic=True)[0]
#         loss += loss_fn(des_subacts[:, :, aa], a_preds)
#     a_preds = nn_pol(all_obs, pol_idx=None, deterministic=True)[0]
#     loss += loss_fn(des_acts, a_preds)
#     print(tt, loss.item())
#
#     optimizer_pol.zero_grad()
#     loss.backward()
#     optimizer_pol.step()
# print('='*10)
# print('='*10)
# input('wuuu')
obs = env.reset()
start = time.time()
for tt in range(T):
    act, agent_info = nn_pol.get_action(obs, pol_idx=None, deterministic=True)
    next_obs, reward, done, env_info = env.step(act)
    replay_buffer.add_sample(
        observation=obs,
        action=act,
        reward=reward,
        terminal=done,
        next_observation=next_obs,
        agent_info=agent_info,
        env_info=env_info,
    )
    obs = next_obs
    if tt > BATCH_SIZE:
        batch = replay_buffer.random_batch(BATCH_SIZE)

        for uu in range(n_intentions):
            # Get batch rewards and terminal for unintentional tasks
            rewards = Variable(ptu.from_numpy(batch['reward_vectors'][:, uu]).float(), requires_grad=False)
            rewards = rewards.unsqueeze(-1)
            terminals = Variable(ptu.from_numpy(batch['terminal_vectors'][:, uu]).float(), requires_grad=False)
            terminals = terminals.unsqueeze(-1)
print('&&&&\n'*6)
total_time = time.time() - start
print('TOTAL TIME:', total_time)
print('DES TIME', T*DT)
print('TOT/DES', total_time/(T*DT))
input('NADA DESPUESSS')
# print(a)
# print(o[0])
# print(b0)

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
