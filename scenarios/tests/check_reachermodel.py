import numpy as np
from robolearn.old_envs.pusher3dof import Pusher3DofBulletEnv
from builtins import input


print("\nCreating Environment...")

# Environment parameters
env_with_img = False
rdn_tgt_pos = False
render = True
obs_like_mjc = False
ntargets = 2
tgt_positions = [[0.1, 0.2], [-0.1, -0.2]]
tgt_types = ['CS', 'C']
sim_timestep = 0.001
frame_skip = int(0.01/sim_timestep)

env = Pusher3DofBulletEnv(render=render, obs_with_img=env_with_img,
                          obs_mjc_gym=obs_like_mjc, ntargets=ntargets,
                          rdn_tgt_pos=rdn_tgt_pos, tgt_types=tgt_types,
                          sim_timestep=sim_timestep,
                          frame_skip=frame_skip,
                          obs_distances=True,
                          half_env=True)

# env.set_tgt_cost_weights(tgt_weights)
env.set_tgt_pos(tgt_positions)


#
mean_init_cond = [20, 15, 5, 0.60, -0.10] # des joint_pos, obstacle
std_init_cond = [30, 30, 10, 0.1, 0.1]  # lim/2
init_joint_pos = [-90, 20, 2]
n_init_cond = 15
cond_mean = np.array(mean_init_cond)
cond_std = np.array(std_init_cond)

tgt_idx = [6, 7, 8]
obst_idx = [9, 10]  # Only X-Y is random
seed = 0

# Set the np seed
np.random.seed(seed)

all_init_conds = np.zeros((int(n_init_cond), 9))

all_rand_data = np.random.rand(n_init_cond, len(cond_mean))


# ADD SPECIFIC DATA:
idx_rdn_data = 5

print(all_rand_data.shape)
rand_data = all_rand_data[idx_rdn_data, :]
init_cond = cond_std*rand_data + cond_mean
joint_pos = np.deg2rad(init_cond[:3])

env_condition = np.zeros(env.obs_dim)
env_condition[:env.action_dim] = joint_pos
# env_condition[obst_idx] = init_cond[3:]

# Temporally hack for getting ee _object
env.add_custom_init_cond(env_condition)
env.reset(condition=-1)
# obs = self.env.get_observation()
des_tgt = env.get_ee_pose()
env.clear_custom_init_cond(-1)

env_condition[:3] = np.deg2rad(init_joint_pos)
env_condition[tgt_idx] = des_tgt
env_condition[obst_idx] = init_cond[3:]

# Now add the target properly
print('INIT COND', env_condition)
env.add_custom_init_cond(env_condition)

input('adsfkhk')

print("Environment:%s OK!." % type(env).__name__)


env.reset()


input('Press key to close')
