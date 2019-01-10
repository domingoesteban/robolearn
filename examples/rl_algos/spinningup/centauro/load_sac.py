import time
import joblib
import os
import os.path as osp
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

from robolearn_gym_envs.pybullet import CentauroTrayEnv
from robolearn.envs.normalized_box_env import NormalizedBoxEnv

Tend = 10.0  # Seconds

SIM_TIMESTEP = 0.01
FRAME_SKIP = 1
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 1
PATHS_PER_EVAL = 2
PATHS_PER_HARD_UPDATE = 12
BATCH_SIZE = 128

SEED = 1010
# NP_THREADS = 6

SUBTASK = 1


def load_policy(fpath, itr='last', deterministic=False):
    # handle which epoch to load from
    if itr == 'last':
        saves = [int(x[11:])
                 for x in os.listdir(fpath)
                 if 'simple_save' in x and len(x) > 11]
        itr = '%d' % max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d' % itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x: \
        sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

    return get_action


def run_policy(env, policy, max_ep_len=None, num_episodes=100, render=True):

    logger = EpochLogger()
    obs, reward, done, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        action = policy(obs)
        obs, reward, done, _ = env.step(action)
        ep_ret += reward
        ep_len += 1

        if done or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            obs, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


def load_env(render=True):

    env_params = dict(
        is_render=True,
        # obs_distances=False,
        obs_distances=True,
        obs_with_img=False,
        # obs_with_ori=True,
        active_joints='RA',
        control_type='joint_tasktorque',
        # _control_type='torque',
        balance_cost_weight=1.0,
        fall_cost_weight=1.0,
        tgt_cost_weight=3.0,
        # tgt_cost_weight=50.0,
        balance_done_cost=0.,  # 2.0*PATH_LENGTH,  # TODO: dont forget same balance weight
        tgt_done_reward=0.,  # 20.0,
        ctrl_cost_weight=1.0e-1,
        use_log_distances=True,
        log_alpha_pos=1e-4,
        log_alpha_ori=1e-4,
        goal_tolerance=0.05,
        min_obj_height=0.60,
        max_obj_height=1.20,
        max_obj_distance=0.20,
        max_time=None,
        sim_timestep=SIM_TIMESTEP,
        frame_skip=FRAME_SKIP,
        subtask=SUBTASK,
        random_init=True,
        seed=SEED,
    )

    env = NormalizedBoxEnv(
        CentauroTrayEnv(**env_params),
        # normalize_obs=True,
        normalize_obs=False,
        online_normalization=False,
        obs_mean=None,
        obs_var=None,
        obs_alpha=0.001,
    )

    return env


def main(args):
    policy = load_policy(args.dir, deterministic=args.deterministic)
    env = load_env(render=not args.norender)

    run_policy(env, policy, args.horizon, args.episodes, not args.norender)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, default='.',
                        help='path to the tf directory')
    parser.add_argument('--horizon', '-H', type=int, default=1000)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--norender', '-nr', action='store_true')
    args = parser.parse_args()

    main(args)
    input('Press a key to close script')
