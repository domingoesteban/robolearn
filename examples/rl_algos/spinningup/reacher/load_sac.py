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

from robolearn_gym_envs.pybullet import Reacher2D3DofGoalCompoEnv

Tend = 10.0  # Seconds

SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP


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

    SEED = 660
    SUBTASK = None

    env_params = dict(
        is_render=render,
        # obs_distances=False,
        obs_distances=True,
        obs_with_img=False,
        # obs_with_ori=True,
        obs_with_ori=False,
        obs_with_goal=True,
        # obs_with_goal=False,
        # goal_pose=(0.65, 0.65),
        goal_pose=(0.65, 0.35),
        # rdn_goal_pos=True,
        rdn_goal_pos=False,
        robot_config=None,
        rdn_robot_config=True,
        goal_cost_weight=4.0e0,
        ctrl_cost_weight=5.0e-1,
        goal_tolerance=0.01,
        use_log_distances=True,
        log_alpha=1e-6,
        # max_time=PATH_LENGTH*DT,
        max_time=None,
        sim_timestep=SIM_TIMESTEP,
        frame_skip=FRAME_SKIP,
        half_env=True,
        subtask=SUBTASK,
        seed=SEED,
    )

    env = Reacher2D3DofGoalCompoEnv(**env_params)

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
