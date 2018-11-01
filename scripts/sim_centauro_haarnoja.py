import argparse

import joblib
import tensorflow as tf

from rllab.sampler.utils import rollout

from rllab.envs.normalized_env import normalize
from robolearn_gym_envs.pybullet import CentauroTrayEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--max-path-length', '-l', type=int, default=500)
    parser.add_argument('--speedup', '-s', type=float, default=1)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.add_argument('--policy_h', type=int)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    return args

def simulate_policy(args):
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            # env = data['algo'].env
        else:
            policy = data['policy']
            # env = data['env']

        SIM_TIMESTEP = 0.01
        FRAME_SKIP = 1
        DT = SIM_TIMESTEP * FRAME_SKIP
        env_params = dict(
            is_render=True,
            obs_with_img=False,
            active_joints='RA',
            control_type='tasktorque',
            # _control_type='torque',
            # _control_type='velocity',
            sim_timestep=SIM_TIMESTEP,
            frame_skip=FRAME_SKIP,
            obs_distances=False,
            balance_cost_weight=2.0,
            fall_cost_weight=2.0,
            tgt_cost_weight=2.0,
            balance_done_cost=2.0,#*PATH_LENGTH,  # TODO: dont forget same balance weight
            tgt_done_reward=2.0,
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
        )

        env = normalize(CentauroTrayEnv(**env_params))

        with policy.deterministic(args.deterministic):
            while True:
                path = rollout(env, policy,
                               max_path_length=args.max_path_length,
                               animated=True, speedup=args.speedup)
                input("Press a key to re-sample...")
if __name__ == "__main__":
    args = parse_args()
    simulate_policy(args)
