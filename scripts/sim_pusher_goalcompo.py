from robolearn.utils.samplers import rollout
from robolearn.torch.core import PyTorchModule
from robolearn.torch.utils.pytorch_util import set_gpu_mode
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv
from robolearn.torch.policies import MultiPolicySelector
from robolearn.torch.policies import WeightedMultiPolicySelector
from robolearn.torch.policies import TanhGaussianPolicy
from robolearn.models.policies import MakeDeterministic
from robolearn.models.policies import ExplorationPolicy
import os
import argparse
import joblib
import uuid
from robolearn.utils.logging import logger
import json
import numpy as np
import robolearn.torch.utils.pytorch_util as ptu

filename = str(uuid.uuid4())
SEED = 110


def simulate_policy(args):

    np.random.seed(SEED)
    ptu.seed(SEED)

    data = joblib.load(args.file)
    if args.deterministic:
        if args.un > -1:
            print('Using the deterministic version of the UNintentional policy '
                  '%02d.' % args.un)
            if 'u_policy' in data:
                policy = MakeDeterministic(
                    MultiPolicySelector(data['u_policy'], args.un))
                    # WeightedMultiPolicySelector(data['u_policy'], args.un))
            else:
                # policy = MakeDeterministic(data['u_policies'][args.un])
                if isinstance(data['policy'], TanhGaussianPolicy):
                    policy = MakeDeterministic(data['policy'])
                else:
                    policy = MakeDeterministic(
                        WeightedMultiPolicySelector(data['policy'], args.un)
                    )
        else:
            print('Using the deterministic version of the Intentional policy.')
            if isinstance(data['policy'], ExplorationPolicy):
                policy = MakeDeterministic(data['policy'])
            else:
                policy = data['policy']
    else:
        if args.un > -1:
            print('Using the UNintentional stochastic policy %02d' % args.un)
            if 'u_policy' in data:
                # policy = MultiPolicySelector(data['u_policy'], args.un)
                policy = WeightedMultiPolicySelector(data['u_policy'], args.un)
            else:
                policy = WeightedMultiPolicySelector(data['policy'], args.un)
                # policy = data['policy'][args.un]
        else:
            print('Using the Intentional stochastic policy.')
            # policy = data['exploration_policy']
            policy = data['policy']

    print("Policy loaded!!")

    # Load environment
    dirname = os.path.dirname(args.file)
    with open(os.path.join(dirname, 'variant.json')) as json_data:
        log_data = json.load(json_data)
        env_params = log_data['env_params']
        H = int(log_data['path_length'])

    env_params.pop('goal', None)
    env_params['is_render'] = True

    if args.subtask and args.un != -1:
        env_params['subtask'] = args.un

    env = NormalizedBoxEnv(
        Pusher2D3DofGoalCompoEnv(**env_params),
        # normalize_obs=True,
        normalize_obs=False,
        online_normalization=False,
        obs_mean=None,
        obs_var=None,
        obs_alpha=0.001,
    )
    print("Environment loaded!!")

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if isinstance(policy, MakeDeterministic):
        if isinstance(policy.stochastic_policy, PyTorchModule):
            policy.stochastic_policy.train(False)
    else:
        if isinstance(policy, PyTorchModule):
            policy.train(False)

    while True:
        if args.record:
            rollout_start_fcn = lambda: \
                env.start_recording_video('pusher_video.mp4')
            rollout_end_fcn = lambda: \
                env.stop_recording_video()
        else:
            rollout_start_fcn = None
            rollout_end_fcn = None

        obs_normalizer = data.get('obs_normalizer')

        if args.H != -1:
            H = args.H

        path = rollout(
            env,
            policy,
            max_path_length=H,
            animated=True,
            obs_normalizer=obs_normalizer,
            rollout_start_fcn=rollout_start_fcn,
            rollout_end_fcn=rollout_end_fcn,
        )
        # plot_rollout_reward(path)

        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])

        logger.dump_tabular()

        if args.record:
            break


def plot_rollout_reward(path):
    import matplotlib.pyplot as plt
    rewards = np.expand_dims(path['rewards'], axis=0)

    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./params.pkl',
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=-1,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--deterministic', action="store_true")
    parser.add_argument('--record', action="store_true")
    parser.add_argument('--env', type=str, default='manipulator')
    parser.add_argument('--un', type=int, default=-1,
                        help='Unintentional id')
    parser.add_argument('--subtask', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
    input('Press a key to finish the script')
