from robolearn.utils.samplers import rollout
from robolearn.torch.core import PyTorchModule
from robolearn.torch.utils.pytorch_util import set_gpu_mode
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn_gym_envs.pybullet import Reacher2D3DofBulletEnv
from robolearn.models.policies import MakeDeterministic
from robolearn.models.policies import ExplorationPolicy
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
        print('Using the deterministic version of the policy.')
        if isinstance(data['policy'], ExplorationPolicy):
            policy = MakeDeterministic(data['policy'])
        else:
            policy = data['policy']
    else:
        print('Using the stochastic policy.')
        policy = data['exploration_policy']

    print("Policy loaded!!")

    # Load environment
    with open('variant.json') as json_data:
        env_params = json.load(json_data)['env_params']

    env_params['is_render'] = True
    env = NormalizedBoxEnv(
        Reacher2D3DofBulletEnv(**env_params),
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
                env.start_recording_video('reacher_video.mp4')
            rollout_end_fcn = lambda: \
                env.stop_recording_video()
        else:
            rollout_start_fcn = None
            rollout_end_fcn = None

        obs_normalizer = data.get('obs_normalizer')

        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=True,
            obs_normalizer=obs_normalizer,
            rollout_start_fcn=rollout_start_fcn,
            rollout_end_fcn=rollout_end_fcn,
        )

        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])

        logger.dump_tabular()

        if args.record:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./params.pkl',
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--deterministic', action="store_true")
    parser.add_argument('--record', action="store_true")
    parser.add_argument('--env', type=str, default='manipulator')
    parser.add_argument('--un', type=int, default=-1,
                        help='Unintentional id')
    args = parser.parse_args()

    simulate_policy(args)
    input('Press a key to finish the script')
