from robolearn.utils.samplers import rollout
from robolearn.torch.core import PyTorchModule
from robolearn.torch.pytorch_util import set_gpu_mode
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn_gym_envs.pybullet import Reacher2D3DofBulletEnv
from robolearn.policies import MakeDeterministic
from robolearn.policies import ExplorationPolicy
import argparse
import joblib
import uuid
from robolearn.core import logger
import json

filename = str(uuid.uuid4())


def simulate_policy(args):
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

    # Load environment
    with open('variant.json') as json_data:
        env_params = json.load(json_data)['env_params']
    env_params['is_render'] = True
    env = NormalizedBoxEnv(
        Reacher2D3DofBulletEnv(**env_params)
    )
    print("Environment loaded!!")

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        if args.record:
            env.start_recording_video('interaction_video.mp4')
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=True,
            # deterministic=args.deterministic,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()
        if args.record:
            env.stop_recording_video()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./progress.csv',
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
