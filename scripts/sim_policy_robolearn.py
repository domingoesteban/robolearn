import gym
from robolearn.envs.normalized_box_env import NormalizedBoxEnv

from robolearn.utils.samplers import rollout
from robolearn.torch.core import PyTorchModule
from robolearn.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from robolearn.core import logger

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = joblib.load(args.file)
    if args.deterministic:
        print('Using the deterministic version of the _i_policy.')
        policy = data['_i_policy']
    else:
        print('Using the stochastic _i_policy.')
        policy = data['exploration_policy']

    # env = data['env']
    env = NormalizedBoxEnv(gym.make(args.env))

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    # else:
    #     set_gpu_mode(False)
    #     _i_policy.cpu()

    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        if args.record:
            env.start_recording_video('prueba.mp4')
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
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--deterministic', action="store_true")
    parser.add_argument('--record', action="store_true")
    parser.add_argument('--env', type=str, default='manipulator')
    args = parser.parse_args()

    if args.env == 'cogimon':
        args.env = 'CogimonLocomotionBulletEnvRender-v0'
    elif args.env == 'manipulator':
        args.env = 'Pusher2D3DofObstacleBulletEnvRender-v0'
    elif args.env == 'pusher':
        args.env = 'Pusher2D3DofObstacleBulletEnvRender-v0'
    else:
        raise NotImplementedError

    simulate_policy(args)
    input('Press a key to finish the script')
