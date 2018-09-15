from robolearn.utils.samplers import rollout
from robolearn.torch.core import PyTorchModule
from robolearn.torch.pytorch_util import set_gpu_mode
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv
from robolearn.torch.policies import MultiPolicySelector
from robolearn.torch.policies import WeightedMultiPolicySelector
from robolearn.policies import MakeDeterministic
import argparse
import joblib
import uuid
from robolearn.core import logger
import json

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = joblib.load(args.file)
    if args.deterministic:
        if args.un > -1:
            print('Using the deterministic version of the UNintentional policy '
                  '%02d.' % args.un)
            if 'u_policy' in data:
                policy = MakeDeterministic(
                    # MultiPolicySelector(data['u_policy'], args.un))
                    WeightedMultiPolicySelector(data['u_policy'], args.un))
            else:
                # policy = MakeDeterministic(data['u_policies'][args.un])
                policy = MakeDeterministic(
                    WeightedMultiPolicySelector(data['policy'], args.un)
                )
        else:
            print('Using the deterministic version of the Intentional policy.')
            policy = MakeDeterministic(data['policy'])
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
    with open('variant.json') as json_data:
        env_params = json.load(json_data)['env_params']
    env_params.pop('goal')
    env_params['is_render'] = True
    env = NormalizedBoxEnv(
        Pusher2D3DofGoalCompoEnv(**env_params)
    )
    print("Environment loaded!!")

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
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