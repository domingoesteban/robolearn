from robolearn.utils.samplers import rollout
from robolearn.torch.core import PyTorchModule
from robolearn.torch.utils.pytorch_util import set_gpu_mode
from robolearn.envs.normalized_box_env import NormalizedBoxEnv

from robolearn.envs.simple_envs.navigation2d.navigation2d_goalcompo_env import Navigation2dGoalCompoEnv
from robolearn.torch.policies import WeightedMultiPolicySelector
from robolearn.models.policies import MakeDeterministic
import argparse
import joblib
import uuid
from robolearn.utils.logging import logger
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
                    WeightedMultiPolicySelector(data['policy'], args.un))
            else:
                policy = MakeDeterministic(
                    WeightedMultiPolicySelector(data['policy'], args.un))
        else:
            print('Using the deterministic version of the Intentional policy.')
            policy = MakeDeterministic(data['policy'])
    else:
        if args.un > -1:
            print('Using the UNintentional stochastic policy %02d' % args.un)
            if 'u_policy' in data:
                # policy = MultiPolicySelector(data['u_policy'], args.un)
                policy = WeightedMultiPolicySelector(data['policy'], args.un)
            else:
                # policy = data['u_policies'][args.un]
                policy = WeightedMultiPolicySelector(data['policy'], args.un)
        else:
            print('Using the Intentional stochastic policy.')
            # policy = data['exploration_policy']
            policy = data['policy']

    print("Policy loaded!!")

    # Load environment
    with open('variant.json') as json_data:
        env_params = json.load(json_data)['env_params']
    env = NormalizedBoxEnv(
        Navigation2dGoalCompoEnv(**env_params)
    )
    print("Environment loaded!!")

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./progress.csv',
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=50,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--deterministic', action="store_true")
    parser.add_argument('--un', type=int, default=-1,
                        help='Unintentional id')
    args = parser.parse_args()

    simulate_policy(args)
    input('Press a key to finish the script')
