from robolearn.utils.samplers import rollout
from robolearn.torch.core import PyTorchModule
from robolearn.torch.pytorch_util import set_gpu_mode
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.envs.simple_envs.goal_composition_env import GoalCompositionEnv
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
            print('Using the deterministic version of the UNintentional i_policy '
                  '%02d.' % args.un)
            policy = data['u_policies'][args.un]
        else:
            print('Using the deterministic version of the Intentional i_policy.')
            policy = data['i_policy']
    else:
        if args.un > -1:
            print('Using the UNintentional stochastic i_policy %02d' % args.un)
            policy = data['u_policies'][args.un]
        else:
            print('Using the Intentional stochastic i_policy.')
            policy = data['exploration_policy']
    print("Policy loaded!!")

    # Load environment
    with open('variant.json') as json_data:
        env_params = json.load(json_data)['env_params']
    env = NormalizedBoxEnv(
        GoalCompositionEnv(**env_params)
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
