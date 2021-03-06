from robolearn.utils.samplers import rollout
from robolearn.torch.core import PyTorchModule
from robolearn.torch.utils.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from robolearn.utils.logging import logger

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = joblib.load(args.file)
    if args.deterministic:
        print('Using the deterministic version of the policy.')
        policy = data['policy']
    else:
        print('Using the stochastic policy.')
        policy = data['exploration_policy']

    print("Policy loaded")
    env = data['env']
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
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--deterministic', action="store_true")
    args = parser.parse_args()

    simulate_policy(args)
