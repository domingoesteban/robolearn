import numpy as np

from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from rllab.misc.instrument import run_experiment_lite
from sac.algos import SAC
# from sac.envs import Navigation2dGoalCompoEnv
from robolearn.envs.simple_envs.multigoal_env import MultiCompositionEnv
from sac.misc.plotter import QFPolicyPlotter
from sac.misc.utils import timestamp
from sac.policies import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction


def run(*_):
    normalize = NormalizedBoxEnv
    env = normalize(MultiCompositionEnv(
        actuation_cost_coeff=1,
        distance_cost_coeff=0.1,
        goal_reward=1,
        init_sigma=0.1,
    ))

    pool = SimpleReplayBuffer(
        max_replay_buffer_size=1e6,
        env_spec=env.spec,
    )

    base_kwargs = dict(
        min_pool_size=30,
        epoch_length=1000,
        n_epochs=1000,
        max_path_length=30,
        batch_size=64,
        n_train_repeat=1,
        eval_render=True,
        eval_n_episodes=10,
        eval_deterministic=True
    )

    M = 100
    qf = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M]
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M]
    )

    policy = GMMPolicy(
        env_spec=env.spec,
        K=4,
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001
    )

    plotter = QFPolicyPlotter(
        qf=qf,
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0],
                          [0.0, 0.0],
                          [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100
    )

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,
        plotter=plotter,

        lr=3E-4,
        scale_reward=3,
        discount=0.99,
        tau=0.001,

        save_full_state=True
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment_lite(
        run,
        exp_prefix='multigoal',
        exp_name=timestamp(),
        snapshot_mode='last',
        n_parallel=1,
        seed=1,
        mode='local',
    )
