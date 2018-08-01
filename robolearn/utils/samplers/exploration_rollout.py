import numpy as np


def exploration_rollout(env, exploration_policy, max_path_length=np.inf,
                        animated=False, deterministic=None, condition=None):
    """
    Execute a single rollout until the task finishes (environment returns done)
    or max_path_length is reached.

    Args:
        env:
        agent:
        max_path_length:
        animated:
        deterministic:

    Returns:
        Rollout dictionary (dict)

        The following value for the following keys will be a 2D array, with the
        first dimension corresponding to the time dimension.
         - observations (np.ndarray)
         - actions (np.ndarray)
         - rewards (np.ndarray)
         - next_observations (np.ndarray)
         - terminals (np.ndarray)

        The next two elements will be lists of dictionaries, with the index into
        the list being the index into the time
         - agent_infos (list)
         - env_infos

    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []

    if condition is None:
        exploration_policy.reset()
        obs = env.reset()
    else:
        exploration_policy.reset(condition)
        obs = env.reset(condition)

    next_obs = None
    path_length = 0

    if animated:
        env.render()

    while path_length < max_path_length:
        if deterministic is None:
            a, agent_info = exploration_policy.get_action(obs)
        else:
            a, agent_info = \
                exploration_policy.get_action(obs, deterministic=deterministic)
        next_obs, reward, done, env_info = env.step(a)

        observations.append(obs)
        rewards.append(reward)
        terminals.append(done)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if done:
            break
        obs = next_obs
        if animated:
            env.render()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)

    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_obs = np.array([next_obs])

    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_obs, 0)
        )
    )

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
