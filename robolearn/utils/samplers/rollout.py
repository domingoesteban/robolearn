import numpy as np


def rollout(env, agent, max_path_length=np.inf, animated=False,
            deterministic=None, obs_normalizer=None,
            rollout_start_fcn=None, rollout_end_fcn=None):
    """
    Execute a single rollout until the task finishes (environment returns done)
    or max_path_length is reached.

    Args:
        env: OpenAI-like environment
        agent: Policy with function get_actions(obs)
        max_path_length:
        animated (Bool): Call env.render() at each timestep or not
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

    obs = env.reset()

    if rollout_start_fcn is not None:
        rollout_start_fcn()

    next_obs = None
    path_length = 0

    if animated:
        env.render()

    while path_length < max_path_length:
        if obs_normalizer is None:
            policy_input = obs
        else:
            policy_input = obs_normalizer.normalize(obs)

        if deterministic is None:
            action, agent_info = agent.get_action(policy_input)
        else:
            action, agent_info = agent.get_action(policy_input,
                                                  deterministic=deterministic)
        next_obs, reward, done, env_info = env.step(action)

        observations.append(obs)
        rewards.append(reward)
        terminals.append(done)
        actions.append(action)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if done:
            break
        obs = next_obs
        if animated:
            env.render()

    if rollout_end_fcn is not None:
        rollout_end_fcn()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 0)

    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 0)
        next_obs = np.expand_dims(next_obs, 0)

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
