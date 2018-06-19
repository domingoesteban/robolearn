
import numpy as np
import gtimer as gt

from robolearn.core.rl_algorithm import RLAlgorithm
from robolearn.policies import ExplorationPolicy
from robolearn.utils.data_management import PathBuilder
from robolearn.core import logger


class IterativeRLAlgorithm(RLAlgorithm):
    def __init__(
            self,
            env,
            exploration_policy: ExplorationPolicy,
            training_env=None,
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            num_updates_per_train_call=1,
            batch_size=1024,
            min_buffer_size=None,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1e6,
            reward_scale=1,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=True,
            eval_sampler=None,
            eval_policy=None,
            replay_buffer=None,
    ):
        """
        Base class for Iterative RL Algorithms
        :param env: Environment used to evaluate.
        :param exploration_policy: Policy used to explore.
        :param training_env: Environment used by the algorithm. By default, a
        copy of `env` will be made.
        :param num_epochs: Number of episodes.
        :param num_steps_per_epoch: Number of timesteps per epoch.
        :param num_steps_per_eval: Number of timesteps per evaluation
        :param num_updates_per_train_call: Used by online training mode.
        :param num_updates_per_epoch: Used by batch training mode.
        :param max_path_length: Max length of sampled path (rollout) from env.
        :param batch_size: Replay buffer batch size.
        :param replay_buffer: External replay_buffer
        :param min_buffer_size: Min buffer size to start training.
        :param replay_buffer_size: Replay buffer size (Maximum number).
        :param discount: discount factor (gamma).
        :param reward_scale: Value to scale environment reward.
        :param render: Visualize or not the environment.
        :param save_replay_buffer: Save or not the ReplBuffer after iterations
        :param save_algorithm: Save or not the algorithm  after iterations.
        :param save_environment: Save or not the environment after interations
        :param eval_sampler: External sampler for evaluation.
        :param eval_policy: Policy to evaluate with.
        """
        RLAlgorithm.__init__(
            self,
            env,
            exploration_policy=exploration_policy,
            training_env=training_env,
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            num_updates_per_train_call=num_updates_per_train_call,
            batch_size=batch_size,
            min_buffer_size=min_buffer_size,
            max_path_length=max_path_length,
            discount=discount,
            replay_buffer_size=replay_buffer_size,
            reward_scale=reward_scale,
            render=render,
            save_replay_buffer=save_replay_buffer,
            save_algorithm=save_algorithm,
            save_environment=save_environment,
            eval_sampler=eval_sampler,
            eval_policy=eval_policy,
            replay_buffer=replay_buffer,
        )

    def train(self, start_epoch=0):
        self.pretrain()

        # Get snapshot of initial stuff
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)

        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch

        gt.reset()
        gt.set_def_unique(False)

        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            for _ in range(self.num_env_steps_per_epoch):
                # Get policy action
                action, agent_info = self._get_action_and_info(
                    observation,
                )
                # Render if it is requested
                if self._render:
                    self.training_env.render()
                # Interact with environment
                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                # Increase counter
                self._n_env_steps_total += 1
                # Create np.array of obtained terminal and reward
                reward = raw_reward * self.reward_scale
                terminal = np.array([terminal])
                reward = np.array([reward])
                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                # Check it we need to start a new rollout
                if terminal or (len(self._current_path_builder) >=
                                self.max_path_length):
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob

            # Iterative learning step
            gt.stamp('sample')
            self._try_to_train()
            gt.stamp('train')

            # Evaluate if requirements are met
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()
