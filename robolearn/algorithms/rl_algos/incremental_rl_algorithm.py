import numpy as np
import gtimer as gt

from robolearn.algorithms.rl_algos.rl_algorithm import RLAlgorithm
from robolearn.utils.logging import logger


class IncrementalRLAlgorithm(RLAlgorithm):
    def __init__(self, *args, **kwargs):
        """
        Base class for Incremental RL Algorithms
        """
        RLAlgorithm.__init__(self, *args, **kwargs)

    def train(self, start_epoch=0):
        self.training_mode(False)

        # Get snapshot of initial stuff
        if start_epoch == 0:
            self._generate_initial_policy_data()
        else:
            self._print_log_header = False

        self._n_env_steps_total = start_epoch * self.num_train_steps_per_epoch

        gt.reset()
        gt.set_def_unique(False)

        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            epoch_steps = self.num_train_steps_per_epoch
            if epoch == 0:
                epoch_steps += self._min_steps_start_train

            observation = self._start_new_rollout()
            for ss in range(epoch_steps):
                # print('epoch:%02d | steps:%03d' % (epoch, ss))
                # Get policy action
                if self._obs_normalizer is None:
                    policy_input = observation
                else:
                    policy_input = self._obs_normalizer.normalize(observation)
                action, agent_info = self._get_action_and_info(
                    policy_input,
                )
                # Render if it is requested
                if self._render:
                    self.env.render()
                # Interact with environment
                next_ob, raw_reward, terminal, env_info = (
                    self.env.step(action)
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
                if terminal or (len(self._current_path) >=
                                self.max_path_length) \
                        or ss == (epoch_steps - 1):
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob

                # Incremental learning step
                gt.stamp('sample')
                self._try_to_train()
                gt.stamp('train')

            # Evaluate if requirements are met
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def _generate_initial_policy_data(self):
        self.training_mode(False)
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)

        self.evaluate(-1)
        logger.record_tabular("Number of train steps total", 0)
        logger.record_tabular("Number of env steps total", 0)
        logger.record_tabular("Number of rollouts total", 0)
        logger.record_tabular('Train Time (s)', 0)
        logger.record_tabular('(Previous) Eval Time (s)', 0)
        logger.record_tabular('Sample Time (s)', 0)
        logger.record_tabular('Epoch Time (s)', 0)
        logger.record_tabular('Total Train Time (s)', 0)
        logger.record_tabular("Epoch", 0)

        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=self._print_log_header)
