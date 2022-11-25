from typing import Optional
from mlrl.meta.meta_env import mask_token_splitter
from mlrl.meta.retro_rewards_rewriter import RetroactiveRewardsRewriter
from mlrl.networks.search_actor_nets import create_action_distribution_network
from mlrl.networks.search_value_net import create_value_network
from mlrl.networks.search_actor_rnn import ActionSearchRNN
from mlrl.networks.search_value_rnn import ValueSearchRNN
from mlrl.utils.render_utils import create_and_save_policy_eval_video
from mlrl.utils.progbar_observer import ProgressBarObserver

import os
from pathlib import Path
import time

import silence_tensorflow.auto  # noqa
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.ppo_learner import PPOLearner
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.train.utils import train_utils
from tf_agents.networks.mask_splitter_network import MaskSplitterNetwork
from tf_agents.train.utils import spec_utils
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment

import wandb


def create_search_ppo_agent(env, config, train_step=None):

    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(env))

    # network_kwargs = config.get('network_kwargs', None) or {
    network_kwargs = {
        'n_heads': 3,
        'n_layers': 2,
        'd_model': 32,
    }

    use_lstms = config.get('n_lstm_layers', 0) > 0
    if use_lstms:
        value_net = ValueSearchRNN(observation_tensor_spec, **config)
        actor_net = ActionSearchRNN(observation_tensor_spec, **config)
    else:
        value_net = create_value_network(observation_tensor_spec, **network_kwargs)

        actor_net = create_action_distribution_network(observation_tensor_spec['search_tree_tokens'],
                                                       action_tensor_spec,
                                                       **network_kwargs)

        actor_net = MaskSplitterNetwork(mask_token_splitter,
                                        actor_net,
                                        input_tensor_spec=observation_tensor_spec,
                                        passthrough_mask=True)

    train_step = train_step or train_utils.create_train_step()

    return PPOAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        actor_net=actor_net,
        value_net=value_net,
        optimizer=tf.keras.optimizers.Adam(1e-5),
        greedy_eval=False,
        importance_ratio_clipping=0.2,
        train_step_counter=train_step,
        compute_value_and_advantage_in_train=False,
        update_normalizers_in_train=False,
        normalize_observations=False,
        use_gae=False,
        use_td_lambda_return=False,
        discount_factor=0.99,
        num_epochs=1,  # deprecated param
    )


def time_id():
    """ Returns an id based on the current time. """
    return int(time.time() * 1e7)


class PPORunner:

    def __init__(self,
                 collect_env: BatchedPyEnvironment,
                 eval_env: Optional[BatchedPyEnvironment] = None,
                 video_env: Optional[BatchedPyEnvironment] = None,
                 train_batch_size: int = 4,
                 train_num_steps: int = 64,
                 summary_interval: int = 1000,
                 collect_sequence_length: int = 4096,
                 policy_save_interval: int = 10000,
                 eval_steps: int = 1000,
                 eval_interval: int = 5,
                 n_video_steps: int = 60,
                 num_iterations: int = 1000,
                 max_envs_to_render_in_video: int = 2,
                 run_name: str = None,
                 rewrite_rewards: bool = False,
                 **config):
        self.eval_interval = eval_interval
        self.num_iterations = num_iterations
        self.eval_steps = eval_steps
        self.policy_save_interval = policy_save_interval
        self.collect_sequence_length = collect_sequence_length
        self.summary_interval = summary_interval
        self.train_num_steps = train_num_steps
        self.train_batch_size = train_batch_size
        self.env_batch_size = collect_env.batch_size
        self.config = config
        self.max_envs_to_render_in_video = max_envs_to_render_in_video
        self.name = run_name or f'ppo_run_{time_id()}'
        self.root_dir = f'runs/{self.name}/{time_id()}'

        self.collect_env = collect_env
        self.eval_env = eval_env

        self.collect_metrics = []
        self.eval_metrics = []

        self.video_env = video_env
        self.videos_dir = self.root_dir + '/videos'
        self.n_video_steps = n_video_steps
        Path(self.videos_dir).mkdir(parents=True, exist_ok=True)

        self.train_step_counter = train_utils.create_train_step()
        self.agent = create_search_ppo_agent(self.collect_env, config, self.train_step_counter)

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.collect_env.batch_size or 1,
            max_length=16384
        )

        def preprocess_seq(experience, info):
            return self.agent.preprocess_sequence(experience), info

        def dataset_fn():
            ds = self.replay_buffer.as_dataset(sample_batch_size=train_batch_size,
                                               num_steps=train_num_steps)
            return ds.map(preprocess_seq).prefetch(5)

        self.saved_model_dir = os.path.join(self.root_dir, learner.POLICY_SAVED_MODEL_DIR)
        collect_env_step_metric = py_metrics.EnvironmentSteps()
        learning_triggers = [
            triggers.PolicySavedModelTrigger(
                self.saved_model_dir,
                self.agent,
                self.train_step_counter,
                interval=policy_save_interval,
                metadata_metrics={
                    triggers.ENV_STEP_METADATA_KEY: collect_env_step_metric
                }),
            triggers.StepPerSecondLogTrigger(self.train_step_counter,
                                             interval=summary_interval),
        ]

        self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            self.agent.collect_policy, use_tf_function=True, batch_time_steps=False)

        metrics = actor.collect_metrics(buffer_size=collect_sequence_length)

        self.rewrite_rewards = rewrite_rewards
        collect_observers = []
        if rewrite_rewards:
            self.reward_rewriter = RetroactiveRewardsRewriter(self.collect_env,
                                                              self.replay_buffer.add_batch)
            collect_observers.append(self.reward_rewriter)
            self.collect_metrics.extend(self.reward_rewriter.get_metrics())
        else:
            self.reward_rewriter = None
            collect_observers.append(self.replay_buffer.add_batch)

        collect_observers.append(ProgressBarObserver(collect_sequence_length))

        self.collect_actor = actor.Actor(
            self.collect_env,
            self.collect_policy,
            self.train_step_counter,
            steps_per_run=collect_sequence_length,
            observers=collect_observers,
            metrics=metrics,
            reference_metrics=[collect_env_step_metric],
            summary_dir=os.path.join(self.root_dir, learner.TRAIN_DIR),
            summary_interval=summary_interval)

        self.collect_metrics.extend(self.collect_actor.metrics)

        self.ppo_learner = PPOLearner(
            self.root_dir,
            self.train_step_counter,
            self.agent,
            experience_dataset_fn=dataset_fn,
            normalization_dataset_fn=dataset_fn,
            num_samples=1, num_epochs=10,  # num samples * num epochs = train steps per run
            triggers=learning_triggers,
            shuffle_buffer_size=collect_sequence_length
        )

        if eval_steps > 0:
            self.eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
                self.agent.policy, use_tf_function=True, batch_time_steps=False)

            eval_metrics = actor.collect_metrics(buffer_size=eval_steps)
            self.eval_metrics.extend(eval_metrics)

            eval_observers = [ProgressBarObserver(eval_steps)]
            if rewrite_rewards:
                self.eval_reward_rewriter = RetroactiveRewardsRewriter(self.eval_env,
                                                                       lambda _: None)
                eval_observers.append(self.eval_reward_rewriter)
                self.eval_metrics.extend(self.eval_reward_rewriter.get_metrics())

            self.eval_actor = actor.Actor(
                self.eval_env,
                self.eval_policy,
                self.train_step_counter,
                metrics=eval_metrics,
                observers=eval_observers,
                reference_metrics=[collect_env_step_metric],
                summary_dir=os.path.join(self.root_dir, 'eval'),
                steps_per_run=eval_steps)

    def get_config(self):
        """ Returns the config of the runner. """
        return {
            'name': self.name,
            'eval_interval': self.eval_interval,
            'num_iterations': self.num_iterations,
            'eval_steps': self.eval_steps,
            'policy_save_interval': self.policy_save_interval,
            'collect_sequence_length': self.collect_sequence_length,
            'summary_interval': self.summary_interval,
            'train_num_steps': self.train_num_steps,
            'train_batch_size': self.train_batch_size,
            'env_batch_size': self.env_batch_size,
            'num_samples': self.ppo_learner._num_samples,
            'num_epochs': self.ppo_learner._num_epochs,
            **self.config
        }

    def run_evaluation(self, i: int):

        self.eval_actor.reset()
        for metric in self.eval_metrics:
            metric.reset()

        start_time = time.time()
        self.eval_actor.run()
        end_time = time.time()
        logs = {
            f'Eval{metric.name}': metric.result() for metric in self.eval_metrics
        }
        logs['EvalTime'] = end_time - start_time
        print('Evaluation stats:')
        print(', '.join([
            f'{name}: {value:.3f}' for name, value in logs.items()
        ]))

        if self.reward_rewriter:
            self.eval_reward_rewriter.reset()

        try:
            video_file = f'{self.videos_dir}/video_{i}.mp4'

            create_and_save_policy_eval_video(
                self.agent.policy, self.video_env, max_steps=self.n_video_steps,
                filename=video_file,
                max_envs_to_show=self.max_envs_to_render_in_video)

            logs['video'] = wandb.Video(video_file, fps=30, format="mp4")

        except Exception as e:
            print(f'Error creating video: {e}')

        return logs

    def pre_iteration(self):
        """
        Called before each iteration.
        Resets actors and clears replay buffer.
        """
        # its very important to reset the actors
        # otherwise the observations can be wrong on the next run
        self.collect_actor.reset()
        for metric in self.collect_metrics:
            metric.reset()

        self.replay_buffer.clear()

    def run(self):
        """
        Performs iterations of the PPO algorithm collect and train loop.
        """
        try:
            wandb.init(project='mlrl', entity='drcope',
                       reinit=True, config=self.get_config())

            for i in range(self.num_iterations):
                iteration_logs = {'iteration': i}

                print(f'Iteration: {i}')
                self.pre_iteration()

                if self.eval_interval > 0 and i % self.eval_interval == 0:
                    eval_logs = self.run_evaluation(i)
                    iteration_logs.update(eval_logs)

                # collect data
                start_time = time.time()
                self.collect_actor.run()
                end_time = time.time()

                collect_metrics = {
                    metric.name: metric.result() for metric in self.collect_metrics
                }
                collect_metrics['CollectTime'] = end_time - start_time
                iteration_logs.update(collect_metrics)
                print('Collect stats:')
                print(', '.join([
                    f'{name}: {value:.3f}' for name, value in collect_metrics.items()
                ]))

                if self.rewrite_rewards:
                    self.reward_rewriter.flush_all()

                # train
                start_time = time.time()
                loss_info = self.ppo_learner.run()
                end_time = time.time()
                print('Training info:')
                print(f'Loss: {loss_info.loss:.5f}, '
                      f'KL Penalty Loss: {loss_info.extra.kl_penalty_loss:.5f}, '
                      f'Entropy: {loss_info.extra.entropy_regularization_loss:.5f}, '
                      f'Value Estimation Loss: {loss_info.extra.value_estimation_loss:.5f}, '
                      f'PG Loss {loss_info.extra.policy_gradient_loss:.5f}, '
                      f'Train Time: {end_time - start_time:.1f} (s)')
                print()

                iteration_logs.update({
                    'loss': loss_info.loss.numpy(),
                    **tf.nest.map_structure(lambda x: x.numpy(), loss_info.extra._asdict())
                })
                iteration_logs['TrainTime'] = end_time - start_time

                wandb.log(iteration_logs)

        except KeyboardInterrupt:
            print('Training interrupted by user.')

        except Exception as e:
            print()
            print(f'Error during training: {e}')
            import debugpy
            debugpy.listen(('0.0.0.0', 5678))
            print('Waiting for debugger...')
            debugpy.wait_for_client()
            print('Debugger attached.')
            debugpy.breakpoint()
            raise e

        finally:
            wandb.finish()
