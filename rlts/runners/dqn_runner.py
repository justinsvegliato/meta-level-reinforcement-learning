from rlts.runners.eval_runner import EvalRunner
from ..config import RUNS_DIR
from ..utils.render_utils import save_policy_eval_video

import json
import time
from typing import List, Callable, Dict, Union
from pathlib import Path
from collections import defaultdict

import silence_tensorflow.auto  # noqa
import tensorflow as tf
from tf_agents.agents import TFAgent
from tf_agents.policies import TFPolicy, py_tf_eager_policy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import py_metrics
from tf_agents.drivers import py_driver

import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def compute_return_stats(environment: TFEnvironment,
                         policy: TFPolicy,
                         num_episodes: int = 3,
                         max_steps: int = 100) -> float:
    """
    Computes mean and standard deviation for returns of a policy on a given environment.
    """

    returns = []
    rewards = []
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        n_steps = 0
        while not tf.reduce_all(time_step.is_last()) and n_steps < max_steps:
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            rewards.append(time_step.reward.numpy())
            n_steps += 1

        returns.append(episode_return)

    returns = np.array(returns)

    return returns.mean(), returns.std(), np.concatenate(rewards)


def time_id():
    """ Returns an id based on the current time. """
    return int(time.time() * 1e7)


class DQNRun:
    """
    Class to handle training and an agent
    """

    @staticmethod
    def from_save(cls,
                  save_dir: Union[Path, str],
                  agent: TFAgent,
                  environment: TFEnvironment,
                  model: tf.keras.Model):

        save_dir = Path(save_dir)
        with open(save_dir / 'config.json', 'r') as f:
            config = json.load(f)

        return cls(**config)

    def __init__(self,
                 agent: TFAgent,
                 environment: TFEnvironment,
                 model: Union[tf.keras.Model, List[tf.keras.Model]],
                 num_epochs=10,
                 collect_steps_per_iteration=1,
                 train_steps_per_epoch=1000,
                 replay_buffer_capacity=16384,  # 100000,
                 experience_batch_size=64,
                 eval_runner: EvalRunner = None,
                 initial_collect_steps=500,
                 video_steps=60,
                 video_freq=1,
                 video_render_fn=None,
                 create_video_fn=None,
                 video_fps=30,
                 metric_fns: Dict[str, Callable[[TFAgent, TFEnvironment], float]] = None,
                 callbacks: List[tf.keras.callbacks.Callback] = None,
                 run_dir=None,
                 end_wandb=True,
                 verbose=1,
                 wandb_entity='drcope',
                 wandb_project='rlts',
                 run_args=None,
                 save_reward_distributions=False,
                 reward_dist_thresh=0.5,
                 log_reporters: callable = None,
                 name='run',
                 **additional_config):
        """
        Creates a new training run.

        Args:
            agent: The agent to train
            environment: The environment to train the agent in
            model: The model being optimised
            num_epochs: The number of epochs to train for. Each epoch is defined as a
                given number of training steps followed by an evaluation of the agent.
            collect_steps_per_iteration: The number of steps to collect data for each
                training step
            train_steps_per_epoch: The number of training steps per epoch
            replay_buffer_capacity: The maximum length of the replay buffer
            experience_batch_size: The batch size for training with experience replay
            eval_env: The environment to evaluate the agent in. If None, the training
                environment is used.
            num_eval_episodes: The number of episodes to run when collecting evaluation
                statistics.
            eval_steps: The maximum number of steps to run per episode when collecting
                evaluation statistics.
            initial_collect_steps: The number of steps to collect with a random policy
                before training begins.
            video_steps: The number of steps to run when creating a video of the agent
            metric_fns: A dictionary of metric functions to use when evaluating the agent.
                The keys are the names of the metrics and the values are the metric
                functions. The metric functions should take an agent and an environment
                and return a float.
        """

        self.run_id = time_id()
        self.name = name
        self.agent = agent
        self.environment = environment
        self.model = model
        self.additional_config = additional_config

        self.eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
            agent.policy, use_tf_function=True, batch_time_steps=False)

        self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True, batch_time_steps=False)

        # Training parameters
        self.train_steps_per_epoch = train_steps_per_epoch
        self.initial_collect_steps = initial_collect_steps
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.experience_batch_size = experience_batch_size
        self.run_args = run_args or dict()

        # Video parameters
        self.create_video_fn = create_video_fn
        self.video_steps = video_steps
        self.video_freq = video_freq
        self.video_fps = video_fps
        self.video_render_fn = video_render_fn or (lambda env: env.render())

        # Run parameters
        self.epoch = 0
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.user_given_callbacks = callbacks or []
        self.callbacks = None  # To be defined in pre execution setup due to wandb config
        self.best_return = -np.inf
        self.save_reward_distributions = save_reward_distributions
        self.reward_dist_thresh = reward_dist_thresh
        self.log_reporters = log_reporters or []

        # Data tracking
        self.eval_runner = eval_runner
        self.collect_metrics: List[py_metrics.py_metric.PyMetric] = [
            py_metrics.NumberOfEpisodes(),
            py_metrics.EnvironmentSteps(),
            py_metrics.AverageReturnMetric(buffer_size=1024),
            py_metrics.AverageEpisodeLengthMetric(buffer_size=1024)
        ]

        self.metric_fns = metric_fns or dict()
        self.run_dir = run_dir or f'{RUNS_DIR}/{agent.name}/{self.name}-{self.run_id}'
        self.model_weights_dir = f'{self.run_dir}/model_weights'
        self.videos_dir = f'{self.run_dir}/videos'
        self.figures_dir = f'{self.run_dir}/figures'

        Path(self.run_dir).mkdir(parents=True)
        Path(self.model_weights_dir).mkdir(parents=True)
        Path(self.videos_dir).mkdir(parents=True)
        Path(self.figures_dir).mkdir(parents=True)

        # Wandb variables
        self.end_wandb = end_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        # TF-agents variables

        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=environment.batch_size,
            max_length=replay_buffer_capacity
        )

        self.collect_driver = py_driver.PyDriver(
            environment, self.collect_policy,
            max_steps=collect_steps_per_iteration,
            observers=[self.replay_buffer.add_batch] + self.collect_metrics
        )

        self.experience_dataset = None  # setup in pre-execution
        self.experience_iterator = None  # setup in pre-execution

        splitter = agent.__dict__.get('_observation_and_action_constraint_splitter')
        self.random_policy = py_tf_eager_policy.PyTFEagerPolicy(RandomTFPolicy(
            environment.time_step_spec(),
            environment.action_spec(),
            observation_and_action_constraint_splitter=splitter
        ), use_tf_function=True, batch_time_steps=False)

        self.initial_collect_driver = py_driver.PyDriver(
            environment, self.random_policy,
            max_steps=initial_collect_steps,
            observers=[self.replay_buffer.add_batch]
        )

    def get_config(self):
        try:
            if isinstance(self.model, list):
                model_config = {
                    model.name: model.get_config()
                    for model in self.model
                }
            else:
                model_config = self.model.get_config()

        except NotImplementedError:
            model_config = None

        return {
            'name': self.name,
            'agent_name': self.agent.name,
            'run_dir': self.run_dir,
            'optimiser_config': self._clean_for_json(self.agent._optimizer.get_config()),
            'model_config': self._clean_for_json(model_config),
            'max_epochs': self.num_epochs,
            'metrics': list(self.metric_fns.keys()),
            'train_steps_per_epoch': self.train_steps_per_epoch,
            'initial_collect_steps': self.initial_collect_steps,
            'experience_batch_size': self.experience_batch_size,
            'collect_steps_per_iteration': self.collect_steps_per_iteration,
            'replay_buffer_capacity': self.replay_buffer_capacity,
            **self.run_args, **self.additional_config
        }

    def training_step(self) -> Dict[str, float]:

        logs = dict()

        # Collect a few steps using collect_policy and save to the replay buffer.
        self.collect_driver.run(self.environment.current_time_step())

        # Sample a batch of data from the buffer and update the agent's network.
        experience, _ = next(self.experience_iterator)
        logs['TrainLoss'] = self.agent.train(experience).loss

        for metric in self.collect_metrics:
            logs[metric.name] = metric.result()

        return logs

    def get_evaluation_stats(self) -> Dict[str, float]:

        if self.eval_runner is not None:
            print('Running evaluation...')
            return self.eval_runner.run()

        return dict()

    def train(self):
        """
        Executes a custom training loop.
        """
        self.callbacks.on_train_begin()

        if self.epoch == 0:
            # Evaluate the agent's policy once before training.
            self.callbacks.on_epoch_begin(0)
            self.callbacks.on_epoch_end(0, self.get_evaluation_stats())
            self.epoch += 1

        while self.epoch <= self.num_epochs:
            self.callbacks.on_epoch_begin(self.epoch)

            for metric in self.collect_metrics:
                metric.reset()

            epoch_logs = dict()

            for reporter in self.log_reporters:
                epoch_logs.update(reporter())

            epoch_logs['Epoch'] = self.epoch
            epoch_logs['TrainingSteps'] = self.epoch * self.train_steps_per_epoch
            epoch_logs['FramesCollected'] = self.epoch * self.train_steps_per_epoch * self.environment.batch_size

            logs = dict()
            for step in range(self.train_steps_per_epoch):
                self.callbacks.on_train_batch_begin(step)
                logs = self.training_step()
                self.callbacks.on_train_batch_end(step, logs)

            epoch_logs.update(logs)
            epoch_logs.update(self.get_evaluation_stats())

            for metric in self.collect_metrics:
                epoch_logs[metric.name] = metric.result()

            video_epoch = self.video_freq and self.epoch % self.video_freq == 0
            new_best_return = self.best_return < epoch_logs['EvalAverageReturn']

            if video_epoch or new_best_return:
                self.create_video(name='best_return' if new_best_return else None)

            if new_best_return:
                self.best_return = epoch_logs['EvalAverageReturn']
                self.save_model_weights(f'best_{self.epoch}_{self.best_return:5f}')

            self.callbacks.on_epoch_end(self.epoch, epoch_logs)
            self.epoch += 1

        self.callbacks.on_train_end()

    def create_reward_distributions(self, rewards):
        _, axs = plt.subplots(1, 2, figsize=(20, 5))
        sns.distplot(rewards, ax=axs[0], kde=False)
        rewards_lt_t = [
            r for r in rewards if r < self.reward_dist_thresh
        ]
        sns.distplot(rewards_lt_t, ax=axs[1], kde=False)

        for ax in axs:
            ax.set_yscale('log')
            ax.set_xlabel('Reward')

        axs[0].set_ylabel('Log Count')
        axs[1].set_ylabel('')
        axs[0].set_title('Full Distribution')
        axs[1].set_title(f'Distribution of rewards < {self.reward_dist_thresh}')

        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/reward_{self.epoch}_distribution.png')

    def create_video(self, policy: py_tf_eager_policy.PyTFEagerPolicy = None, name: str = None) -> dict:
        """
        Creates a video of the agent's policy in action. Saves the video to the
        videos directory and uploads it to wandb if wandb is enabled.
        """
        policy = policy or self.eval_policy

        if name is None:
            name = f'eval_video_{self.epoch}'
        else:
            name = f'eval_video_{name}_{self.epoch}'
        video_file = f'{self.videos_dir}/{name}.mp4'

        try:
            print(f'Creating video: {video_file}')

            if self.create_video_fn is not None:
                self.create_video_fn(policy, video_file)
                wandb.log({'Epoch': self.epoch, name: wandb.Video(video_file, format='mp4')})

            elif self.eval_runner is not None:
                save_policy_eval_video(
                    policy, self.eval_runner.eval_env, self.video_render_fn, video_file,
                    max_steps=self.video_steps, fps=self.video_fps
                )
                wandb.log({'Epoch': self.epoch, name: wandb.Video(video_file, format='mp4')})

        except Exception as e:
            print('Failed to create evaluation video:', e)

        return dict()

    def _setup_callbacks(self):
        self.wandb_run = wandb.init(project=self.wandb_project,
                                    entity=self.wandb_entity,
                                    dir=self.run_dir,
                                    config=self.get_config())

        callbacks = self.user_given_callbacks
        if not any(isinstance(c, wandb.keras.WandbCallback) for c in callbacks):
            callbacks.append(wandb.keras.WandbCallback(save_weights_only=True))
        if not any(isinstance(c, tf.keras.callbacks.ModelCheckpoint) for c in callbacks):
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(self.run_dir,
                                                                monitor='eval_return_mean',
                                                                save_weights_only=True))

        self.callbacks = tf.keras.callbacks.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=self.verbose != 0,
            model=self.get_callback_model(),
            verbose=self.verbose,
            epochs=self.num_epochs,
            steps=self.train_steps_per_epoch
        )

    def get_callback_model(self):
        if isinstance(self.model, list):
            return self.model[0]
        return self.model

    def _pre_execute_setup(self):
        print('Setting up run...')
        self._setup_callbacks()
        self.save_config()

        self.agent.train = common.function(self.agent.train)

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        print('Collecting initial experience...')
        self.environment.reset()
        self.initial_collect_driver.run(self.environment.current_time_step())

        self.experience_dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.experience_batch_size,
            num_steps=2
        ).prefetch(3)

        self.experience_iterator = iter(self.experience_dataset)
        self.epoch = 0

    def execute(self):
        self._pre_execute_setup()

        print('Starting training...')
        try:
            self.train()
            self.wandb_run.finish()
            print('Run Complete.')
        except KeyboardInterrupt:
            print('Run Interrupted!')

        self.save()
        return self.get_history()

    def get_history(self) -> Dict[str, list]:
        """
        Returns:
            The training history as a dictionary mapping metric
            names to lists of historical values.
        """
        return self.get_callback_model().history.history or dict()

    def save(self):
        self.create_video()
        self.save_config()
        self.save_model_weights()

    def save_config(self):
        with open(f'{self.run_dir}/config.json', mode='w') as f:
            print(f'Saving run config to {f}')
            json.dump(self.get_config(), f, indent=4, separators=(", ", ": "), sort_keys=True)

        with open(f'{self.run_dir}/history.json', mode='w') as f:
            print(f'Saving run history to {f}')
            json.dump(self._clean_for_json(self.get_history()), f, indent=4, separators=(", ", ": "), sort_keys=True)

    def save_model_weights(self, info: str = ''):
        if info:
            info = f'_{info}'

        # try:
        #     self.model.save(f'{self.model_dir}/{self.model.name}{info}')

        if isinstance(self.model, list):
            for m in enumerate(self.model):
                path = f'{self.model_weights_dir}/{m.name}{info}'
                print(f'Saving model weights to {path}')
                m.save_weights(path)
        else:
            path = f'{self.model_weights_dir}/{self.model.name}{info}'
            print(f'Saving model weights to {path}')
            self.model.save_weights(path)

    @staticmethod
    def _clean_for_json(item):
        if item is None:
            return 'N/A'
        elif type(item) in [str, int, float, bool]:
            return item
        elif isinstance(item, tf.Tensor):
            # assuming 1D tensor
            return float(item.numpy())
        elif isinstance(item, list):
            return [DQNRun._clean_for_json(x) for x in item]
        elif type(item) in [np.float32, np.float32]:
            return float(item)
        elif type(item) in [np.int32, np.int64]:
            return int(item)
        elif isinstance(item, tuple):
            return tuple([DQNRun._clean_for_json(x) for x in item])
        elif type(item) in [dict, defaultdict]:
            return {
                DQNRun._clean_for_json(k): DQNRun._clean_for_json(v)
                for k, v in item.items()
            }

        try:
            return str(item)
        except Exception:
            raise ValueError(f'Unexpected item type in history: {item=}')

    # @staticmethod
    # def load(self, path: str):
    #     with open(f'{path}/config.json', mode='r') as f:
    #         config = json.load(f)

        # run = DQNRun(**config['run_config'])
        # run.get_callback_model().history.history = config['history']
        # run.load_model_weights(path)

        # return run
