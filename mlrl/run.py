from .config import RUNS_DIR
from .utils.render_utils import create_policy_eval_video

import json
import time
from typing import List, Callable, Dict
from pathlib import Path
from collections import defaultdict

import tensorflow as tf
from tf_agents.agents import TFAgent
from tf_agents.policies import TFPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import numpy as np
import wandb


def compute_return_stats(environment: TFEnvironment,
                         policy: TFPolicy,
                         num_episodes: int = 3,
                         max_steps: int = 100) -> float:
    """
    Computes mean and standard deviation for returns of a policy on a given environment.
    """

    returns = []
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        n_steps = 0
        while not time_step.is_last() and n_steps < max_steps:
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            n_steps += 1

        returns.append(episode_return)

    returns = np.array(returns)

    return returns.mean(), returns.std()


def time_id():
    """ Returns an id based on the current time. """
    return int(time.time() * 1e7)


class TrainingRun:
    """
    Class to handle training and an agent
    """

    def __init__(self,
                 agent: TFAgent,
                 environment: TFEnvironment,
                 model: tf.keras.Model,
                 num_epochs=10,
                 collect_steps_per_iteration=1,
                 train_steps_per_epoch=1000,
                 replay_buffer_max_length=100000,
                 experience_batch_size=64,
                 eval_env: TFEnvironment = None,
                 num_eval_episodes=1,
                 eval_steps=250,
                 initial_collect_steps=500,
                 video_steps=60,
                 metric_fns: Dict[str, Callable[[TFAgent, TFEnvironment], float]] = None,
                 callbacks: List[tf.keras.callbacks.Callback] = None,
                 run_dir=None,
                 end_wandb=True,
                 verbose=1,
                 wandb_entity='drcope',
                 wandb_project='mlrl',
                 name='run'):
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
            replay_buffer_max_length: The maximum length of the replay buffer
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

        # Training parameters
        self.train_steps_per_epoch = train_steps_per_epoch
        self.eval_steps = eval_steps
        self.num_eval_episodes = num_eval_episodes
        self.initial_collect_steps = initial_collect_steps
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.experience_batch_size = experience_batch_size
        self.video_steps = video_steps

        self.epoch = 0
        self.eval_env = eval_env or environment

        # Data tracking

        self.metric_fns = metric_fns or dict()
        self.run_dir = run_dir or f'{RUNS_DIR}/{agent.name}/{self.name}-{self.run_id}'
        self.model_weights_dir = f'{self.run_dir}/model_weights'
        self.videos_dir = f'{self.run_dir}/videos'

        self.num_epochs = num_epochs
        self.verbose = verbose

        Path(self.run_dir).mkdir(parents=True)
        Path(self.model_weights_dir).mkdir(parents=True)
        Path(self.videos_dir).mkdir(parents=True)

        self.user_given_callbacks = callbacks or []
        self.callbacks = None  # To be defined in pre execution setup due to wandb config

        # Wandb variables
        self.end_wandb = end_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        # TF-agents variables

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=environment.batch_size,
            max_length=replay_buffer_max_length
        )

        self.experience_dataset = None  # setup in pre-execution
        self.experience_iterator = None  # setup in pre-execution

        self.random_policy = RandomTFPolicy(
            environment.time_step_spec(),
            environment.action_spec(),
            observation_and_action_constraint_splitter=agent._observation_and_action_constraint_splitter
        )

    def get_config(self):
        try:
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
        }

    def training_step(self) -> Dict[str, float]:

        logs = dict()

        # Collect a few steps using collect_policy and save to the replay buffer.
        self.collect_data(self.agent.collect_policy, self.collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, _ = next(self.experience_iterator)
        logs['train_loss'] = self.agent.train(experience).loss

        return logs

    def get_evaluation_stats(self) -> Dict[str, float]:
        logs = dict()
        return_mean, return_std = compute_return_stats(
            self.eval_env, self.agent.policy,
            self.num_eval_episodes, self.eval_steps
        )

        logs['eval_return_mean'] = return_mean
        logs['eval_return_std'] = return_std

        logs.update({
            metric: fn(self.agent, self.eval_env)
            for metric, fn in self.metric_fns.items()
        })

        return logs

    def collect_step(self, policy: TFPolicy):
        time_step = self.environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = self.environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)

    def collect_data(self, policy: TFPolicy, steps: int):
        for _ in range(steps):
            self.collect_step(policy)

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

            self.environment.reset()
            step_logs = []
            for step in range(self.train_steps_per_epoch):
                self.callbacks.on_train_batch_begin(step)
                logs = self.training_step()
                step_logs.append(logs)
                self.callbacks.on_train_batch_end(step, logs)

            epoch_logs = self.get_evaluation_stats()
            epoch_logs.update({
                f'mean_{k}': np.mean([log[k] for log in step_logs])
                for k in step_logs[0].keys()
            })
            self.create_evaluation_video()

            self.callbacks.on_epoch_end(self.epoch, epoch_logs)
            self.epoch += 1

        self.callbacks.on_train_end()

    def create_evaluation_video(self):
        """
        Creates a video of the agent's policy in action. Saves the video to the
        videos directory and uploads it to wandb if wandb is enabled.
        """
        try:
            video_file = f'{self.videos_dir}/eval_video_{self.epoch}.mp4'
            create_policy_eval_video(
                self.agent.policy, self.eval_env,
                max_steps=self.video_steps, filename=video_file
            )
            wandb.log({f'eval_video_{self.epoch}': wandb.Video(video_file, format='mp4')})

        except Exception as e:
            print('Failed to create evaluation video:', e)

    def _setup_callbacks(self):
        self.wandb_run = wandb.init(project=self.wandb_project,
                                    entity=self.wandb_entity,
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
            model=self.model,
            verbose=self.verbose,
            epochs=self.num_epochs,
            steps=self.train_steps_per_epoch
        )

    def _pre_execute_setup(self):
        self._setup_callbacks()

        self.agent.train = common.function(self.agent.train)

        # Reset the train step
        self.agent.train_step_counter.assign(0)

        self.collect_data(self.random_policy, self.initial_collect_steps)

        self.experience_dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.experience_batch_size,
            num_steps=2
        ).prefetch(3)

        self.experience_iterator = iter(self.experience_dataset)

        self.epoch = 0

    def execute(self):
        self._pre_execute_setup()

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
        return self.model.history.history

    def save(self):

        config = {
            'run_config': self.get_config(),
            'history': self._clean_for_json(self.get_history())
        }

        with open(f'{self.run_dir}/config.json', mode='w') as f:
            print(f'Saving run config to {f}')
            json.dump(config, f, indent=4, separators=(", ", ": "), sort_keys=True)

        print(f'Saving model weights to {self.run_dir}/model')
        self.model.save_weights(f'{self.run_dir}/model')

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
            return [TrainingRun._clean_for_json(x) for x in item]
        elif type(item) in [np.float32, np.float32]:
            return float(item)
        elif type(item) in [np.int32, np.int64]:
            return int(item)
        elif isinstance(item, tuple):
            return tuple([TrainingRun._clean_for_json(x) for x in item])
        elif type(item) in [dict, defaultdict]:
            return {
                TrainingRun._clean_for_json(k): TrainingRun._clean_for_json(v)
                for k, v in item.items()
            }

        try:
            return str(item)
        except Exception:
            raise ValueError(f'Unexpected item type in history: {item=}')
