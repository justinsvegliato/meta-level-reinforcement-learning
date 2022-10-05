from mlrl.experiments.experiment_utils import parse_args, create_meta_env
from mlrl.meta.search_tree import ObjectState
from mlrl.meta.meta_env import MetaEnv
from mlrl.maze.maze_state import RestrictedActionsMazeState
from mlrl.maze.manhattan_q import ManhattanQHat
from mlrl.maze.maze_env import make_maze_env
from mlrl.meta.meta_env import mask_token_splitter
from mlrl.meta.search_networks import create_action_distribution_network
from mlrl.meta.search_networks import create_value_network
from mlrl.utils.render_utils import create_policy_eval_video

import os
from pathlib import Path
import time
from typing import Type

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
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.train.utils import spec_utils

import wandb


def get_maze_name(config: dict) -> str:
    name = ''

    if config.get('procgen_maze', False):
        name += 'procgen_'

    n = config.get('maze_size', 5)
    agent = config.get('agent', 'ppo')
    name += f'{n}x{n}_maze_{agent}'

    return name


def create_maze_meta_env(object_state_cls: Type[ObjectState],
                         args: dict) -> MetaEnv:
    maze_n = args.get('maze_size', 5)
    procgen = bool(args.get('procgen_maze', False))

    object_env = make_maze_env(
        seed=args.get('seed', 0),
        maze_size=(maze_n, maze_n),
        goal_reward=1,
        render_shape=(64, 64),
        generate_new_maze_on_reset=procgen,
    )

    q_hat = ManhattanQHat(object_env)

    return create_meta_env(
        object_env,
        object_state_cls.extract_state(object_env),
        q_hat,
        args,
        object_action_to_string=lambda a: object_env.ACTION[a]
    )


def create_search_ppo_agent(env, config, train_step=None):

    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(env))

    network_kwargs = config.get('network_kwargs', None) or {
        'n_heads': 3,
        'n_layers': 2,
        'head_dim': 32,
    }

    value_net = create_value_network(observation_tensor_spec, **network_kwargs)

    actor_net = create_action_distribution_network(observation_tensor_spec['search_tree_tokens'],
                                                   action_tensor_spec,
                                                   **network_kwargs)

    masked_actor_net = MaskSplitterNetwork(mask_token_splitter,
                                           actor_net,
                                           input_tensor_spec=observation_tensor_spec,
                                           passthrough_mask=True)

    train_step = train_step or train_utils.create_train_step()

    return PPOAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        actor_net=masked_actor_net,
        value_net=value_net,
        optimizer=tf.keras.optimizers.Adam(3e-4),
        train_step_counter=train_step,
        compute_value_and_advantage_in_train=False,
        update_normalizers_in_train=False,
        normalize_observations=False,
        discount_factor=0.99,
        num_epochs=1,  # deprecated param
    )


def time_id():
    """ Returns an id based on the current time. """
    return int(time.time() * 1e7)


class PPORunner:

    def __init__(self,
                 env_batch_size: int = 2,
                 env_multithreading: bool = True,
                 train_batch_size: int = 2,
                 train_num_steps: int = 128,
                 summary_interval: int = 1000,
                 collect_sequence_length: int = 2048,
                 policy_save_interval: int = 10000,
                 eval_steps: int = 1000,
                 eval_interval: int = 5,
                 num_iterations: int = 1000,
                 **config):
        self.eval_interval = eval_interval
        self.num_iterations = num_iterations
        self.eval_steps = eval_steps
        self.policy_save_interval = policy_save_interval
        self.collect_sequence_length = collect_sequence_length
        self.summary_interval = summary_interval
        self.train_num_steps = train_num_steps
        self.train_batch_size = train_batch_size
        self.env_multithreading = env_multithreading
        self.env_batch_size = env_batch_size
        self.config = config
        self.name = get_maze_name(config)
        self.root_dir = f'runs/{self.name}/{time_id()}'

        self.videos_dir = self.root_dir + '/videos'
        Path(self.videos_dir).mkdir(parents=True, exist_ok=True)

        self.env = BatchedPyEnvironment([
            GymWrapper(create_maze_meta_env(RestrictedActionsMazeState, config))
            for _ in range(env_batch_size)
        ], multithreading=False)
        self.env.reset()

        if eval_steps > 0:
            self.eval_env = BatchedPyEnvironment([
                GymWrapper(create_maze_meta_env(RestrictedActionsMazeState, config))
                for _ in range(env_batch_size)
            ], multithreading=False)
            self.eval_env.reset()

        self.train_step_counter = train_utils.create_train_step()
        self.agent = create_search_ppo_agent(self.env, config, self.train_step_counter)

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env.batch_size or 1,
            max_length=10000
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

        self.collect_actor = actor.Actor(
            self.env,
            self.agent.collect_policy,
            self.train_step_counter,
            steps_per_run=collect_sequence_length,
            observers=[self.replay_buffer.add_batch],
            metrics=actor.collect_metrics(buffer_size=collect_sequence_length),
            reference_metrics=[collect_env_step_metric],
            summary_dir=os.path.join(self.root_dir, learner.TRAIN_DIR),
            summary_interval=summary_interval)

        self.ppo_learner = PPOLearner(
            self.root_dir,
            self.train_step_counter,
            self.agent,
            experience_dataset_fn=dataset_fn,
            normalization_dataset_fn=dataset_fn,
            num_samples=1, num_epochs=20,  # num samples * num epochs = train steps per run
            triggers=learning_triggers,
            shuffle_buffer_size=collect_sequence_length
        )

        if eval_steps > 0:
            self.eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
                self.agent.policy, use_tf_function=True)

            self.eval_actor = actor.Actor(
                self.eval_env,
                self.eval_greedy_policy,
                self.train_step_counter,
                metrics=actor.eval_metrics(buffer_size=10),
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

        self.eval_actor.run_and_log()
        print('Evaluation stats:')
        print(', '.join([
            f'{metric.name}: {metric.result():.3f}'
            for metric in self.eval_actor.metrics
        ]))

        logs = {
            f'Eval{metric.name}': metric.result()
            for metric in self.eval_actor.metrics
        }

        try:
            video_file = f'{self.videos_dir}/video_{i}.mp4'
            create_policy_eval_video(self.agent.policy, self.env, max_steps=120,
                                     filename=video_file, max_envs_to_show=1)
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
        self.eval_actor.reset()
        for metric in self.eval_actor.metrics:
            metric.reset()
        for metric in self.collect_actor.metrics:
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
                self.collect_actor.run()
                print('Collect stats:')
                print(', '.join([
                    f'{metric.name}: {metric.result():.3f}'
                    for metric in self.collect_actor.metrics
                ]))
                iteration_logs.update({
                    metric.name: metric.result()
                    for metric in self.collect_actor.metrics
                })

                # train
                loss_info = self.ppo_learner.run()
                print('Training info:')
                print(f'Loss: {loss_info.loss:.5f}, '
                      f'KL Penalty Loss: {loss_info.extra.kl_penalty_loss:.5f}, '
                      f'Entropy: {loss_info.extra.entropy_regularization_loss:.5f}, '
                      f'Value Estimation Loss: {loss_info.extra.value_estimation_loss:.5f}, '
                      f'PG Loss {loss_info.extra.policy_gradient_loss:.5f}')
                print()

                iteration_logs.update({
                    'loss': loss_info.loss.numpy(),
                    **tf.nest.map_structure(lambda x: x.numpy(), loss_info.extra._asdict())
                })

                wandb.log(iteration_logs)

        except KeyboardInterrupt:
            print('Training interrupted by user.')

        finally:
            wandb.finish()


def main():
    args = parse_args()
    ppo_runner = PPORunner(**args)
    ppo_runner.run()


if __name__ == "__main__":
    main()
