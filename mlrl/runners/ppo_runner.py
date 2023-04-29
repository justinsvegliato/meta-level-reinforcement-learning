from mlrl.meta.retro_rewards_rewriter import RetroactiveRewardsRewriter
from mlrl.utils.progbar_observer import ProgressBarObserver
from mlrl.utils import time_id
from mlrl.runners.eval_runner import EvalRunner
from mlrl.meta.meta_policies.search_ppo_agent import create_search_ppo_agent

import cProfile
import gc
from typing import Optional, Callable
import math
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
from tf_agents.train.utils import train_utils
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment

import wandb


class PPORunner:
    """
    Class for running PPO experiments.
    Handles collecting data, training, and evaluation.
    Creates a new directory for each run and syncs it with wandb.
    """

    def __init__(self,
                 collect_env: BatchedPyEnvironment,
                 eval_env: Optional[BatchedPyEnvironment] = None,
                 video_env: Optional[BatchedPyEnvironment] = None,
                 train_batch_size: int = 4,
                 summary_interval: int = 1000,
                 collect_steps: int = 4096,
                 policy_save_interval: int = 1,
                 eval_steps: int = 1000,
                 eval_interval: int = 5,
                 video_steps: int = 60,
                 num_iterations: int = 1000,
                 run_name: str = None,
                 rewrite_rewards: bool = False,
                 profile_run: bool = False,
                 gc_interval: int = 5,
                 model_save_metric: str = 'AverageReturn',
                 model_save_metric_comparator: str = 'max',
                 end_of_epoch_callback: Callable[[dict, 'PPORunner'], None] = None,
                 **config):
        self.eval_interval = eval_interval
        self.num_iterations = num_iterations
        self.eval_steps = eval_steps
        self.n_collect_envs = collect_env.batch_size or 1
        self.policy_save_interval = policy_save_interval
        self.collect_steps = collect_steps
        self.summary_interval = summary_interval
        self.train_num_steps = math.ceil(collect_steps / self.n_collect_envs)
        self.train_batch_size = train_batch_size

        self.profile_run = profile_run
        self.config = config
        self.name = run_name or f'ppo_run_{time_id()}'
        self.root_dir = f'outputs/runs/{self.name}/'
        self.gc_interval = gc_interval

        self.collect_env = collect_env
        self.eval_env = eval_env

        self.collect_metrics = []
        self.end_of_epoch_callback = end_of_epoch_callback

        self.video_env = video_env
        self.videos_dir = self.root_dir + '/videos'
        self.video_steps = video_steps
        Path(self.videos_dir).mkdir(parents=True, exist_ok=True)

        self.model_save_metric = model_save_metric
        self.model_save_comparator = max if model_save_metric_comparator.lower() in ['max', 'greater', '>'] else min
        self.model_save_metric_best = float('-inf') if self.model_save_comparator is max else float('inf')

        self.train_step_counter = train_utils.create_train_step()

        start_lr = config.get('learning_rate', 3e-4)

        def learning_rate_fn():
            # Linearly decay the learning rate.
            p = self.train_step_counter / self.num_iterations
            return start_lr * (1 - p)

        self.learning_rate = learning_rate_fn
        config['learning_rate'] = self.learning_rate

        self.agent = create_search_ppo_agent(self.collect_env, config, self.train_step_counter)

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.collect_env.batch_size or 1,
            max_length=self.train_num_steps
        )

        def preprocess_seq(experience, info):
            return self.agent.preprocess_sequence(experience), info

        def dataset_fn():
            ds = self.replay_buffer.as_dataset(sample_batch_size=self.train_batch_size,
                                               num_steps=self.train_num_steps)
            return ds.map(preprocess_seq).prefetch(5)

        self.saved_model_dir = os.path.join(self.root_dir, learner.POLICY_SAVED_MODEL_DIR, 'ckpt')
        self.best_model_dir = os.path.join(self.root_dir, learner.POLICY_SAVED_MODEL_DIR, 'best')
        collect_env_step_metric = py_metrics.EnvironmentSteps()

        best_model_trigger = triggers.PolicySavedModelTrigger(
            self.best_model_dir,
            self.agent,
            self.train_step_counter,
            interval=1,
            metadata_metrics={
                triggers.ENV_STEP_METADATA_KEY: collect_env_step_metric
            }
        )

        self.save_best_model = best_model_trigger._save_fn

        ckpt_trigger = triggers.PolicySavedModelTrigger(
            self.saved_model_dir,
            self.agent,
            self.train_step_counter,
            interval=policy_save_interval,
            metadata_metrics={
                triggers.ENV_STEP_METADATA_KEY: collect_env_step_metric
            }
        )

        self.save_checkpoint = ckpt_trigger._save_fn

        learning_triggers = [
            ckpt_trigger,
            triggers.StepPerSecondLogTrigger(self.train_step_counter,
                                             interval=summary_interval),
        ]

        self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            self.agent.collect_policy, use_tf_function=True, batch_time_steps=False)

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

        collect_observers.append(ProgressBarObserver(
            collect_steps,
            metrics=[m for m in self.collect_metrics if 'return' in m.name.lower()],
            update_interval=1
        ))

        self.collect_actor = actor.Actor(
            self.collect_env,
            self.collect_policy,
            self.train_step_counter,
            steps_per_run=collect_steps,
            observers=collect_observers,
            metrics=actor.collect_metrics(buffer_size=collect_steps),
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
            num_samples=5 * max(1, self.collect_steps // (self.train_batch_size * self.train_num_steps)),
            triggers=learning_triggers,
            shuffle_buffer_size=collect_steps
        )

        # set up evaluation runner
        if eval_steps > 0 and self.eval_env is not None:
            self.evaluator = EvalRunner(
                eval_steps,
                eval_env,
                self.agent.policy,
                video_env=video_env,
                videos_dir=self.videos_dir,
                rewrite_rewards=rewrite_rewards,
                step_counter=self.train_step_counter)
        else:
            self.evaluator = None

    def get_config(self):
        """ Returns the config of the runner. """
        return {
            'name': self.name,
            'eval_interval': self.eval_interval,
            'num_iterations': self.num_iterations,
            'eval_steps': self.eval_steps,
            'policy_save_interval': self.policy_save_interval,
            'collect_steps': self.collect_steps,
            'summary_interval': self.summary_interval,
            'train_num_steps': self.train_num_steps,
            'train_batch_size': self.train_batch_size,
            'env_batch_size': self.n_collect_envs,
            'num_learn_samples': self.ppo_learner._num_samples,
            'root_dir': self.root_dir,
            **self.config
        }

    def run_evaluation(self, i: int):
        logs = {}

        if self.evaluator is not None:
            eval_logs = self.evaluator.run()
            logs.update(eval_logs)

        try:
            if self.video_env is not None:
                video_file = self.evaluator.create_policy_eval_video(self.video_steps,
                                                                     f'video_{i}')
                logs['video'] = wandb.Video(video_file, fps=30, format="mp4")

        except Exception as e:
            print(f'Error creating video: {e}')

        return logs

    def collect(self):
        # its very important to reset the actors
        # otherwise the observations can be wrong on the next run
        self.collect_actor.reset()
        for metric in self.collect_metrics:
            metric.reset()

        self.replay_buffer.clear()

        logs = {}

        start_time = time.time()
        self.collect_actor.run()
        end_time = time.time()

        collect_metrics = {
            metric.name: metric.result() for metric in self.collect_metrics
        }
        collect_metrics['CollectTime'] = end_time - start_time
        logs.update(collect_metrics)

        print('Collect stats:')
        print(', '.join([
            f'{name}: {value:.3f}' for name, value in collect_metrics.items()
            if isinstance(value, float)
        ]))

        if self.rewrite_rewards:
            self.reward_rewriter.flush_all()

        return logs

    def train(self):
        logs = {}

        logs['TrainingSteps'] = self.train_step_counter.numpy()
        self.train_step_counter.assign_add(1)

        if isinstance(self.learning_rate, type(callable)):
            logs['LearningRate'] = self.learning_rate

        start_time = time.time()
        loss_info = self.ppo_learner.run()
        end_time = time.time()
        print('Training info:')
        print(f'Loss: {loss_info.loss or 0:.5f}, '
              f'KL Penalty Loss: {loss_info.extra.kl_penalty_loss or 0:.5f}, '
              f'Entropy: {loss_info.extra.entropy_regularization_loss or 0:.5f}, '
              f'Value Estimation Loss: {loss_info.extra.value_estimation_loss or 0:.5f}, '
              f'PG Loss {loss_info.extra.policy_gradient_loss or 0:.5f}, '
              f'Train Time: {end_time - start_time:.1f} (s)')
        print()

        logs.update({
            'loss': loss_info.loss.numpy(),
            **tf.nest.map_structure(lambda x: x.numpy(), loss_info.extra._asdict())
        })
        logs['TrainTime'] = end_time - start_time

        return logs

    def save_best(self):
        print(f'Saving new best model with '
              f'{self.model_save_metric} = {self.model_save_metric_best or 0:.3f}')
        self.save_best_model()

    def _run(self):
        wandb.init(project='mlrl', entity='drcope',
                   reinit=True, config=self.get_config())

        for i in range(self.num_iterations):
            iteration_logs = {'iteration': i}
            print(f'Iteration: {i}')

            iteration_logs['TrainStep'] = self.train_step_counter.numpy()

            if self.eval_interval > 0 and i % self.eval_interval == 0:
                eval_logs = self.run_evaluation(i)
                iteration_logs.update(eval_logs)

            iteration_logs.update(self.collect())
            iteration_logs.update(self.train())

            if self.model_save_metric in iteration_logs:
                val = iteration_logs[self.model_save_metric]
                new_val = self.model_save_comparator(val, self.model_save_metric_best)
                if new_val != self.model_save_metric_best:
                    self.model_save_metric_best = new_val
                    self.save_best()

            if self.end_of_epoch_callback is not None:
                self.end_of_epoch_callback(iteration_logs, self)

            wandb.log(iteration_logs)

            if (self.gc_interval > 0 and i % self.gc_interval == 0) or \
                    iteration_logs['CollectTime'] + iteration_logs['TrainTime'] > 180:
                gc.collect()

    def run(self):
        """
        Performs iterations of the PPO algorithm collect and train loop.
        """
        try:
            if self.profile_run:
                pr = cProfile.Profile()
                pr.enable()
                with tf.profiler.experimental.Profile(self.root_dir + '/profile'):
                    self._run()
            else:
                self._run()

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
            self.save_checkpoint()
            wandb.finish()
            if self.profile_run:
                pr.disable()
                pr.dump_stats(self.root_dir + '/profile/profile.prof')
                print('Profile saved to: ' + self.root_dir + '/profile/profile.prof')
