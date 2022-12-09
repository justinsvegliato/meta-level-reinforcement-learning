from mlrl.meta.retro_rewards_rewriter import RetroactiveRewardsRewriter
from mlrl.utils.render_utils import create_and_save_policy_eval_video
from mlrl.utils.progbar_observer import ProgressBarObserver

from typing import Optional
import time

import silence_tensorflow.auto  # noqa
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train.utils import train_utils
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment


class EvalRunner:

    def __init__(self,
                 eval_steps: int,
                 eval_env: BatchedPyEnvironment,
                 policy,
                 rewrite_rewards: bool = False,
                 video_env: Optional[BatchedPyEnvironment] = None,
                 videos_dir: str = None,
                 use_tf_function: bool = True,
                 step_counter=None):
        self.eval_env = eval_env
        self.video_env = video_env or eval_env
        self.videos_dir = videos_dir or '.'

        self.eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=use_tf_function, batch_time_steps=False)

        self.metrics = []

        eval_observers = []
        self.rewrite_rewards = rewrite_rewards
        if rewrite_rewards:
            self.eval_reward_rewriter = RetroactiveRewardsRewriter(self.eval_env,
                                                                   lambda _: None)
            eval_observers.append(self.eval_reward_rewriter)
            self.metrics.extend(self.eval_reward_rewriter.get_metrics())
        else:
            self.eval_reward_rewriter = None

        actor_collect_metrics = actor.collect_metrics(buffer_size=eval_steps)
        self.progbar = ProgressBarObserver(
            eval_steps,
            metrics=[m for m in actor_collect_metrics if 'return' in m.name.lower()],
            update_interval=1
        )
        eval_observers.append(self.progbar)

        self.step_counter = step_counter or train_utils.create_train_step()
        self.eval_actor = actor.Actor(
            self.eval_env,
            self.eval_policy,
            self.step_counter,
            metrics=actor_collect_metrics,
            observers=eval_observers,
            reference_metrics=[py_metrics.EnvironmentSteps()],
            steps_per_run=eval_steps)

        self.metrics.extend(self.eval_actor.metrics)

    def get_metrics(self):
        return self.metrics

    def run(self):
        self.eval_actor.reset()
        for metric in self.metrics:
            metric.reset()
        self.progbar.reset()

        start_time = time.time()
        self.eval_actor.run()
        end_time = time.time()

        logs = {
            f'Eval{metric.name}': metric.result()
            for metric in self.metrics
        }
        logs['EvalTime'] = end_time - start_time

        print('Evaluation stats:')
        print(', '.join([
            f'{name}: {value:.3f}' for name, value in logs.items()
        ]))

        if self.eval_reward_rewriter is not None:
            self.eval_reward_rewriter.reset()

        return logs

    def create_policy_eval_video(self, steps: int, filename: str = 'video') -> str:
        if self.video_env is None:
            return None

        video_file = f'{self.videos_dir}/{filename}.mp4'

        create_and_save_policy_eval_video(
            self.eval_policy, self.video_env,
            max_steps=steps,
            filename=video_file,
            rewrite_rewards=self.rewrite_rewards)

        return video_file
