from mlrl.meta.meta_policies.a_star_policy import AStarPolicy
from mlrl.meta.meta_policies.random_policy import (
    create_random_search_policy, create_random_search_policy_no_terminate
)
from mlrl.meta.retro_rewards_rewriter import RetroactiveRewardsRewriter
from mlrl.utils import get_current_git_commit, sanitize_dict, time_id
from mlrl.utils.render_utils import create_and_save_meta_policy_video
from mlrl.utils.progbar_observer import ProgressBarObserver

import argparse
import json
from pathlib import Path
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
                 eval_env: BatchedPyEnvironment,
                 policy,
                 eval_steps: Optional[int] = None,
                 rewrite_rewards: bool = False,
                 video_policy=None,
                 video_env: Optional[BatchedPyEnvironment] = None,
                 videos_dir: str = None,
                 use_tf_function: bool = True,
                 convert_to_eager: bool = True,
                 metrics: Optional[list] = None,
                 observers: Optional[list] = None,
                 stop_eval_condition: callable = None,
                 prog_bar: Optional[ProgressBarObserver] = None,
                 step_counter=None):
        self.eval_env = eval_env
        self.video_env = video_env or eval_env
        self.videos_dir = videos_dir or '.'
        self.stop_eval_condition = stop_eval_condition
        Path(self.videos_dir).mkdir(parents=True, exist_ok=True)

        if eval_steps is None and stop_eval_condition is None:
            raise ValueError('Either eval_steps or stop_eval_condition must be specified.')

        self.eval_steps = eval_steps or 1

        if convert_to_eager:
            self.eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
                policy, use_tf_function=use_tf_function, batch_time_steps=False)
        else:
            self.eval_policy = policy

        if video_policy is None:
            self.video_policy = self.eval_policy
        elif convert_to_eager:
            self.video_policy = py_tf_eager_policy.PyTFEagerPolicy(
                video_policy, use_tf_function=use_tf_function, batch_time_steps=False)
        else:
            self.video_policy = video_policy

        self.metrics = metrics or []

        eval_observers = observers or []
        self.rewrite_rewards = rewrite_rewards
        if rewrite_rewards:
            self.eval_reward_rewriter = RetroactiveRewardsRewriter(self.eval_env,
                                                                   lambda _: None)
            eval_observers.append(self.eval_reward_rewriter)
            self.metrics.extend(self.eval_reward_rewriter.get_metrics())
        else:
            self.eval_reward_rewriter = None

        actor_collect_metrics = actor.collect_metrics(buffer_size=self.eval_steps)

        if self.eval_steps > 1:
            self.prog_bar = prog_bar or ProgressBarObserver(
                self.eval_steps,
                metrics=[m for m in actor_collect_metrics if 'return' in m.name.lower()],
                update_interval=1
            )
            eval_observers.append(self.prog_bar)
        else:
            self.prog_bar = prog_bar

        self.step_counter = step_counter or train_utils.create_train_step()
        self.eval_actor = actor.Actor(
            self.eval_env,
            self.eval_policy,
            self.step_counter,
            metrics=actor_collect_metrics,
            observers=eval_observers,
            reference_metrics=[py_metrics.EnvironmentSteps()],
            steps_per_run=self.eval_steps)

        py_metrics.AverageEpisodeLengthMetric()

        self.metrics.extend(self.eval_actor.metrics)

    def get_metrics(self):
        return self.metrics

    def run(self):
        self.eval_actor.reset()
        for metric in self.metrics:
            metric.reset()

        if self.prog_bar is not None:
            self.prog_bar.reset()

        start_time = time.time()

        try:
            if self.stop_eval_condition is not None:
                while not self.stop_eval_condition():
                    self.eval_actor.run()
            else:
                self.eval_actor.run()

        except KeyboardInterrupt:
            print('\nEvaluation interrupted.')

        end_time = time.time()

        logs = {
            f'Eval{metric.name}': metric.result()
            for metric in self.metrics
            if metric.result() is not None
        }
        logs['EvalTime'] = end_time - start_time

        print('Evaluation stats:')
        print(', '.join([
            f'{name}: {value:.3f}' for name, value in logs.items()
            if isinstance(value, float)
        ]))

        if self.eval_reward_rewriter is not None:
            self.eval_reward_rewriter.reset()

        return logs

    def create_policy_eval_video(
            self, steps: int, filename: str = 'video', **video_kwargs) -> str:
        if self.video_env is None:
            return None

        video_file = f'{self.videos_dir}/{filename}.mp4'

        create_and_save_meta_policy_video(
            self.video_policy, self.video_env,
            max_steps=steps,
            filename=video_file,
            rewrite_rewards=self.rewrite_rewards,
            **video_kwargs)

        return video_file


def run_evaluator(
        eval_policy,
        eval_env,
        eval_steps: int = 16384,
        eval_dir: str = 'outputs/eval',
        **config):

    no_tf_fn = ['random_no_terminate', 'a_star']
    runner = EvalRunner(
        policy=eval_policy,
        eval_env=eval_env,
        eval_steps=eval_steps,
        use_tf_function=config['policy'] not in no_tf_fn,
        rewrite_rewards=config.get('rewrite_rewards', True),
    )

    print('Running evaluation...')
    logs = runner.run()

    print('Evaluation stats:')
    for name, value in logs.items():
        print(f'{name}: {value:.3f}')

    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)

    stats_path = eval_path / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(sanitize_dict(logs), f, indent=4)

    config_path = eval_path / 'config.json'
    with open(config_path, 'w') as f:
        config_json = {
            'eval_steps': eval_steps,
            'git_commit': get_current_git_commit(),
            **config
        }
        json.dump(sanitize_dict(config_json), f, indent=4)


if __name__ == '__main__':
    # TODO: generalise to other environments (currently only works for maze env)
    # TODO: add support for video recording
    # TODO: add support loading trained policy from checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_steps', type=int, default=8196)
    parser.add_argument('--n_eval_envs', type=int, default=64)
    parser.add_argument('--env_multithreading', type=bool, default=True)
    parser.add_argument('--policy', type=str, default='random',
                        help='Policy to evaluate, one of: "all", "random", '
                             '"random_no_terminate", "a_star"')

    parser.add_argument('--expand_all_actions', type=bool, default=True,
                        help='Whether to expand all actions in the meta environment '
                             'with each computational action.')
    parser.add_argument('--max_tree_size', type=int, default=32,
                        help='Maximum number of nodes in the search tree.')
    parser.add_argument('--meta_discount', type=float, default=0.99,
                        help='Discount factor in meta-level environment.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--computational_rewards', type=bool, default=True,
                        help='Whether to use computational rewards.')
    parser.add_argument('--rewrite_rewards', type=bool, default=True,
                        help='Whether to rewrite computational rewards.')
    parser.add_argument('--finish_on_terminate', type=bool, default=True,
                        help='Whether to finish meta-level episode on computational terminate action.')

    config = vars(parser.parse_args())

    n_eval_envs = config.get('n_eval_envs', 64)
    env_multithreading = config.get('env_multithreading', True)

    from mlrl.experiments.maze_meta import create_batched_maze_envs
    eval_env = create_batched_maze_envs(
        n_eval_envs,
        enable_render=False,
        **config)

    make_policy_fns = {
        'random': create_random_search_policy,
        'random_no_terminate': create_random_search_policy_no_terminate,
        'a_star': AStarPolicy,
    }

    eval_policy_name = config['policy']
    if eval_policy_name == 'all':
        for policy_name in make_policy_fns:
            eval_policy = make_policy_fns[policy_name](eval_env)
            eval_dir = f'outputs/baselines/maze-gym/{policy_name}/{time_id()}'
            run_evaluator(eval_policy, eval_env, eval_dir=eval_dir, **config)

    else:
        if eval_policy_name in make_policy_fns:
            eval_policy = make_policy_fns[eval_policy_name](eval_env)
        else:
            raise ValueError(f'Unknown eval policy: {eval_policy_name}')

        eval_dir = f'outputs/baselines/maze-gym/{eval_policy_name}/{time_id()}'

        run_evaluator(eval_policy, eval_env, eval_dir=eval_dir, **config)
