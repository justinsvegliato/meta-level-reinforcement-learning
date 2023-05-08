from mlrl.runners.eval_runner import EvalRunner
from mlrl.meta.meta_policies.a_star_policy import AStarPolicy
from mlrl.meta.meta_policies.random_policy import create_random_search_policy, create_random_search_policy_no_terminate
from mlrl.meta.meta_policies.terminator_policy import TerminatorPolicy
from mlrl.experiments.procgen_meta import create_batched_procgen_meta_envs, load_pretrained_q_network
from mlrl.experiments.procgen_meta import reset_object_level_metrics, get_object_level_metrics
from mlrl.procgen.time_limit_observer import TimeLimitObserver
from mlrl.utils import time_id
from mlrl.utils.system import restrict_gpus

from tensorflow.keras.utils import Progbar

import argparse
from typing import Dict, Callable
from pathlib import Path
import json
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def test_policies_with_pretrained_model(policy_creators: Dict[str, callable],
                                        args: dict,
                                        outputs_dir: Path,
                                        percentile=0.75,
                                        n_object_level_episodes=10,
                                        video_args: dict = None,
                                        max_object_level_steps=50,
                                        n_envs=8,
                                        run_id: str = None,
                                        results_observer: Callable[[dict], None] = None):
    create_video = video_args is not None

    args.update({
        'pretrained_percentile': percentile,
        'expand_all_actions': True,
        'finish_on_terminate': True,
    })

    if 'n_envs' in args:
        args.pop('n_envs')

    object_config = load_pretrained_q_network(
        folder=args['pretrained_runs_folder'],
        run=args['pretrained_run'],
        percentile=args.get('pretrained_percentile', 0.75),
        verbose=False
    )

    prog_bar = Progbar(
        n_object_level_episodes,
        unit_name='episode',
        stateful_metrics=['ObjectLevelMeanReward',
                          'ObjectLevelMeanStepsPerEpisode',
                          'ObjectLevelEpisodes',
                          'ObjectLevelCurrentEpisodeReturn',
                          'ObjectLevelCurrentEpisodeSteps']
    )

    def completed_n_object_level_episodes(batched_meta_env, n: int) -> bool:
        n_complete = sum([
            env.object_level_metrics.get_num_episodes()
            for env in batched_meta_env.envs
        ])

        metrics = get_object_level_metrics(batched_meta_env)
        prog_bar.update(n_complete, values=metrics.items())

        return n_complete >= n

    results = []
    for policy_name, create_policy in policy_creators.items():

        batched_meta_env = create_batched_procgen_meta_envs(
            n_envs=n_envs,
            object_config=object_config,
            **args
        )

        if create_video:
            video_env = create_batched_procgen_meta_envs(
                n_envs=2,
                object_config=object_config,
                **args
            )
        else:
            video_env = None

        eval_runner = EvalRunner(
            eval_env=batched_meta_env,
            policy=create_policy(batched_meta_env),
            rewrite_rewards=True,
            use_tf_function=False,
            convert_to_eager=False,
            stop_eval_condition=lambda: completed_n_object_level_episodes(batched_meta_env,
                                                                          n_object_level_episodes),
            video_env=video_env,
            video_policy=create_policy(video_env) if video_env is not None else None,
            videos_dir=outputs_dir / 'videos'
        )

        for env in batched_meta_env.envs:
            time_limit = TimeLimitObserver(env, max_object_level_steps)
            env.object_level_transition_observers.append(time_limit)
            env.object_level_metrics.episode_complete_callback = \
                lambda stats: results_observer.add_episode_stats(run_id, policy_name, percentile, stats)

        print(f'Evaluating {policy_name}')
        reset_object_level_metrics(batched_meta_env)
        eval_results = eval_runner.run()
        object_level_results = get_object_level_metrics(batched_meta_env)

        evaluations = {
            'Meta-level Policy': policy_name,
            'Run ID': run_id,
            **args,
            **object_config,
            **eval_results,
            **object_level_results
        }
        if results_observer is not None:
            results_observer(evaluations)

        results.append(evaluations)

        if create_video:
            eval_runner.create_policy_eval_video(filename=f'{policy_name}_{percentile=}', **video_args)

    return results


class ResultsAccumulator:

    def __init__(self, output_dir: Path):
        self.results = []
        self.episode_stats = []
        self.results_df = None
        self.episode_stats_df = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, results: dict):
        self.results.append(results)
        self.results_df = pd.DataFrame(self.results)
        self.results_df.to_csv(self.output_dir / 'results.csv', index=False)

        self.episode_stats_df = pd.DataFrame(self.episode_stats)
        self.episode_stats_df.to_csv(self.output_dir / 'episode_stats.csv', index=False)

        self.plot_results()

    def add_episode_stats(self, run_id: str, policy: str, percentile: float, stats: dict):
        self.episode_stats.append({
            'Run ID': run_id or '',
            'Meta-level Policy': policy,
            'Pretrained Percentile': percentile,
            'Number of Steps': stats['steps'],
            'Return': stats['return'],
        })
        self.episode_stats_df = pd.DataFrame(self.episode_stats)
        self.episode_stats_df.to_csv(self.output_dir / 'episode_stats.csv', index=False)

        self.plot_results()

    def plot_results(self):
        try:
            self.episode_stats_df = pd.DataFrame(self.episode_stats)
            self.results_df = pd.DataFrame(self.results)
            self.plot_rewritten_returns()
            self.plot_object_level_returns()
        except Exception as e:
            print(f'Failed to plot results: {e}')

    def plot_rewritten_returns(self):
        plot_name, plot_key = 'Mean Rewritten Meta Return', 'EvalRewrittenAverageReturn'

        if plot_key not in self.results_df.columns:
            return

        plt.figure(figsize=(15, 10))

        sns.lineplot(data=self.results_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', alpha=0.25)
        sns.scatterplot(data=self.results_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', legend=False)

        plt.xlabel('Pretrained Percentile')
        plt.ylabel(plot_name)
        plt.title(f'{plot_name} vs Pretrained Percentile')

        plt.savefig(self.output_dir / 'mean-object-return-vs-percentile.png')
        plt.close()

    def plot_object_level_returns(self):
        plot_name, plot_key = 'Mean Object-level Return', 'ObjectLevelMeanReward'

        if plot_key not in self.results_df.columns:
            return

        plt.figure(figsize=(15, 10))
        sns.lineplot(data=self.episode_stats_df, x='Pretrained Percentile', y='Return', hue='Meta-level Policy', alpha=0.5)
        sns.scatterplot(data=self.results_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', legend=False)

        plt.xlabel('Pretrained Percentile')
        plt.ylabel(plot_name)
        plt.title(f'{plot_name} vs Pretrained Percentile')

        plt.savefig(self.output_dir / 'object-return-vs-percentile.png')
        plt.close()


def create_parser():
    parser = argparse.ArgumentParser()

    # System parameters
    parser.add_argument('-gpus', '--gpus', nargs='+', type=int, default=None,
                        help='GPU ids to use. If not specified, all GPUs will be used.')

    # Run parameters
    parser.add_argument('--pretrained_runs_folder', type=str, default='runs')
    parser.add_argument('--pretrained_run', type=str, default='run-16823527592836354')
    parser.add_argument('--max_tree_size', type=int, default=64)
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--percentiles', type=float, nargs='+',
                        default=[0.1, 0.25, 0.5, 0.75, 0.9])

    # Video parameters
    parser.add_argument('--create_videos', action='store_true', default=False)
    parser.add_argument('--video_fps', type=int, default=1)
    parser.add_argument('--video_steps', type=int, default=120)

    return parser


def parse_args():
    return vars(create_parser().parse_args())


def main():
    args = parse_args()

    print('Running with args:')
    for k, v in args.items():
        print(f'\t- {k}: {v}')

    if args.get('gpus'):
        restrict_gpus(args['gpus'])

    n_object_level_episodes = args.get('n_episodes', 10)
    max_object_level_steps = args.get('max_steps', 500)

    if args.get('create_videos', False):
        video_args = {
            'fps': args.get('video_fps', 1),
            'steps': args.get('video_steps', 60)
        }
    else:
        video_args = None

    policy_creators = {
        # 'Instant Terminate': TerminatorPolicy,
        # 'AStar': AStarPolicy,
        # 'Random': create_random_search_policy,
        'Random (No Terminate)': create_random_search_policy_no_terminate
    }
    output_dir = Path('outputs/baseline/procgen') / time_id()
    print(f'Writing results to {output_dir}')

    results_accumulator = ResultsAccumulator(output_dir=output_dir)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(args, f)

    percentiles = args.get('percentiles') or [0.1, 0.25, 0.5, 0.75, 0.9]
    for percentile in percentiles:
        print(f'Evaluating with pretrained model at return {percentile = }')
        test_policies_with_pretrained_model(
            policy_creators, args, output_dir,
            percentile=percentile,
            n_object_level_episodes=n_object_level_episodes,
            video_args=video_args,
            results_observer=results_accumulator,
            max_object_level_steps=max_object_level_steps
        )


if __name__ == '__main__':
    main()
