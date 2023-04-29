from mlrl.runners.eval_runner import EvalRunner
from mlrl.meta.meta_policies.a_star_policy import AStarPolicy
from mlrl.meta.meta_policies.random_policy import create_random_search_policy, create_random_search_policy_no_terminate
from mlrl.meta.meta_policies.terminator_policy import TerminatorPolicy
from mlrl.experiments.procgen_meta import create_batched_procgen_meta_envs, load_pretrained_q_network
from mlrl.experiments.procgen_meta import reset_object_level_metrics, get_object_level_metrics
from mlrl.utils import time_id
from mlrl.utils.system import restrict_gpus

import argparse
from typing import Dict, Callable
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def test_policies_with_pretrained_model(policy_creators: Dict[str, callable],
                                        args: dict,
                                        outputs_dir: Path,
                                        percentile=0.75,
                                        eval_steps_per_env=1000,
                                        video_args: dict = None,
                                        results_observer: Callable[[dict], None] = None):
    create_video = video_args is not None

    args.update({
        'pretrained_percentile': percentile,
        'expand_all_actions': True,
        'finish_on_terminate': True,
    })

    object_config = load_pretrained_q_network(
        folder=args['pretrained_runs_folder'],
        run=args['pretrained_run'],
        percentile=args.get('pretrained_percentile', 0.75),
        verbose=False
    )
    n_envs = args.pop('n_envs', 16)

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
            eval_steps_per_env * n_envs,
            batched_meta_env,
            create_policy(batched_meta_env),
            videos_dir=outputs_dir / 'videos',
            video_env=video_env,
            video_policy=create_policy(video_env),
            rewrite_rewards=True,
            use_tf_function=False
        )

        print(f'Evaluating {policy_name}')
        reset_object_level_metrics(batched_meta_env)
        eval_results = eval_runner.run()
        object_level_results = get_object_level_metrics(batched_meta_env)

        evaluations = {
            'Meta-level Policy': policy_name,
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
        self.results_df = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, results: dict):
        self.results.append(results)
        self.results_df = pd.DataFrame(self.results)
        self.results_df.to_csv(self.output_dir / 'results.csv', index=False)
        self.plot_results()

    def plot_results(self):
        plot_name, plot_key = 'Mean Object-level Return', 'ObjectLevelMeanReward'

        plt.figure(figsize=(15, 10))

        sns.lineplot(data=self.results_df, x='pretrained_percentile', y='pretrained_return', label='Pretrained Return', alpha=0.25, color='r')
        sns.scatterplot(data=self.results_df, x='pretrained_percentile', y='pretrained_return', color='r', legend=False)

        sns.lineplot(data=self.results_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', alpha=0.25)
        sns.scatterplot(data=self.results_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', legend=False)

        plt.xlabel('Pretrained Percentile')
        plt.ylabel(plot_name)
        plt.title(f'{plot_name} vs Pretrained Percentile')

        plt.savefig(self.output_dir / 'results.png')


def parse_args():
    parser = argparse.ArgumentParser()

    # System parameters
    parser.add_argument('-gpus', '--gpus', nargs='+', type=int, default=None,
                        help='GPU ids to use. If not specified, all GPUs will be used.')

    # Run parameters
    parser.add_argument('--pretrained_runs_folder', type=str, default='runs')
    parser.add_argument('--pretrained_run', type=str, default='run-16823527592836354')
    parser.add_argument('--max_tree_size', type=int, default=64)
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--eval_steps_per_env', type=int, default=1000)

    # Video parameters
    parser.add_argument('--no_video', action='store_true', default=False)
    parser.add_argument('--video_fps', type=int, default=1)
    parser.add_argument('--video_steps', type=int, default=120)

    return vars(parser.parse_args())


def main():
    args = parse_args()

    print('Running with args:')
    for k, v in args.items():
        print(f'\t- {k}: {v}')

    if args.get('gpus'):
        restrict_gpus(args['gpus'])

    eval_steps_per_env = args.get('eval_steps_per_env', 1000)

    if args.get('no_video', False):
        video_args = None
    else:
        video_args = {
            'fps': args.get('video_fps', 1),
            'steps': args.get('video_steps', 60)
        }

    policy_creators = {
        'Terminator': TerminatorPolicy,
        'AStar': AStarPolicy,
        'Random': create_random_search_policy,
        'RandomNoTerminate': create_random_search_policy_no_terminate
    }

    results_accumulator = ResultsAccumulator(output_dir=Path('outputs/baseline/procgen') / time_id())

    for percentile in [0.25, 0.5, 0.75, 0.9]:
        print(f'Evaluating with pretrained model at return {percentile = }')
        test_policies_with_pretrained_model(
            policy_creators, args, results_accumulator.output_dir,
            percentile=percentile,
            eval_steps_per_env=eval_steps_per_env,
            video_args=video_args,
            results_observer=results_accumulator
        )


if __name__ == '__main__':
    main()
