from numpy import format_parser
from mlrl.procgen.batched_procgen_meta_env import BatchedProcgenMetaEnv
from mlrl.runners.eval_runner import EvalRunner
from mlrl.meta.meta_env import MetaEnv, aggregate_object_level_metrics
from mlrl.meta.meta_policies.a_star_policy import AStarPolicy
from mlrl.meta.meta_policies.random_policy import create_random_search_policy, create_random_search_policy_no_terminate
from mlrl.meta.meta_policies.terminator_policy import TerminatorPolicy
from mlrl.experiments.procgen_meta import create_batched_procgen_meta_envs, load_pretrained_q_network
from mlrl.experiments.procgen_meta import reset_object_level_metrics, get_object_level_metrics
from mlrl.procgen.time_limit_observer import TimeLimitObserver
from mlrl.utils.render_utils import save_video
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


def evaluate_meta_policy(create_policy: callable,
                         n_envs: int,
                         n_object_level_episodes: int,
                         outputs_dir: Path,
                         object_config: dict,
                         args: dict,
                         max_object_level_steps: int = 500,
                         episode_complete_cb: Callable[[dict], None] = None,
                         create_video: bool = False):
    if create_video:
        args['env_multithreading'] = False

    batched_meta_env = create_batched_procgen_meta_envs(
        n_envs=n_envs,
        object_config=object_config,
        **args
    )

    meta_envs = [env for env in batched_meta_env.envs]

    # allocates each environment an equal number of episodes to complete
    # any remaining episodes are allocated to the first few environments
    # these allocations prevent mean returns from being skewed by environments
    # that happen to end early, and thereby complete more episodes

    allocations = [n_object_level_episodes // n_envs] * n_envs
    for i in range(n_object_level_episodes % n_envs):
        allocations[i] += 1

    prog_bar = Progbar(
        n_object_level_episodes,
        unit_name='episode',
        stateful_metrics=['ObjectLevelMeanReward',
                          'ObjectLevelMeanStepsPerEpisode',
                          'ObjectLevelEpisodes',
                          'ObjectLevelCurrentEpisodeReturn',
                          'ObjectLevelCurrentEpisodeSteps']
    )

    def get_metrics():
        return aggregate_object_level_metrics([
            meta_env.get_object_level_metrics()
            for meta_env in meta_envs
        ])

    envs_to_remove = []

    def register_callbacks(i, env):
        time_limit = TimeLimitObserver(env, max_object_level_steps)
        env.object_level_transition_observers.append(time_limit)
        if episode_complete_cb is not None:
            env.object_level_metrics.episode_complete_callbacks.append(episode_complete_cb)
        
        if create_video:
            video_maker = MetaPolicyObjectlevelVideoMaker(outputs_dir / 'videos', env, f'object-level-video-{i}')
            env.object_level_transition_observers.append(video_maker)
            env.object_level_metrics.episode_complete_callbacks.append(video_maker.save_video)

        def remove_if_allocations_complete(*_):
            if allocations[i] == env.object_level_metrics.get_num_episodes():
                envs_to_remove.append(env)

        env.object_level_metrics.episode_complete_callbacks.append(remove_if_allocations_complete)

    for i, env in enumerate(meta_envs):
        register_callbacks(i, env)

    def completed_n_object_level_episodes() -> bool:
        n_complete = sum([
            min(env.object_level_metrics.get_num_episodes(), allocations[i])
            for i, env in enumerate(meta_envs)
        ])

        metrics = get_metrics()
        prog_bar.update(n_complete, values=metrics.items())

        return n_complete == n_object_level_episodes

    policy = create_policy(batched_meta_env)

    from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
    policy = PyTFEagerPolicy(policy, use_tf_function=False, batch_time_steps=False)
    # eval_runner = EvalRunner(
    #     eval_env=batched_meta_env,
    #     policy=create_policy(batched_meta_env),
    #     rewrite_rewards=True,
    #     use_tf_function=False,
    #     convert_to_eager=False,
    #     stop_eval_condition=completed_n_object_level_episodes
    # )

    reset_object_level_metrics(batched_meta_env)
    # eval_results = eval_runner.run()

    eval_results = dict()
    batched_meta_env.reset()
    while not completed_n_object_level_episodes():
        # create_time_step handles envs being removed dynamically
        time_step = batched_meta_env.create_time_step()
        action_step = policy.action(time_step)
        batched_meta_env.step(action_step.action)

        for env in envs_to_remove:
            batched_meta_env.remove_env(env)
        envs_to_remove.clear()

    object_level_results = get_metrics()

    evaluations = {
        **args,
        **object_config,
        **eval_results,
        **object_level_results
    }
    return evaluations


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
    n_envs = min(n_envs, n_object_level_episodes)

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

    print(object_config)

    results = []
    for policy_name, create_policy in policy_creators.items():
        print(f'Evaluating {policy_name}')
        evaluations = evaluate_meta_policy(
            create_policy=create_policy,
            n_envs=n_envs,
            n_object_level_episodes=n_object_level_episodes,
            outputs_dir=outputs_dir,
            object_config=object_config,
            args=args,
            max_object_level_steps=max_object_level_steps,
            episode_complete_cb=lambda stats: results_observer.add_episode_stats(run_id, policy_name, percentile, stats),
            create_video=create_video
        )
        evaluations['Meta-level Policy'] = policy_name
        evaluations['Run ID'] = run_id
        results.append(evaluations)

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
        except Exception as ignored:
            pass

    def plot_rewritten_returns(self):
        plot_name, plot_key = 'Mean Rewritten Meta Return', 'EvalRewrittenAverageReturn'

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

        plt.figure(figsize=(15, 10))
        sns.lineplot(data=self.episode_stats_df, x='Pretrained Percentile', y='Return', hue='Meta-level Policy', alpha=0.5)
        sns.scatterplot(data=self.results_df, x='pretrained_percentile', y=plot_key, hue='Meta-level Policy', legend=False)

        plt.xlabel('Pretrained Percentile')
        plt.ylabel(plot_name)
        plt.title(f'{plot_name} vs Pretrained Percentile')

        plt.savefig(self.output_dir / 'object-return-vs-percentile.png')
        plt.close()


class MetaPolicyObjectlevelVideoMaker:

    def __init__(self,
                 output_dir: Path,
                 meta_env: MetaEnv,
                 video_name: str = 'object-level-video',
                 fps: int = 2):
        self.output_dir = output_dir
        self.video_name = video_name
        self.meta_env = meta_env
        self.fps = fps
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.videos_created = 0
        self.frames = []

    def __call__(self, *_):
        self.frames.append(self.meta_env.render())
    
    def save_video(self, *_):
        path = str(self.output_dir / f'{self.video_name}-{self.videos_created}')
        save_video(self.frames, path, fps=self.fps)
        self.videos_created += 1
        self.frames = []


def create_parser():
    parser = argparse.ArgumentParser()

    # System parameters
    parser.add_argument('-gpus', '--gpus', nargs='+', type=int, default=None,
                        help='GPU ids to use. If not specified, all GPUs will be used.')

    # Run parameters
    parser.add_argument('--pretrained_runs_folder', type=str, default='runs')
    parser.add_argument('--pretrained_run', type=str, default='run-16823527592836354')
    parser.add_argument('--max_tree_size', type=int, default=64)
    parser.add_argument('--n_envs', type=int, default=20)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--percentiles', type=float, nargs='+',
                        default=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    parser.add_argument('--compute_meta_rewards', action='store_true', default=False)

    # Video parameters
    parser.add_argument('--create_videos', action='store_true', default=False)
    parser.add_argument('--video_fps', type=int, default=1)
    parser.add_argument('--video_steps', type=int, default=120)

    return parser


def parse_args():
    return vars(create_parser().parse_args())


def main():
    eval_args = parse_args()

    print('Running with args:')
    for k, v in eval_args.items():
        print(f'\t- {k}: {v}')

    if eval_args.get('gpus'):
        restrict_gpus(eval_args['gpus'])

    n_object_level_episodes = eval_args.get('n_episodes', 10)
    max_object_level_steps = eval_args.get('max_steps', 500)

    if eval_args.get('create_videos', False):
        video_args = {
            'fps': eval_args.get('video_fps', 1),
            'steps': eval_args.get('video_steps', 60)
        }
    else:
        video_args = None

    policy_creators = {
        'Instant Terminate': TerminatorPolicy,
        'AStar': AStarPolicy,
        # 'Random': create_random_search_policy,
        'Random (No Terminate)': create_random_search_policy_no_terminate
    }
    output_dir = Path('outputs/baseline/procgen') / time_id()
    print(f'Writing results to {output_dir}')

    results_accumulator = ResultsAccumulator(output_dir=output_dir)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(eval_args, f)

    percentiles = eval_args.get('percentiles') or [0.1, 0.25, 0.5, 0.75, 0.9]
    for percentile in percentiles:
        print(f'Evaluating with pretrained model at return {percentile = }')
        test_policies_with_pretrained_model(
            policy_creators, eval_args, output_dir,
            percentile=percentile,
            n_object_level_episodes=n_object_level_episodes,
            video_args=video_args,
            results_observer=results_accumulator,
            n_envs=eval_args.get('n_envs', 8),
            max_object_level_steps=max_object_level_steps
        )


if __name__ == '__main__':
    main()
