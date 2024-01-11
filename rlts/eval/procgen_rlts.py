import json
from pathlib import Path
from typing import List, Optional, Union
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rlts.meta.meta_policies.search_ppo_agent import load_ppo_agent
from rlts.train.procgen_meta import create_batched_procgen_meta_envs, load_pretrained_q_network
from rlts.eval.procgen_baseline_meta import test_policies_with_pretrained_model, ResultsAccumulator, create_parser
from rlts.utils.system import restrict_gpus, get_most_recently_modified_directory
from rlts.utils.wandb_utils import get_wandb_info_from_run_dir
from rlts.utils import time_id

sns.set()


def load_meta_policy_from_checkpoint(run: dict,
                                     epoch: int,
                                     override_run_args: dict = None):

    ckpt_dir = run['root_dir'] / f'network_checkpoints/step_{epoch}'

    print(f'Loading meta policy from checkpoint: {ckpt_dir}')

    exclude_keys = ['learning_rate', 'name']
    run_args = {
        k: v for k, v in run['config'].items()
        if k not in exclude_keys
    }
    run_args.update(override_run_args or {})
    run['run_args'] = run_args

    load_pretrained_q_network(
        folder=run_args['pretrained_runs_folder'],
        run=run_args['pretrained_run'],
        epoch=run_args['object_level_config']['pretrained_epoch'],
        verbose=False
    )

    batched_meta_env = create_batched_procgen_meta_envs(
        1, run_args['object_level_config'], **run_args
    )

    agent = load_ppo_agent(batched_meta_env, run_args, ckpt_dir)

    return agent.policy


def load_most_recent_meta_policy(run: List[dict]):

    ckpts_dir = run['root_dir'] / f'network_checkpoints/'
    ckpt_dir = get_most_recently_modified_directory(ckpts_dir)
    match = re.search(r'step_(\d+)$', ckpt_dir.stem)
    model_epoch = int(match.group(1))

    run['model_epoch'] = model_epoch
    run['best_policy'] = load_meta_policy_from_checkpoint(run, model_epoch)


def determine_metric_for_best(df: pd.DataFrame):
    if 'EvalRewrittenAverageReturn' in df.columns:
        return 'EvalRewrittenAverageReturn'
    if 'Eval/RewrittenAverageReturn' in df.columns:
        return 'Eval/RewrittenAverageReturn'
    if 'Collect/RewrittenAverageReturn' in df.columns:
        return 'Collect/RewrittenAverageReturn'
    if 'Collect/AverageReturn' in df.columns:
        return 'Collect/AverageReturn'

    raise ValueError('Could not determine metric for best model. '
                     'Please specify manually.')


def load_best_meta_policy(run: List[dict],
                          output_path: Path = None,
                          selection_method: str = 'best',
                          override_run_args: dict = None,
                          metric_for_best: Optional[str] = None,
                          max_load_epoch: Optional[int] = None,
                          smoothing_radius: int = 1):

    _, ax = plt.subplots()
    wanbd_run = run['run']
    df = run['history']
    df.sort_values(by='TrainStep', inplace=True)

    steps_with_ckpts = [
        int(p.stem[5:])
        for p in (run['root_dir'] / 'network_checkpoints').glob(r'step_*')
    ]

    df = df[df['TrainStep'].isin(steps_with_ckpts)]

    if max_load_epoch is not None:
        df = df[df['TrainStep'] <= max_load_epoch]

    if metric_for_best is None:
        metric_for_best = determine_metric_for_best(df)

    df = df[df[metric_for_best].notna()]
    y = df[metric_for_best].array
    smooth_d = 1 + 2 * smoothing_radius
    y_smooth = np.concatenate([
        [y[0]] * smoothing_radius,
        np.convolve(y, np.ones(smooth_d) / smooth_d, mode='valid'),
        [y[-1]] * smoothing_radius
    ])

    x = df['TrainStep'].array
    xs = list(x)

    if selection_method == 'best':
        model_epoch = max(xs, key=lambda i: y[xs.index(i)])
    elif selection_method == 'best_smoothed':
        model_epoch = max(xs, key=lambda i: y_smooth[xs.index(i)])
    elif selection_method == 'last':
        model_epoch = max(xs)
    else:
        raise ValueError(f'Unknown selection method: {selection_method}. '
                         'Options are: "best", "last", "best_smoothed"')

    run['model_epoch'] = model_epoch

    run['best_policy'] = load_meta_policy_from_checkpoint(run, model_epoch, override_run_args)

    line, *_ = ax.plot(x, y, alpha=0.25)
    df.sort_values(by='TrainStep', inplace=True)
    ax.plot(df['TrainStep'], y_smooth, label=wanbd_run.name, color=line.get_color())
    ax.scatter(model_epoch, y_smooth[xs.index(model_epoch)], color=line.get_color())

    # # legend outside plot
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.legend()
    ax.set_xlabel('Train Step')
    ax.set_ylabel('Mean Rewritten Return')
    ax.set_title('Meta Policy Training')

    if output_path:
        plt.savefig(output_path)


def parse_args():
    parser = create_parser()
    parser.add_argument('--model_selection', type=str, default='best',
                        help='Which model to use for evaluation, one of: "best", "last", "best_smoothed"')
    parser.add_argument('--smoothing_radius', type=int, default=1)
    # parser.add_argument('--env', type=str, default='bigfish',
    #                     help='Environment to evaluate on, one of: "bigfish", "coinrun"')
    parser.add_argument('--min_computation_steps', type=int, default=30)
    parser.add_argument('--load_epoch', type=int, default=None)
    return vars(parser.parse_args())


def load_runs(output_dir: Optional[Path] = None,
              model_selection: str = 'best',
              env: Union[Optional[List[str]], str] = None,
              percentiles: Union[Optional[List[float]], float] = None,
              **eval_args):

    meta_policy_model_paths = {
        'bigfish': {
            0.1: Path('outputs/runs/ppo_run_06-55-11-09-05-2023/'),
            0.75: Path('outputs/runs/ppo_run_2023-11-19-23-13-50/'),
        },
        'fruitbot': {1.0: Path('outputs/runs/ppo_run_15-48-02-12-05-2023/')},
        'coinrun': {1.0: Path('outputs/runs/ppo_run_52-06-02-13-05-2023/')},
        'bossfight': {1.0: Path('outputs/runs/ppo_run_2023-05-15-07-39-48/'),
                      0.25: Path('outputs/runs/ppo_run_2023-05-15-17-52-00/')},
        'caveflyer': {0.1: Path('outputs/runs/ppo_run_2024-01-09-15-35-14/')},
    }

    if isinstance(env, str):
        env = [env]

    if env is None or len(env) == 0:
        env = meta_policy_model_paths.keys()

    if isinstance(percentiles, float):
        percentiles = [percentiles]

    if percentiles is None or len(percentiles) == 0:
        percentiles = set([
            percentile
            for env_name, env_model_paths in meta_policy_model_paths.items()
            if env_name in env
            for percentile in env_model_paths.keys()
        ])

    runs = [
        get_wandb_info_from_run_dir(root_dir)
        for env_name, env_model_paths in meta_policy_model_paths.items()
        if env_name in env
        for percentile, root_dir in env_model_paths.items()
        if percentile in percentiles
    ]

    for run in runs:
        if eval_args.get('load_epoch', None) is not None:
            load_meta_policy_from_checkpoint(run, eval_args['load_epoch'])
            continue

        loaded_from_wandb = run['run'] is not None
        load_best = model_selection != 'last'
        if loaded_from_wandb and load_best:
            run_name = run['run'].name
            print(f'Loading best meta policy for run {run_name}')

            if output_dir is not None:
                output_path = output_dir / f'meta_policy_training_curve_{run_name}.png'
            else:
                output_path = None

            load_best_meta_policy(run,
                                  output_path,
                                  model_selection,
                                  smoothing_radius=eval_args.get('smoothing_radius', 1))
        else:
            print('Loading most recent meta policy')
            load_most_recent_meta_policy(run)

    return runs


def main():

    eval_args = parse_args()

    print('Running with args:')
    for k, v in eval_args.items():
        print(f'\t- {k}: {v}')

    if eval_args.get('gpus'):
        restrict_gpus(eval_args['gpus'])

    object_env = eval_args['env']
    output_dir = Path('outputs/eval/procgen') / object_env / time_id()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(output_dir=output_dir, **eval_args)

    with open(output_dir / 'wandb_runs_info.json', 'w') as f:
        json.dump(runs, f, default=lambda _: '<not serializable>')

    n_object_level_episodes = eval_args.get('n_episodes', 10)
    max_object_level_steps = eval_args.get('max_steps', 500)

    if eval_args.get('create_videos', False):
        video_args = {
            'fps': eval_args.get('video_fps', 1),
            'steps': eval_args.get('video_steps', 60)
        }
    else:
        video_args = None

    results_accumulator = ResultsAccumulator(output_dir=output_dir)
    print(f'Writing results to {output_dir}')

    for run in runs:
        percentile = run['config']['pretrained_percentile']
        policy_creators = {
            'Learned Meta-Policy': lambda _: run['best_policy']
        }
        run['run_args']['min_computation_steps'] = eval_args['min_computation_steps']

        run_id = run['run_id']
        run_name = run['run'].name if run['run'] is not None else '<unknown>'
        print(f'Evaluating on with pretrained model at percentile {percentile} [{run_id}/{run_name}]')
        test_policies_with_pretrained_model(policy_creators,
                                            percentile=percentile,
                                            meta_config=run['run_args'],
                                            outputs_dir=output_dir,
                                            results_observer=results_accumulator,
                                            video_args=video_args,
                                            max_object_level_steps=max_object_level_steps,
                                            n_envs=eval_args['n_envs'],
                                            n_object_level_episodes=n_object_level_episodes,
                                            run_id=f'{run_id}/{run_name}')

    print(f'Finished evaluation. Find results at {output_dir}')


if __name__ == '__main__':
    main()
