from mlrl.utils.system import restrict_gpus
from mlrl.meta.meta_policies.a_star_policy import AStarPolicy
from mlrl.meta.meta_policies.random_policy import create_random_search_policy_no_terminate
from mlrl.meta.meta_policies.terminator_policy import TerminatorPolicy
from mlrl.eval.eval_utils import ResultsAccumulator
from mlrl.eval.procgen_baseline_meta import test_policies_with_pretrained_model, create_parser
from mlrl.utils import time_id
from mlrl.utils.system import restrict_gpus

from pathlib import Path
import json


def parse_args():
    parser = create_parser()
    
    parser.add_argument('--max_tree_sizes', type=int, nargs='+',
                        default=[16, 32, 128])  # 64 was the default in other scripts

    parser.add_argument('--envs', type=str, nargs='+', default=None)

    return vars(parser.parse_args())


def main():
    eval_args = parse_args()

    default_pretrained_runs = {
        'fruitbot': 'run-16833079943304386',
        'coinrun': 'run-16838619373401126',
        'bossfight': 'run-16839105526160484',
        'bigfish': 'run-16823527592836354'
    }
    meta_policy_model_paths = {
        'bigfish': {0.1: Path('outputs/runs/ppo_run_06-55-11-09-05-2023/')},
        'fruitbot': {1.0: Path('outputs/runs/ppo_run_15-48-02-12-05-2023/')},
        'coinrun': {1.0: Path('outputs/runs/ppo_run_52-06-02-13-05-2023/')},
        'bossfight': 0.25: Path('outputs/runs/ppo_run_2023-05-15-17-52-00/')
    }

    if eval_args['pretrained_run'] is None  and eval_args['env'] in default_pretrained_runs:
        eval_args['pretrained_run'] = default_pretrained_runs[eval_args['env']]
    else:
        raise ValueError('Must specify pretrained_run or env')

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

    policy_creators = dict()
    if 'random' in eval_args.get('baselines'):
        policy_creators['Random (No Terminate)'] = create_random_search_policy_no_terminate
    if 'a_star' in eval_args.get('baselines'):
        policy_creators['AStar'] = AStarPolicy
    if 'instant_terminate' in eval_args.get('baselines'):
        policy_creators['Instant Terminate'] = TerminatorPolicy

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