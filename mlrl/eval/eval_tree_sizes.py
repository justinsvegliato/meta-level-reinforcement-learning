from mlrl.utils.system import restrict_gpus
from mlrl.meta.meta_policies.a_star_policy import AStarPolicy
from mlrl.meta.meta_policies.random_policy import create_random_search_policy_no_terminate
from mlrl.eval.eval_utils import ResultsAccumulator
from mlrl.eval.procgen_baseline_meta import test_policies_with_pretrained_model, create_parser
from mlrl.eval.procgen_rlts import load_best_meta_policy
from mlrl.utils.wandb_utils import get_wandb_info_from_run_dir
from mlrl.utils import time_id
from mlrl.utils.system import restrict_gpus

from pathlib import Path


def parse_args():
    parser = create_parser()

    parser.add_argument('--model_selection', type=str, default='best',
                        help='Which model to use for evaluation, one of: "best", "last", "best_smoothed"')
    parser.add_argument('--smoothing_radius', type=int, default=1)

    parser.add_argument('--max_tree_sizes', type=int, nargs='+',
                        default=[16, 32, 128])  # 64 was the default in other scripts
    parser.add_argument('--min_computation_steps', type=int, default=128)
    parser.add_argument('--envs', type=str, nargs='+', default=['coinrun', 'fruitbot', 'bigfish', 'bossfight'])

    return vars(parser.parse_args())


def main():
    eval_args = parse_args()

    # default_pretrained_runs = {
    #     'fruitbot': 'run-16833079943304386',
    #     'coinrun': 'run-16838619373401126',
    #     'bossfight': 'run-16839105526160484',
    #     'bigfish': 'run-16823527592836354'
    # }
    meta_policy_model_paths = {
        'bigfish': Path('outputs/runs/ppo_run_36-37-06-02-05-2023/'),
        'fruitbot': Path('outputs/runs/ppo_run_15-48-02-12-05-2023/'),
        'coinrun': Path('outputs/runs/ppo_run_52-06-02-13-05-2023/'),
        'bossfight': Path('outputs/runs/ppo_run_2023-05-15-17-52-00/')
    }

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

    runs = {
        env: get_wandb_info_from_run_dir(meta_policy_model_paths[env])
        for env in eval_args['envs']
    }

    baseline_policy_creators = dict()
    if 'random' in eval_args.get('baselines'):
        baseline_policy_creators['Random (No Terminate)'] = create_random_search_policy_no_terminate
    if 'a_star' in eval_args.get('baselines'):
        baseline_policy_creators['AStar'] = AStarPolicy

    output_dir = Path('outputs/eval/procgen/tree_size') / time_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Writing results to {output_dir}')

    results_accumulator = ResultsAccumulator(output_dir=output_dir)
    print(f'Evaluating with max tree sizes: {eval_args["max_tree_sizes"]}')
    for env, run in runs.items():
        run_name = run['run'].name
        run_id = run['run_id']

        policy_creators = {
            'Learned Meta-Policy': lambda _: run['best_policy'],
            **baseline_policy_creators
        }

        for max_tree_size in eval_args['max_tree_sizes']:

            override_run_args = {
                'max_tree_size': max_tree_size,
                'min_computation_steps': eval_args['min_computation_steps']
            }

            print(f'Loading best meta policy for run {run_name}')
            output_path = output_dir / f'meta_policy_training_curve_{run_name}.png'
            load_best_meta_policy(run, output_path,
                                  selection_method=eval_args['model_selection'],
                                  smoothing_radius=eval_args['smoothing_radius'],
                                  override_run_args=override_run_args)

            run_args = run['run_args']

            print(f'Running evaluations on {env} with run {run_name} ({run_id}) and max tree size {max_tree_size}')
            test_policies_with_pretrained_model(policy_creators,
                                                args=run_args,
                                                outputs_dir=output_dir / env / f'max_tree_size_{max_tree_size}',
                                                results_observer=results_accumulator,
                                                video_args=video_args,
                                                max_object_level_steps=max_object_level_steps,
                                                n_envs=eval_args['n_envs'],
                                                n_object_level_episodes=n_object_level_episodes,
                                                run_id=f'{run_id}/{run_name}')


if __name__ == '__main__':
    main()
