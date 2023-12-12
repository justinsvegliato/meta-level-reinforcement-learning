from rlts.eval.procgen_baseline_meta import test_policies_with_pretrained_model
from rlts.eval.procgen_rlts import load_best_meta_policy
from rlts.eval.procgen_baseline_meta import create_parser
from rlts.eval.eval_utils import ResultsAccumulator
from rlts.utils.wandb_utils import get_wandb_info_from_run_dir
from rlts.utils import time_id

from pathlib import Path


def parse_args():
    parser = create_parser()
    parser.add_argument('--model_selection', type=str, default='best',
                        help='Which model to use for evaluation, one of: "best", "last", "best_smoothed"')
    parser.add_argument('--smoothing_radius', type=int, default=1)
    parser.add_argument('--min_computation_steps', type=int, default=0)
    parser.add_argument('--costs', type=float, nargs='+', default=[0., 0.05, 0.1, 0.2, 0.025, 0.075])
    return vars(parser.parse_args())


def main():
    eval_args = parse_args()

    env_run_paths = {
        'bigfish': Path('outputs/runs/ppo_run_2023-05-15-06-57-08/')
    }

    run_path = env_run_paths[eval_args['env']]

    run = get_wandb_info_from_run_dir(run_path)
    load_best_meta_policy(run, selection_method='best')

    policy = run['best_policy']
    run_args = run['run_args']
    run_args['min_computation_steps'] = eval_args['min_computation_steps']

    output_dir = Path('outputs/eval/procgen/cost') / time_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Writing results to {output_dir}')

    results_accumulator = ResultsAccumulator(output_dir=output_dir)

    for cost_of_computation in eval_args['costs']:
        run_args['random_cost_of_computation'] = False
        run_args['cost_of_computation'] = cost_of_computation
        print('Testing with cost of computation', cost_of_computation)
        test_policies_with_pretrained_model({'RLTS': lambda _: policy},
                                            meta_config=run_args,
                                            n_envs=eval_args['n_envs'],
                                            n_object_level_episodes=eval_args['n_episodes'],
                                            max_object_level_steps=eval_args['max_steps'],
                                            results_observer=results_accumulator)


if __name__ == '__main__':
    main()
