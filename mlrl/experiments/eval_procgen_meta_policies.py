import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List

from mlrl.meta.meta_policies.search_ppo_agent import load_ppo_agent
from mlrl.experiments.procgen_meta import create_batched_procgen_meta_envs, load_pretrained_q_network
from mlrl.experiments.procgen_baseline_meta import test_policies_with_pretrained_model, ResultsAccumulator, create_parser
from mlrl.utils.system import restrict_gpus
from mlrl.utils.wandb_utils import get_wandb_info_from_run_dir
from mlrl.utils import time_id

sns.set()


def load_policy_from_checkpoint(run: dict, epoch: int):

    ckpt_dir = run['root_dir'] / f'network_checkpoints/step_{epoch}'

    exclude_keys = ['learning_rate', 'name']
    run_args = {
        k: v for k, v in run['config'].items()
        if k not in exclude_keys
    }

    run['run_args'] = run_args

    load_pretrained_q_network(
        folder=run_args['pretrained_runs_folder'],
        run=run_args['pretrained_run'],
        percentile=run_args['pretrained_percentile'],
        verbose=False
    )

    batched_meta_env = create_batched_procgen_meta_envs(
        1, run_args['object_level_config'], **run_args
    )

    agent = load_ppo_agent(batched_meta_env, run_args, ckpt_dir)

    return agent.policy


def load_best_policies(runs: List[dict], output_dir: Path, selection_method: str = 'best'):

    _, ax = plt.subplots()
    for run in runs:
        wanbd_run = run['run']
        df = run['history']
        df = df[df['EvalRewrittenAverageReturn'].notna()]
        y = df['EvalRewrittenAverageReturn'].array
        y_smooth = np.concatenate([[y[0]], np.convolve(y, np.ones(3) / 3, mode='valid'), [y[-1]]])

        x = df['TrainStep'].array
        xs = list(x)

        if selection_method == 'best':
            model_epoch = max(xs, key=lambda i: y_smooth[xs.index(i)])
        elif selection_method == 'last':
            model_epoch = max(xs)
        else:
            raise ValueError(f'Unknown selection method: {selection_method}')

        run['model_epoch'] = model_epoch

        run['best_policy'] = load_policy_from_checkpoint(run, model_epoch)

        line, *_ = ax.plot(x, y, alpha=0.25)
        ax.plot(df['TrainStep'], y_smooth, label=wanbd_run.name, color=line.get_color())
        ax.scatter(model_epoch, y_smooth[xs.index(model_epoch)], color=line.get_color())

    # # legend outside plot
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.legend()
    ax.set_xlabel('Train Step')
    ax.set_ylabel('Mean Rewritten Return')
    ax.set_title('Meta Policy Training')

    plt.savefig(output_dir / 'meta_policy_training_curves.png')


def parse_args():
    parser = create_parser()
    parser.add_argument('--model_selection', type=str, default='best',
                        help='Which model to use for evaluation, one of: "best", "last"')
    return vars(parser.parse_args())


def main():

    eval_args = parse_args()

    print('Running with args:')
    for k, v in eval_args.items():
        print(f'\t- {k}: {v}')

    if eval_args.get('gpus'):
        restrict_gpus(eval_args['gpus'])
    output_dir = Path('outputs/eval/procgen') / time_id()
    print(f'Writing results to {output_dir}')

    results_accumulator = ResultsAccumulator(output_dir=output_dir)

    meta_policy_model_paths = [
        # Path('outputs/runs/ppo_run_51-48-04-01-05-2023/'),
        # Path('outputs/runs/ppo_run_01-51-04-01-05-2023/'),
        # Path('outputs/runs/ppo_run_39-44-04-01-05-2023/'),
        Path('outputs/runs/ppo_run_36-37-06-02-05-2023/'),
        Path('outputs/runs/ppo_run_29-19-06-02-05-2023/'),
        Path('outputs/runs/ppo_run_38-34-06-02-05-2023/'),
        # Path('outputs/runs/ppo_run_37-21-09-02-05-2023/'),
    ]

    runs = [
        get_wandb_info_from_run_dir(root_dir)
        for root_dir in meta_policy_model_paths
    ]

    load_best_policies(runs, output_dir, selection_method=eval_args['model_selection'])

    n_object_level_episodes = eval_args.get('n_episodes', 10)
    max_object_level_steps = eval_args.get('max_steps', 500)

    if eval_args.get('no_video', False):
        video_args = None
    else:
        video_args = {
            'fps': eval_args.get('video_fps', 1),
            'steps': eval_args.get('video_steps', 60)
        }

    for run in runs:
        percentile = run['config']['pretrained_percentile']

        policy_creators = {
            f'Learned Meta-Policy': lambda _: run['best_policy']
        }

        test_policies_with_pretrained_model(policy_creators,
                                            percentile=percentile,
                                            args=run['run_args'],
                                            outputs_dir=output_dir,
                                            results_observer=results_accumulator,
                                            video_args=video_args,
                                            max_object_level_steps=max_object_level_steps,
                                            n_envs=1,
                                            n_object_level_episodes=n_object_level_episodes)


if __name__ == '__main__':
    main()
