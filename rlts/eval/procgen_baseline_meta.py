from rlts.meta.meta_env import aggregate_object_level_metrics
from rlts.meta.meta_policies.a_star_policy import AStarPolicy
from rlts.meta.meta_policies.random_policy import create_random_search_policy_no_terminate
from rlts.meta.meta_policies.terminator_policy import TerminatorPolicy
from rlts.eval.eval_utils import ResultsAccumulator, MetaPolicyObjectlevelVideoMaker
from rlts.train.procgen_meta import create_batched_procgen_meta_envs, load_pretrained_q_network
from rlts.train.procgen_meta import reset_object_level_metrics
from rlts.procgen.time_limit_observer import TimeLimitObserver
from rlts.utils import time_id, clean_for_json
from rlts.utils.system import restrict_gpus

import argparse
from typing import Dict, Callable
from pathlib import Path
import json

from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tensorflow.keras.utils import Progbar


def evaluate_meta_policy(create_policy: callable,
                         n_envs: int,
                         n_object_level_episodes: int,
                         object_config: dict,
                         args: dict,
                         outputs_dir: Path = '.',
                         max_object_level_steps: int = 500,
                         episode_complete_cb: Callable[[dict], None] = None,
                         remove_envs_on_completion: bool = True,
                         create_video: bool = False):
    if create_video:
        args['env_multithreading'] = False

    batched_meta_env = create_batched_procgen_meta_envs(
        n_envs=n_envs,
        object_config=object_config,
        **args
    )

    policy = create_policy(batched_meta_env)

    if isinstance(policy, AStarPolicy) and remove_envs_on_completion:
        print('AStarPolicy does not support remove_envs_on_completion. Setting to False')
        remove_envs_on_completion = False

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
            video_maker = MetaPolicyObjectlevelVideoMaker(outputs_dir / 'videos',
                                                          env, f'object-level-video-{i}')
            env.object_level_transition_observers.append(video_maker)
            env.object_level_metrics.episode_complete_callbacks.append(video_maker.save_video)

        def remove_if_allocations_complete(*_):
            if allocations[i] == env.object_level_metrics.get_num_episodes():
                envs_to_remove.append(env)

        if remove_envs_on_completion:
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

    policy = PyTFEagerPolicy(policy, use_tf_function=False, batch_time_steps=False)

    reset_object_level_metrics(batched_meta_env)

    eval_results = dict()
    batched_meta_env.reset()
    while not completed_n_object_level_episodes():
        # create_time_step handles envs being removed dynamically
        time_step = batched_meta_env.create_time_step()
        action_step = policy.action(time_step)
        batched_meta_env.step(action_step.action)

        if remove_envs_on_completion:
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
                                        outputs_dir: Path = None,
                                        percentile=None,
                                        n_object_level_episodes=10,
                                        video_args: dict = None,
                                        max_object_level_steps=50,
                                        n_envs=8,
                                        run_id: str = None,
                                        results_observer: Callable[[dict], None] = None):
    create_video = video_args is not None
    n_envs = min(n_envs, n_object_level_episodes)
    results_observer = results_observer or ResultsAccumulator()

    args.update({
        'expand_all_actions': True,
        'finish_on_terminate': True,
    })

    if percentile is not None:
        args['pretrained_percentile'] = percentile
    else:
        percentile = args['pretrained_percentile']

    if 'n_envs' in args:
        args.pop('n_envs')

    object_config = load_pretrained_q_network(
        folder=args['pretrained_runs_folder'],
        run=args['pretrained_run'],
        percentile=args['pretrained_percentile'],
        verbose=False
    )

    if outputs_dir is not None:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        with open(outputs_dir / f'object_level_config_{percentile=}.json', 'w') as f:
            json.dump(clean_for_json(object_config), f)

    results = []
    for policy_name, create_policy in policy_creators.items():

        def on_episode_complete(epsiode_stats):
            stats = {
                **args,
                **object_config,
                **epsiode_stats
            }
            results_observer.add_episode_stats(
                run_id, policy_name, percentile, stats)

        print(f'Evaluating {policy_name}')
        if outputs_dir is not None:
            policy_outputs_dir = outputs_dir / policy_name
            policy_outputs_dir.mkdir(parents=True, exist_ok=True)
        else:
            policy_outputs_dir = None

        evaluations = evaluate_meta_policy(
            create_policy=create_policy,
            n_envs=n_envs,
            n_object_level_episodes=n_object_level_episodes,
            outputs_dir=policy_outputs_dir,
            object_config=object_config,
            args=args,
            max_object_level_steps=max_object_level_steps,
            episode_complete_cb=on_episode_complete,
            create_video=create_video
        )
        evaluations['Meta-level Policy'] = policy_name
        evaluations['Run ID'] = run_id
        results.append(evaluations)

    return results_observer, results


def create_parser():
    parser = argparse.ArgumentParser()

    # System parameters
    parser.add_argument('-gpus', '--gpus', nargs='+', type=int, default=None,
                        help='GPU ids to use. If not specified, all GPUs will be used.')

    # Run parameters
    parser.add_argument('--pretrained_runs_folder', type=str, default='runs')
    parser.add_argument('--pretrained_run', type=str, default=None)
    parser.add_argument('--env', type=str, default='bigfish')
    parser.add_argument('--max_tree_size', type=int, default=64)
    parser.add_argument('--n_envs', type=int, default=20)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--percentiles', type=float, nargs='+',
                        default=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    parser.add_argument('--compute_meta_rewards', action='store_true', default=False)
    parser.add_argument('--baselines', type=str, nargs='+',
                        default=['random', 'a_star', 'instant_terminate'],
                        help='Baselines to run evaluations for. Options are: '
                             'random, a_star, instant_terminate')

    # Video parameters
    parser.add_argument('--create_videos', action='store_true', default=False)
    parser.add_argument('--video_fps', type=int, default=1)
    parser.add_argument('--video_steps', type=int, default=120)
    parser.add_argument('--render_plans', action='store_true', default=False)

    return parser


def parse_args():
    return vars(create_parser().parse_args())


def main():
    eval_args = parse_args()

    default_pretrained_runs = {
        'fruitbot': 'run-16833079943304386',
        'coinrun': 'run-16838619373401126',
        'bossfight': 'run-16839105526160484',
        'bigfish': 'run-16823527592836354',
        'caveflyer': 'run-16947090705703208'
    }

    env_name = eval_args['env']
    if eval_args['pretrained_run'] is None and env_name in default_pretrained_runs:
        eval_args['pretrained_run'] = default_pretrained_runs[env_name]
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

    output_dir = Path(f'outputs/baseline/procgen/{env_name}') / time_id()
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
