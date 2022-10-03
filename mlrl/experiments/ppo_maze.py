from mlrl.experiments.experiment_utils import (
    parse_args, create_agent, create_training_run,
    create_batched_tf_meta_env, create_meta_env
)
from mlrl.meta.search_tree import ObjectState
from mlrl.meta.meta_env import MetaEnv
from mlrl.maze.maze_state import RestrictedActionsMazeState, MazeState
from mlrl.maze.manhattan_q import ManhattanQHat
from mlrl.maze.maze_env import make_maze_env

import os
from pathlib import Path
import logging
from typing import Type

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import ppo_learner
from tf_agents.train import triggers
from tf_agents.train.utils import train_utils


def get_maze_name(args) -> str:
    name = ''

    if args.get('maze_procgen', False):
        name += 'procgen_'

    n = args.get('maze_size', 5)
    agent = args['agent']
    name += f'{n}x{n}_maze_{agent}'

    return name


def create_maze_meta_env(object_state_cls: Type[ObjectState],
                         args: dict) -> MetaEnv:
    maze_n = args.get('maze_size', 5)
    procgen = bool(args.get('maze_procgen', False))

    object_env = make_maze_env(
        seed=args.get('seed', 0),
        maze_size=(maze_n, maze_n),
        goal_reward=1,
        render_shape=(64, 64),
        generate_new_maze_on_reset=procgen,
    )

    q_hat = ManhattanQHat(object_env)

    return create_meta_env(
        object_env,
        object_state_cls.extract_state(object_env),
        q_hat,
        args,
        object_action_to_string=lambda a: object_env.ACTION[a]
    )


def main():

    args = parse_args()

    from mlrl.config import RUNS_DIR
    from mlrl.run import time_id
    env_name = get_maze_name(args)
    root_dir = f'{RUNS_DIR}/ppo_agent/{env_name}-{time_id()}'
    Path(root_dir).mkdir(parents=True, exist_ok=True)

    if args.get('restricted_maze_states', True):
        object_state_cls = RestrictedActionsMazeState
    else:
        object_state_cls = MazeState

    env = create_batched_tf_meta_env(
        lambda: create_maze_meta_env(object_state_cls, args),
        args.get('env_batch_size')
    )
    eval_env = create_batched_tf_meta_env(
        lambda: create_maze_meta_env(object_state_cls, args),
        args.get('env_batch_size')
    )

    agent, _ = create_agent(env, **args)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,
        max_length=10000
    )

    train_step = train_utils.create_train_step()

    saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
    Path(saved_model_dir).mkdir(parents=True, exist_ok=True)

    collect_env_step_metric = py_metrics.EnvironmentSteps()
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            agent,
            train_step,
            interval=args.get('policy_save_interval', 1000),
            metadata_metrics={
                triggers.ENV_STEP_METADATA_KEY: collect_env_step_metric
            }),
        triggers.StepPerSecondLogTrigger(train_step, interval=args.get('summary_interval', 1000)),
    ]

    def training_dataset_fn():
        return replay_buffer.as_dataset(
            sample_batch_size=env.batch_size,
            sequence_preprocess_fn=agent.preprocess_sequence)

    agent_learner = ppo_learner.PPOLearner(
        root_dir,
        train_step,
        agent,
        experience_dataset_fn=training_dataset_fn,
        normalization_dataset_fn=training_dataset_fn,
        num_samples=1,
        num_epochs=args.get('num_epochs', 10),
        minibatch_size=args.get('minibatch_size', 64),
        shuffle_buffer_size=args.get('collect_sequence_length', 2048),
        triggers=learning_triggers)

    tf_collect_policy = agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)

    collect_actor = actor.Actor(
        env,
        collect_policy,
        train_step,
        steps_per_run=args.get('collect_sequence_length', 2048),
        observers=[replay_buffer.add_batch],
        metrics=actor.collect_metrics(buffer_size=10) + [collect_env_step_metric],
        reference_metrics=[collect_env_step_metric],
        summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
        summary_interval=args.get('summary_interval', 1000))

    eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
        agent.policy, use_tf_function=True)

    eval_interval = args.get('eval_interval', 10000)
    eval_episodes = args.get('eval_episodes', 10)
    if eval_interval:
        logging.info('Intial evaluation.')
        eval_actor = actor.Actor(
            eval_env,
            eval_greedy_policy,
            train_step,
            metrics=actor.eval_metrics(eval_episodes),
            reference_metrics=[collect_env_step_metric],
            summary_dir=os.path.join(root_dir, 'eval'),
            episodes_per_run=eval_episodes)

    eval_actor.run_and_log()

    videos_dir = os.path.join(root_dir, 'videos')
    Path(videos_dir).mkdir(parents=True, exist_ok=True)

    logging.info('Training on %s', env_name)
    last_eval_step = 0
    num_iterations = 2000
    for i in range(num_iterations):
        collect_actor.run()
        agent_learner.run()

        if (eval_interval and
            (agent_learner.train_step_numpy >= eval_interval + last_eval_step
                or i == num_iterations - 1)):
            logging.info('Evaluating.')

            video_file = f'{videos_dir}/eval_video_{i}.mp4'
            from mlrl.utils.render_utils import create_policy_eval_video
            create_policy_eval_video(
                eval_greedy_policy, eval_env,
                max_steps=300, filename=video_file
            )

            eval_actor.run_and_log()
            last_eval_step = agent_learner.train_step_numpy

    logging.info('Done training.')


if __name__ == "__main__":
    main()
