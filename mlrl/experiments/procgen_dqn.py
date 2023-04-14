from typing import List, Optional, Tuple, Callable
import numpy as np
import cv2

import procgen
from procgen import ProcgenGym3Env
import gym3
from gym3 import ExtractDictObWrapper

import tensorflow as tf
from tf_agents.utils import common
from tf_agents.drivers import py_driver
from tf_agents.specs import tensor_spec
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.categorical_q_network import CategoricalQNetwork
from tf_agents.agents import CategoricalDqnAgent

from mlrl.utils.render_utils import save_video
from mlrl.runners.eval_runner import EvalRunner
from mlrl.runners.dqn_runner import DQNRun
from mlrl.utils.env_wrappers import ImagePreprocessWrapper, FrameStack
from mlrl.utils.procgen_gym3_wrapper import ProcgenGym3Wrapper


def create_dqn_agent(env, train_steps_per_epoch=10000) -> Tuple[tf.keras.Model, DdqnAgent]:

    q_net = QNetwork(
        env.observation_spec(),
        env.action_spec(),
        conv_layer_params=[(64, 8, 4), (64, 4, 2), (64, 3, 2)],
        fc_layer_params=[512]
    )

    # build weights
    print('Building Q-Network weights...')
    ts = env.current_time_step()
    q_net(ts.observation)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4)
    train_step_counter = tf.Variable(0)

    agent = DdqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        target_update_period=10000,
        train_step_counter=train_step_counter
    )
    agent.initialize()

    q_model = tf.keras.Sequential([q_net])
    q_model(env.current_time_step().observation)

    return q_model, agent


def create_categorical_dqn_agent(
        env, train_steps_per_epoch=10000
        ) -> Tuple[tf.keras.Model, CategoricalDqnAgent]:

    categorical_q_net = CategoricalQNetwork(
        tensor_spec.from_spec(env.observation_spec()),
        tensor_spec.from_spec(env.action_spec()),
        conv_layer_params=[(64, 8, 4), (64, 4, 2), (64, 3, 2)],
        fc_layer_params=[512]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4)
    train_step_counter = tf.Variable(0)

    # build weights
    print('Building Categorical Q-Network weights...')
    ts = env.current_time_step()
    categorical_q_net(ts.observation)

    agent = CategoricalDqnAgent(
        tensor_spec.from_spec(env.time_step_spec()),
        tensor_spec.from_spec(env.action_spec()),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        target_update_period=10000,
        min_q_value=0, max_q_value=32,
        gamma=0.999,
        train_step_counter=train_step_counter
    )

    agent.initialize()

    q_model = tf.keras.Sequential([categorical_q_net])
    q_model(env.current_time_step().observation)

    return q_model, agent


def make_procgen(
        procgen_env_name: str,
        n_envs: Optional[int] = 64,
        action_repeats = 4,
        frame_stack = 4):

    procgen_gym3 = ExtractDictObWrapper(ProcgenGym3Env(
                            num=n_envs,
                            num_threads=min(n_envs, 32),
                            env_name='bigfish',
                            use_backgrounds=False,
                            restrict_themes=True,
                            distribution_mode='easy'), key='rgb')

    wrapped_procgen_gym3 = ProcgenGym3Wrapper(procgen_gym3, action_repeats=action_repeats)
    env = ImagePreprocessWrapper(wrapped_procgen_gym3)
    env = FrameStack(env, frame_stack)    
    env.reset()

    return env


def create_policy_eval_video_frames(
        policy, env,
        render_fn=None,
        steps: int = 60) -> List[np.ndarray]:
    """
    Creates video frames of the policy acting in the given environment.
    If the environment is a batched environment, then multiple episodes
    will be shown stacked vertically.

    Args:
        policy (TFPolicy): The policy to evaluate.
        env (TFEnvironment): The environment to evaluate the policy in.
        steps (int): The maximum number of steps to run the policy for.

    Returns:
        List[np.ndarray]: A list of numpy arrays containing the frames of the video.
    """
    render_fn = render_fn or (lambda e: e.render())

    frames = []

    def observe(_):
        frames.append(render_fn(env))

    driver = py_driver.PyDriver(
        env, policy,
        max_steps=steps,
        observers=[observe]
    )
    
    driver.run(env.current_time_step())

    return frames


def get_grid_dim(n: int) -> Tuple[int, int]:
    """
    Generates the dimensions of a grid closest to
    square with the given number of cells.
    """
    sqrt_n = int(np.ceil(np.sqrt(n)))
    p = 1
    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            p = n / i
    q = n / p
    return int(p), int(q)


def render_env(env):
    """
    Renderers one of a batched env. Scales image for video quality.
    """
    img = env.render()
    img = cv2.resize(img, (4*img.shape[0], 4*img.shape[1]),
                        interpolation=cv2.INTER_NEAREST)
    return img


def create_video_renderer(procgen_env_name: str) -> Callable:
    """
    Returns a callable that produces a video of 12 games
    played simulataneously by a given policy, saved to a given file path
    """
    n_video_envs = 12
    frame_skip = 4
    video_env = make_procgen(procgen_env_name, n_video_envs, action_repeats=frame_skip)
    p, q = get_grid_dim(n_video_envs)

    def render_fn(vectorised_env, *_):
        _, observation, _ = vectorised_env.observe()

        def render_env(i):
            img = observation[i]
            img = cv2.resize(img, (4*img.shape[0], 4*img.shape[1]),
                            interpolation=cv2.INTER_NEAREST)
            return img
        
        frames = [render_env(i) for i in range(n_video_envs)]
        
        return np.concatenate([
            np.concatenate(frames[i*p:(i+1)*p]) for i in range(q)
        ], axis=1)

    fps = 16 // frame_skip

    def create_video(policy, video_file):
        frames = create_policy_eval_video_frames(
            policy, video_env, 
            render_fn=render_fn, steps=fps*60*n_video_envs
        )
        return save_video(frames, video_file, fps=fps)

    return create_video


def main():
    n_collect_envs = 64
    procgen_env_name = 'bigfish'
    collect_env = make_procgen(procgen_env_name, n_envs=n_collect_envs)

    print('Collect env time step spec: ', collect_env.time_step_spec())
    
    train_steps_per_epoch = 20000

    print('Creating agent...')
    q_net, agent = create_dqn_agent(
        collect_env,
        train_steps_per_epoch=train_steps_per_epoch
    )

    n_eval_envs = 64
    eval_envs = make_procgen(procgen_env_name, n_envs=n_eval_envs)
    eval_runner = EvalRunner(n_eval_envs * 1000, eval_envs, agent.policy)

    print(f'Creating {procgen_env_name} DQN run...')
    dqn_run = DQNRun(
        agent, collect_env, q_net,
        eval_runner=eval_runner,
        create_video_fn=create_video_renderer(procgen_env_name),
        video_freq=1,
        train_steps_per_epoch=train_steps_per_epoch,
        num_epochs=500,
        experience_batch_size=64,
        procgen_env_name=procgen_env_name
    )

    dqn_run.execute()


if __name__ == '__main__':
    main()
