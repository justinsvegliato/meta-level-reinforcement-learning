from typing import List, Optional, Tuple, Callable
import numpy as np
import gym
import procgen
import cv2

import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.utils import common
from tf_agents.drivers import py_driver
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.networks.sequential import Sequential

from mlrl.models.autoencoder import Autoencoder
from mlrl.utils.render_utils import save_video
from mlrl.runners.eval_runner import EvalRunner
from mlrl.runners.dqn_runner import DQNRun


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class QNet(tf.keras.Model):
    
    def __init__(self, autoencoder: Autoencoder, n_actions: int):
        super(QNet, self).__init__()
        self.ae = autoencoder
        self.n_actions = n_actions
        self.q = tf.keras.layers.Dense(n_actions)
    
    def call(self, x, training=False):
        z = self.ae.encode(x, training=training)
        return self.q(z, training=training)


def make_q_net(tf_env) -> QNet:
    ts = tf_env.current_time_step()
    n_actions = 1 + int(tf_env.action_spec().maximum)

    autoencoder = Autoencoder()
    q_net = QNet(autoencoder, n_actions)

    # build weights
    autoencoder(ts.observation)
    q_net(ts.observation)

    return q_net


def make_coinrun(n_envs: Optional[int] = None):
    if n_envs is not None:
        return BatchedPyEnvironment([make_coinrun() for _ in range(n_envs)])
    return GymWrapper(gym.make('procgen-coinrun-v0'))


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
    p = 1
    for i in range(1, int(np.ceil(np.sqrt(n)))):
        if n % i == 0:
            p = n / i
    q = n / p
    return int(p), int(q)


def render_env(env):
    """
    Renderers one of a batched env. Scales image for video quality.
    """
    cenv = env.env
    img = cenv.observe()[1][0]
    img = cv2.resize(img, (4*img.shape[0], 4*img.shape[1]),
                    interpolation=cv2.INTER_NEAREST)
    return img


def create_video_renderer() -> Callable:
    """
    Returns a callable that produces a video of 12 coinrun games
    played simulataneously by a given policy, saved to a given file path
    """
    n_video_envs = 12

    video_env = BatchedPyEnvironment([
        make_coinrun() for _ in range(n_video_envs)
    ])
    video_env.reset()


    p, q = get_grid_dim(n_video_envs)


    def render_fn(batched_env, *_):
        frames = [
            render_env(e) for e in batched_env.envs
        ]
        
        return np.concatenate([
            np.concatenate(frames[i*q:(i+1)*q]) for i in range(p)
        ], axis=1)

    def create_video(policy, video_file):
        frames = create_policy_eval_video_frames(
            policy, video_env, 
            render_fn=render_fn, steps=15*60*n_video_envs
        )
        return save_video(frames, video_file, fps=15)

    return create_video


def get_coinrun_tf_env():
    gym_env = make_coinrun()
    gym_env._action_is_discrete = False  # gym wrapper takes action from numpy array unless this is set
    tf_env = TFPyEnvironment(gym_env)
    tf_env.reset()
    return tf_env


def main():
    collect_env = make_coinrun(n_envs=16)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    train_step_counter = tf.Variable(0)

    tf_env = get_coinrun_tf_env()
    q_net = make_q_net(tf_env)

    agent = DdqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=Sequential([q_net]),
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        target_update_period=20000,
        train_step_counter=train_step_counter
    )
    agent.initialize()

    eval_runner = EvalRunner(
        16000, make_coinrun(n_envs=16), agent.policy)

    dqn_run = DQNRun(
        agent, collect_env, q_net,
        eval_runner=eval_runner,
        create_video_fn=create_video_renderer(),
        video_freq=5,
        train_steps_per_epoch=10000,
        num_epochs=500
    )

    dqn_run.execute()


if __name__ == '__main__':
    main()
