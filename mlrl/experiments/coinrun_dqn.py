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
from tf_agents.environments import ActionRepeat

from mlrl.models.autoencoder import Autoencoder
from mlrl.utils.render_utils import save_video
from mlrl.runners.eval_runner import EvalRunner
from mlrl.runners.dqn_runner import DQNRun
from mlrl.utils.env_wrappers import ImagePreprocessWrapper, FrameStack


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


class QNet(tf.keras.Model):
    
    def __init__(self, n_actions: int, enc_dim=64, n_channels=3):
        super(QNet, self).__init__()

        self.n_actions = n_actions
        self.enc_dim = enc_dim
        self.n_channels = n_channels

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (8, 8), 
                                   activation='relu', 
                                   input_shape=(64, 64, n_channels), strides=4),
            tf.keras.layers.Conv2D(64, (4, 4), activation='relu', strides=2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(enc_dim, activation='relu'),
        ], name='encoder')

        self.q = tf.keras.layers.Dense(n_actions, activation=None, name='q')

    def call(self, x, training=False):
        z = self.encoder(x, training=training)
        return self.q(z, training=training)

    def get_config(self):
        return {
            'n_actions': self.n_actions,
            'enc_dim': self.enc_dim,
            'n_channels': self.n_channels
        }


def create_dqn_agent(tf_env, train_steps_per_epoch=10000) -> QNet:
    ts = tf_env.current_time_step()
    n_actions = 1 + int(tf_env.action_spec().maximum)
    n_channels = tf_env.observation_spec().shape[-1]

    autoencoder = Autoencoder()
    q_net = QNet(n_actions=n_actions,
                 n_channels=n_channels,
                 enc_dim=512)

    target_q_net = QNet(n_actions=n_actions,
                        n_channels=n_channels,
                        enc_dim=512)

    # build weights
    q_net(ts.observation)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4)
    train_step_counter = tf.Variable(0)

    agent = DdqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=Sequential([q_net]),
        target_q_network=Sequential([target_q_net]),
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        target_update_period=10000,
        train_step_counter=train_step_counter
    )
    agent.initialize()

    return q_net, agent


# def make_coinrun(n_envs: Optional[int] = None):
#     if n_envs is not None:
#         return BatchedPyEnvironment([make_coinrun() for _ in range(n_envs)])
    
#     env = gym.make('procgen-coinrun-v0',
#                    use_backgrounds=False,
#                    restrict_themes=True,
#                    distribution_mode='easy')

#     return GymWrapper(env)

def make_coinrun(n_envs: Optional[int] = None):
    if n_envs is not None:
        return BatchedPyEnvironment([make_coinrun() for _ in range(n_envs)])

    base_env = gym.make('procgen-coinrun-v0',
                        use_backgrounds=False,
                        restrict_themes=True,
                        distribution_mode='easy')
    
    gym_wrapped = GymWrapper(base_env)
    # gym wrapper takes action from numpy array unless this is set
    gym_wrapped._action_is_discrete = False
    
    def render(mode='rgb_array'):
        img = gym_wrapped.current_time_step().observation
        img = cv2.resize(img, (4*img.shape[0], 4*img.shape[1]),
                         interpolation=cv2.INTER_NEAREST)
        return img

    gym_wrapped.render = render

    env = ActionRepeat(gym_wrapped, 4)
    env = FrameStack(ImagePreprocessWrapper(env), 4)

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


# def render_env(env):
#     """
#     Renderers one of a batched env. Scales image for video quality.
#     """
#     cenv = env.env
#     img = cenv.observe()[1][0]
#     img = cv2.resize(img, (4*img.shape[0], 4*img.shape[1]),
#                     interpolation=cv2.INTER_NEAREST)
#     return img

    
def render_env(env):
    """
    Renderers one of a batched env. Scales image for video quality.
    """
    img = env.render()
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
    ], multithreading=True)
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
    tf_env = TFPyEnvironment(gym_env)
    tf_env.reset()
    return tf_env


def main():
    n_collect_envs = 64
    collect_env = make_coinrun(n_envs=n_collect_envs)

    print('Collect env ts spec: ', collect_env.time_step_spec())
    
    train_steps_per_epoch = 5000

    tf_env = get_coinrun_tf_env()
    q_net, agent = create_dqn_agent(
        tf_env,
        train_steps_per_epoch=train_steps_per_epoch
    )

    n_eval_envs = 16
    eval_runner = EvalRunner(
        n_eval_envs * 1000, make_coinrun(n_envs=n_eval_envs), agent.policy)

    dqn_run = DQNRun(
        agent, collect_env, q_net,
        eval_runner=eval_runner,
        create_video_fn=create_video_renderer(),
        video_freq=1,
        train_steps_per_epoch=train_steps_per_epoch,
        num_epochs=500,
        experience_batch_size=64
    )

    dqn_run.execute()


if __name__ == '__main__':
    main()
