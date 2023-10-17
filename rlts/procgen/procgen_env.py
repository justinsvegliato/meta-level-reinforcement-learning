from typing import Optional

from procgen import ProcgenGym3Env
from gym3 import ExtractDictObWrapper
from tf_agents.environments.py_environment import PyEnvironment
from rlts.utils.env_wrappers import ImagePreprocessWrapper, FrameStack
from rlts.utils.procgen_gym3_wrapper import ProcgenGym3Wrapper


def make_vectorised_procgen(
        config: dict,
        procgen_env_name: str = None,
        seed: Optional[int] = None,
        n_envs: Optional[int] = 64) -> PyEnvironment:

    env_name = procgen_env_name or config.get('env', 'coinrun')
    action_repeats = config.get('action_repeats', 4)
    frame_stack = config.get('frame_stack', 0)
    grayscale = config.get('grayscale', True)

    procgen_gym3 = ExtractDictObWrapper(ProcgenGym3Env(
        num=n_envs,
        num_threads=min(n_envs, 32),
        env_name=env_name,
        use_backgrounds=False,
        restrict_themes=True,
        rand_seed=seed,
        distribution_mode='easy'), key='rgb')

    wrapped_procgen_gym3 = ProcgenGym3Wrapper(procgen_gym3, action_repeats=action_repeats)
    env = ImagePreprocessWrapper(wrapped_procgen_gym3, grayscale=grayscale)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)

    env.reset()

    return env
