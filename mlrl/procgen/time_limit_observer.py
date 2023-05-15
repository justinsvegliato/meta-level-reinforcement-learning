from mlrl.meta.meta_env import MetaEnv


class TimeLimitObserver:

    def __init__(self, env: MetaEnv, max_steps: int):
        self.env = env
        self.object_env = env.object_env
        self.metrics = env.object_level_metrics
        self.max_steps = max_steps

    def __call__(self, obs, a, r, done, info):
        if not done and self.metrics.n_steps >= self.max_steps:
            # -1 action forces termination
            # https://github.com/openai/procgen/blob/5e1dbf341d291eff40d1f9e0c0a0d5003643aebf/procgen/src/game.cpp#L124
            self.object_env.step(-1)
            self.metrics(None, None, [], True, None)  # register episode termination with metrics
            info['time_limit_reached'] = True
