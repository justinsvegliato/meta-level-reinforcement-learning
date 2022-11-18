from tf_agents.utils.nest_utils import split_nested_tensors
from tf_agents.trajectories import trajectory
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment


class RetroactiveRewardsWriter:

    def __init__(self, env: PyEnvironment, add_batch: callable):
        self.env = env
        if isinstance(env, BatchedPyEnvironment):
            self.n_envs = env.batch_size
        else:
            self.n_envs = 1
        self.add_batch = add_batch
        self.trajectories = [[] for _ in range(self.n_envs)]

    def get_env_traj(self, traj: trajectory.Trajectory, i: int) -> trajectory.Trajectory:
        """
        Constructs a trajectory object specific to the ith environment in the batch.
        """
        policy_info = traj.policy_info[i] if traj.policy_info else ()
        observations = split_nested_tensors(traj.observation,
                                            self.env.observation_spec(),
                                            len(self.env.envs))
        return trajectory.Trajectory(
            step_type=traj.step_type[i],
            action=traj.action[i],
            observation=observations[i],
            next_step_type=traj.next_step_type[i],
            policy_info=policy_info,
            reward=traj.reward[i],
            discount=traj.discount[i]
        )

    def flush_trajectory(self, i: int):
        for traj in self.trajectories[i]:
            self.add_batch(traj)
        self.trajectories[i] = []

    def __call__(self, traj):
        for i in range(self.n_envs):
            env_traj = self.get_env_traj(traj, i)
            self.trajectory.appends(env_traj)
            
