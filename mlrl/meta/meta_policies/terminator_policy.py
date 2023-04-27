from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories import policy_step
import tensorflow as tf


class TerminatorPolicy(PyPolicy):
    """ A meta-level policy that always instantly terminates. """

    def __init__(self, env: BatchedPyEnvironment):
        super().__init__(env.time_step_spec(), env.action_spec())
        self.batch_size = env.batch_size

    def _action(self, time_step, policy_state):
        action = tf.convert_to_tensor([0] * self.batch_size, dtype=self.action_spec.dtype)
        return policy_step.PolicyStep(action, policy_state, info=())
