from mlrl.experiments.ppo_maze import create_batched_maze_meta_envs
from mlrl.meta.retro_rewards_rewriter import RetroactiveRewardsRewriter

from unittest.mock import patch
import numpy as np

from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.trajectories.policy_step import PolicyStep


def test_retro_reward_rewriting_traj_handling():
    rewritten_trajs = []
    env, *_ = create_batched_maze_meta_envs(2, seed=0)
    env.reset()
    reward_rewriter = RetroactiveRewardsRewriter(env, rewritten_trajs.append)

    def step_env(actions):
        ts = env.current_time_step()
        policy_step = PolicyStep(action=np.array(actions, dtype=np.int32),
                                 state=(), info=())
        next_ts = env.step(policy_step.action)
        traj = Trajectory(step_type=ts.step_type,
                          observation=ts.observation,
                          action=policy_step.action,
                          reward=next_ts.reward,
                          next_step_type=next_ts.step_type,
                          policy_info=policy_step.info,
                          discount=next_ts.discount)

        reward_rewriter(traj)

        return traj

    # step the env with computational actions expanding the root node
    step_env([1, 1])

    assert len(reward_rewriter.trajectories) == 1, 'one trajectory should be stored'
    assert len(rewritten_trajs) == 0, 'no trajectories should be rewritten yet'

    traj_0_wrapper = reward_rewriter.trajectories[0]
    assert traj_0_wrapper.traj is not None, 'trajectory should be stored'

    with patch.object(traj_0_wrapper, 'rewrite_reward', wraps=traj_0_wrapper.rewrite_reward) as mock_rewrite:
        # step the env with one terminate action
        step_env([0, 1])

        assert mock_rewrite.call_count == 1, 'the first batch of the trajectory should be rewritten'
        args, _ = mock_rewrite.call_args
        assert args[0] == 0, 'the first batch of the trajectory should be rewritten'
        assert traj_0_wrapper.is_rewrite_handled(0), 'traj wrapper should know that the first batch was rewritten'
        assert not traj_0_wrapper.all_rewrites_handled(), 'traj wrapper should know that not all batches were rewritten'

    traj_1_wrapper = reward_rewriter.trajectories[1]
    # the trajectory should be rewritten when both have terminated
    assert len(rewritten_trajs) == 0, 'no trajectories should be rewritten yet'
    assert len(reward_rewriter.trajectories) == 2, 'two trajectory should be stored'

    with patch.object(traj_0_wrapper, 'rewrite_reward', wraps=traj_0_wrapper.rewrite_reward) as mock_rewrite_0,\
            patch.object(traj_1_wrapper, 'rewrite_reward', wraps=traj_1_wrapper.rewrite_reward) as mock_rewrite_1:
        # step the second env with one terminate action
        step_env([1, 0])

        assert mock_rewrite_0.call_count == 1, 'the second batch of the first trajectory should be rewritten'
        args, _ = mock_rewrite_0.call_args
        assert args[0] == 1, 'the second batch of the first trajectory should be rewritten'

        assert mock_rewrite_1.call_count == 1, 'the second batch of the second trajectory should be rewritten'
        args, _ = mock_rewrite_1.call_args
        assert args[0] == 1, 'the second batch of the second trajectory should be rewritten'
        # args, _ = mock_rewrite.call_args
        # assert args[0] == 1, 'the first batch of the trajectory should be rewritten'

    # the first trajectory should be rewritten when both have terminated
    assert len(rewritten_trajs) == 2, 'the first trajectory should be rewritten'
    assert len(reward_rewriter.trajectories) == 1, 'one trajectory should be stored'

    reward_rewriter.flush_all()
    assert len(rewritten_trajs) == 3, 'all trajectories should be rewritten'
    assert len(reward_rewriter.trajectories) == 0, 'no trajectories should be stored'
