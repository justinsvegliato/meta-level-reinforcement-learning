from rlts.meta.meta_env import mask_token_splitter
from rlts.networks.search_actor_nets import create_action_distribution_network
from rlts.networks.search_transformer import SearchTransformer
from rlts.networks.search_value_net import create_value_network
from rlts.networks.search_actor_rnn import ActionSearchRNN
from rlts.networks.search_value_rnn import ValueSearchRNN

import silence_tensorflow.auto  # noqa
import tensorflow as tf
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.train.utils import train_utils
from tf_agents.networks.mask_splitter_network import MaskSplitterNetwork
from tf_agents.train.utils import spec_utils

import os


def create_search_ppo_agent(env, config, train_step=None, return_networks=False):

    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(env))

    network_kwargs = {
        'n_heads': 3,
        'n_layers': 2,
        'd_model': 32,
    }

    use_lstms = config.get('n_lstm_layers', 0) > 0
    if use_lstms:
        value_net = ValueSearchRNN(observation_tensor_spec, **config)
        actor_net = ActionSearchRNN(observation_tensor_spec, **config)
    else:
        # search_transformer = SearchTransformer(**network_kwargs)
        value_net = create_value_network(observation_tensor_spec,
                                         search_transformer=SearchTransformer(**network_kwargs))

        actor_net = create_action_distribution_network(observation_tensor_spec['search_tree_tokens'],
                                                       action_tensor_spec,
                                                       search_transformer=SearchTransformer(**network_kwargs))

        actor_net = MaskSplitterNetwork(mask_token_splitter,
                                        actor_net,
                                        input_tensor_spec=observation_tensor_spec,
                                        passthrough_mask=True)

    train_step = train_step or train_utils.create_train_step()

    agent = PPOAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        actor_net=actor_net,
        value_net=value_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=config.get('learning_rate', 3e-4), epsilon=1e-5),
        greedy_eval=False,
        importance_ratio_clipping=0.1,
        train_step_counter=train_step,
        compute_value_and_advantage_in_train=False,
        update_normalizers_in_train=False,
        normalize_observations=False,
        use_gae=True,
        use_td_lambda_return=True,
        discount_factor=0.99,
        num_epochs=1,  # deprecated param
    )

    if return_networks:
        actor_model = tf.keras.Sequential([actor_net])
        actor_model(env.reset().observation)
        value_model = tf.keras.Sequential([value_net])
        value_model(env.current_time_step().observation)
        return agent, actor_model, value_model

    return agent


def load_ppo_agent(env, args, ckpt_dir: str = None):

    agent, actor_net, value_net = create_search_ppo_agent(env, args,
                                                          return_networks=True)

    actor_net.load_weights(os.path.join(ckpt_dir, 'actor_network'))
    value_net.load_weights(os.path.join(ckpt_dir, 'value_network'))

    # checkpoint = tf.train.Checkpoint(policy=agent.policy)

    # if ckpt_dir:
    #     file_prefix = os.path.join(ckpt_dir,
    #                                tf.saved_model.VARIABLES_DIRECTORY,
    #                                tf.saved_model.VARIABLES_FILENAME)

    #     checkpoint.restore(file_prefix)

    return agent
