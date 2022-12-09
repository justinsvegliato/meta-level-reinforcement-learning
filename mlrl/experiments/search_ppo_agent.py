from mlrl.meta.meta_env import mask_token_splitter
from mlrl.networks.search_actor_nets import create_action_distribution_network
from mlrl.networks.search_transformer import SearchTransformer
from mlrl.networks.search_value_net import create_value_network
from mlrl.networks.search_actor_rnn import ActionSearchRNN
from mlrl.networks.search_value_rnn import ValueSearchRNN

import silence_tensorflow.auto  # noqa
import tensorflow as tf
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.train.utils import train_utils
from tf_agents.networks.mask_splitter_network import MaskSplitterNetwork
from tf_agents.train.utils import spec_utils


def create_search_ppo_agent(env, config, train_step=None):

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
        search_transformer = SearchTransformer(**network_kwargs)
        value_net = create_value_network(observation_tensor_spec,
                                         search_transformer=search_transformer)

        actor_net = create_action_distribution_network(observation_tensor_spec['search_tree_tokens'],
                                                       action_tensor_spec,
                                                       search_transformer=search_transformer)

        actor_net = MaskSplitterNetwork(mask_token_splitter,
                                        actor_net,
                                        input_tensor_spec=observation_tensor_spec,
                                        passthrough_mask=True)

    train_step = train_step or train_utils.create_train_step()

    return PPOAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        actor_net=actor_net,
        value_net=value_net,
        optimizer=tf.keras.optimizers.Adam(1e-5),
        greedy_eval=False,
        importance_ratio_clipping=0.2,
        train_step_counter=train_step,
        compute_value_and_advantage_in_train=False,
        update_normalizers_in_train=False,
        normalize_observations=False,
        use_gae=False,
        use_td_lambda_return=False,
        discount_factor=0.99,
        num_epochs=1,  # deprecated param
    )
