from pgportfolio.ddpg.environment.ddpg_env import PortfolioEnv, DataGenerator
from pgportfolio.tools.configprocess import load_config
from pgportfolio.ddpg.model.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from pgportfolio.ddpg.model.actor import ActorNetwork
from pgportfolio.ddpg.model.critic import CriticNetwork
from pgportfolio.ddpg.model.ddpg import DDPG
from pgportfolio.ddpg.environment.ddpg_env import PortfolioEnv, MultiActionPortfolioEnv
from pgportfolio.ddpg.crypto_trading import StockActor, StockCritic, obs_normalizer, get_model_path, get_result_path, get_variable_scope

import numpy as np
import tensorflow as tf
import tflearn
import collections
import pandas as pd
import os

Result = collections.namedtuple("Result",
                                [
                                 "test_pv",
                                 "test_log_mean",
                                 "test_log_mean_free",
                                 "test_history",
                                 "config",
                                 "net_dir",
                                 "backtest_test_pv",
                                 "backtest_test_history",
                                 "backtest_test_log_mean"])
def test_model(env, model):
    observation, info = env.reset()
    done = False
    while not done:
        action = model.predict_single(observation)
        observation, _, done, _ = env.step(action)
    env.render()

def test_model_multiple(env, models):
    observations_list = []
    actions_list = []
    info_list = []
    observation, info = env.reset()
    done = False
    test_pc_vector = []
    csv_dir = './train_package/train_summary.csv'
    while not done:
        actions = []
        for model in models:
            actions.append(model.predict_single(observation))
        actions = np.array(actions)
        observation, _, done, info = env.step(actions)
        test_pc_vector.append(info['portfolio_change'])
        observations_list.append(observation)
        actions_list.append(actions)
        info_list.append(info)
    # df_performance = env.render()
    np_test_pc_vector = np.array(test_pc_vector, dtype=np.float32)
    result = Result(test_pv=[1],
                    test_log_mean=[1],
                    test_log_mean_free=[1],
                    test_history=[1],
                    config=[1],
                    net_dir=[1],
                    backtest_test_pv=[1],
                    backtest_test_history=[''.join(str(e) + ', ' for e in np_test_pc_vector)],
                    backtest_test_log_mean=[1])
    new_data_frame = pd.DataFrame(result._asdict()).set_index("net_dir")
    dataframe = new_data_frame
    dataframe.to_csv(csv_dir)
    return observations_list, info_list, actions_list

if __name__ == '__main__':
    config = load_config(1)
    n_classes = config['input']['coin_number'] + 1
    window_length = config['input']['window_size']
    batch_size = config['training']['batch_size']
    action_bound = 1.
    tau = 1e-3

    models = []
    model_names = []
    window_length_lst = [window_length]
    predictor_type_lst = ['cnn']
    use_batch_norm = True
    is_training = False

    # instantiate environment, 16 stocks, with trading cost, window_length 3, start_date sample each time
    for window_length in window_length_lst:
        for predictor_type in predictor_type_lst:
            name = 'DDPG_window_{}_predictor_{}'.format(window_length, predictor_type)
            model_names.append(name)
            tf.reset_default_graph()
            sess = tf.Session()
            tflearn.config.init_training_mode()
            action_dim = [n_classes]
            state_dim = [n_classes, window_length]
            variable_scope = get_variable_scope(window_length, predictor_type, use_batch_norm)
            with tf.variable_scope(variable_scope):
                actor = StockActor(sess, state_dim, action_dim, action_bound, 1e-4, tau, batch_size, predictor_type,
                                   use_batch_norm)
                critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                                     learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars(),
                                     predictor_type=predictor_type, use_batch_norm=use_batch_norm)
                actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

                model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
                summary_path = get_result_path(window_length, predictor_type, use_batch_norm)

                ddpg_model = DDPG(None, sess, actor, critic, actor_noise, obs_normalizer=obs_normalizer,
                                  config_file='config/cryptocurrency.json', model_save_path=model_save_path,
                                  summary_path=summary_path)
                ddpg_model.initialize(load_weights=True, verbose=True)
                models.append(ddpg_model)

    # evaluate the model with dates seen in training but from the second different stocks dataset
    env = MultiActionPortfolioEnv(config, model_names, is_training)

    observations_list, info_list, actions_list = test_model_multiple(env, models[:1])
    print(info_list)

