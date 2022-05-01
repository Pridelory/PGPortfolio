"""
Environment for our deep deterministic policy gradient method

functions
0. load dataset
1. env.reset() --> state s
2. env.step() --> reward, new state s'
"""
import gym
import gym.spaces
from pgportfolio.marketdata.datamatrices import DataMatrices
import numpy as np

eps = 1e-8

class DataGenerator(object):
    """ Provide data for the RL environment """

    def __init__(self, config, is_training):
        self.config = config
        self.train_config = config["training"]
        self.input_config = config["input"]
        self.__window_size = self.input_config["window_size"]
        self.__coin_number = self.input_config["coin_number"]
        self.__batch_size = self.train_config["batch_size"]
        self._matrix = DataMatrices.create_from_config(config)
        self.training_data = np.transpose(self._matrix.get_pure_training_data(), (1, 2, 0))
        self.testing_data = np.transpose(self._matrix.get_pure_testing_data(), (1, 2, 0))
        self.step = 0
        # self.steps = config["training"]["steps"]
        self.idx = 0
        self.is_training = is_training

    def _step(self):
        self.step += 1
        if self.is_training:
            # retrieve the data of a window
            obs = self.training_data[:, self.step:self.step + self.__window_size, :].copy()
            # used to compute the optimal function ans so on
            ground_truth_obs = self.training_data[:, self.step + self.__window_size:self.step + self.__window_size + 1, :].copy()
            done = self.step >= self.training_data.shape[1] - self.__window_size - 1
            # done = self.step >= self.steps
        else:
            # retrieve the data of a window
            obs = self.testing_data[:, self.step:self.step + self.__window_size, :].copy()
            # used to compute the optimal function ans so on
            ground_truth_obs = self.testing_data[:, self.step + self.__window_size:self.step + self.__window_size + 1,
                               :].copy()
            done = self.step >= self.testing_data.shape[1] - self.__window_size - 1
            # done = self.step >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        if self.is_training:
            # start somewhere randomly
            self.step = np.random.randint(low = 0, high = self.training_data.shape[1] - self.__window_size - 1)
            return self.training_data[:, self.step:self.step + self.__window_size, :].copy(), \
                   self.training_data[:, self.step + self.__window_size:self.step + self.__window_size + 1, :].copy()
        else:
            # self.step = np.random.randint(low=0, high=self.testing_data.shape[1] - self.__window_size - 1)
            self.step = 30
            return self.testing_data[:, self.step:self.step + self.__window_size, :].copy(), \
                   self.testing_data[:, self.step + self.__window_size:self.step + self.__window_size + 1, :].copy()

    def get_coin_number(self):
        return self.__coin_number

    def get_window_length(self):
        return self.__window_size

class PortfolioSim(object):
    """ Calculate transaction costs and rewards and so on """

    def __init__(self):
        self.trading_cost = 0.0025
        self.p0 = 0
        self.infos = []

    def _step(self, w1, y1):
        """
        Step Numbered equations are from https://arxiv.org/abs/1706.10059
        :param w1: new action of portfolio weights
        :param y1: price relative vector also called return
        :return:
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        # eq7
        # The weights of w1 at the end of the period
        dw1 = (y1 * w1) / (np.dot(y1, w1) + eps)

        # eq16
        # 1 - mu1 discounting rate
        mu1 = self.trading_cost * (np.abs(dw1 - w1)).sum()
        assert mu1 < 1.0, 'Cost is larger than current holding'

        # eq11
        # compute the total assets at the end of the period after making portfolio decision w1
        p1 = self.p0 * (1 - mu1) * np.dot(y1, w1)

        # eq3
        # rate of return
        ror = p1 / self.p0 - 1

        # eq4
        # log rate of return
        lror = np.log((p1 + eps) / (self.p0 + eps))
        # define reward in the context of RL
        reward = lror

        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": lror,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": ror,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = 1.0

class PortfolioEnv(gym.Env):
    """
    A true environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different financial products.
    """

    def __init__(self, config, is_training):
        self.src = DataGenerator(config, is_training)
        self.sim = PortfolioSim()

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(self.src.get_coin_number() + 1,), dtype=np.float32)  # include cash

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.src.get_coin_number(), self.src.get_window_length(),
                                                                               4), dtype=np.float32)
        self.infos = []

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        # normalise the action to (0, 1)
        action = np.clip(action, 0, 1)

        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]

        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action

        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.src.get_window_length(), observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/open)
        close_price_vector = observation[:, -1, 0]
        open_price_vector = observation[:, -1, 3]
        y1 = close_price_vector / open_price_vector
        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        return observation, reward, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        # self.sim.reset()
        for sim in self.sim:
            sim.reset()
        observation, ground_truth_obs = self.src.reset()
        cash_observation = np.ones((1, self.src.get_window_length(), observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        return observation, info

class MultiActionPortfolioEnv(PortfolioEnv):
    def __init__(self, config, model_name, is_training):
        super(MultiActionPortfolioEnv, self).__init__(config, is_training)
        self.model_name = model_name
        self.sim = [PortfolioSim() for _ in range(1)]
        self.infos = []

    def _step(self, action):
        """ Step the environment by a vector of actions

        Args:
            action: (num_models, num_stocks + 1)

        Returns:

        """
        assert action.ndim == 2, 'Action must be a two dimensional array with shape (num_models, num_stocks + 1)'
        # normalise just in case
        action = np.clip(action, 0, 1)
        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (np.sum(weights, axis=1, keepdims=True) + eps)
        # so if weights are all zeros we normalise to [1,0...]
        weights[:, 0] += np.clip(1 - np.sum(weights, axis=1), 0, 1)
        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(np.sum(weights, axis=1), np.ones(shape=(weights.shape[0])), 3,
                                       err_msg='weights should sum to 1. action="%s"' % weights)
        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.src.get_window_length(), observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/open)
        close_price_vector = observation[:, -1, 0]
        open_price_vector = observation[:, -1, 3]
        y1 = close_price_vector / open_price_vector

        rewards = np.empty(shape=(weights.shape[0]))
        info = {}
        current_info = {}
        dones = np.empty(shape=(weights.shape[0]), dtype=bool)
        for i in range(weights.shape[0]):
            reward, current_info, done2 = self.sim[i]._step(weights[i], y1)
            rewards[i] = reward
            info['return'] = current_info['return']
            dones[i] = done2

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs
        info['portfolio_change'] = (1 - current_info['cost']) * np.dot(y1, weights.reshape((12)))
        self.infos.append(info)

        return observation, rewards, np.all(dones) or done1, info

    # def _reset(self):
    #     self.infos = []
    #     for sim in self.sim:
    #         sim.reset()
    #     observation, ground_truth_obs = self.src.reset()
    #     cash_observation = np.ones((1, self.src.gert, observation.shape[2]))
    #     observation = np.concatenate((cash_observation, observation), axis=0)
    #     cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
    #     ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
    #     info = {}
    #     info['next_obs'] = ground_truth_obs
    #     return observation, info