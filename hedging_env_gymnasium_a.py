import numpy as np
import gymnasium as gym
from gymnasium import spaces

class HedgingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, asset_price_model, D, T, num_steps, trading_cost_para,r,
                 L=100, strike_price=None, initial_holding_delta=False, mode="CF",
                  risk_averse_param=0.1, **kwargs):
        super(HedgingEnv, self).__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Actions bound in [0, 1]

        # Observation space adjustment based on mode
        if mode == "PL" or mode =="RA":
            # Including current_option_price and delta for 'PL' mode
            self.observation_space = spaces.Box(
                low=np.array([0, 80, 0, 0], dtype=np.float32),  # Adjusted to float32 for SB3 compatibility
                high=np.array([L, 120, num_steps, 20], dtype=np.float32),
                dtype=np.float32)
        else:
            # Basic setup for 'CF' mode
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0], dtype=np.float32),
                high=np.array([L, np.inf, num_steps], dtype=np.float32),
                dtype=np.float32)
            
        
        # parameters for price movement and pricing
        self.n = 0
        self.T = T
        self.D = D
        self.num_steps = num_steps
        self.mode = mode
        self.L = L
        self.r = r
        self.done = False
        self.asset_price_model = asset_price_model
        self.current_price = asset_price_model.get_current_price()
        self.option_price_model = kwargs.get('option_price_model', None)
        self.current_option_price = self.option_price_model.compute_option_price(self.n, self.current_price, self.asset_price_model.sigma)
        self.delta = self.option_price_model.compute_delta(0, self.current_price)
        self.strike_price = strike_price if strike_price else self.current_price
        
        # parameters for action and rewards
        self.initial_holding_delta = initial_holding_delta
        self.trading_cost_para = trading_cost_para
        self.risk_averse_param = risk_averse_param
        if initial_holding_delta:
            self.initial_holding = self.L*self.delta
        else:
            self.initial_holding = 0
        self.h = self.initial_holding

        # parameters for evaluating performance
        self.cost = 0

    def step(self, action):
        # Scale the action to the actual range of holdings
        action[0] = round(action[0],3) * self.L
        new_h = action[0]
        delta_h = new_h - self.h

        self.asset_price_model.compute_next_price()
        next_price = self.asset_price_model.get_current_price()
        self.n += 1
        self.done = self.n >= self.num_steps
        reward = self._compute_reward(new_h, next_price, delta_h)

        self.h = new_h
        self.current_price = next_price
        state = self.get_state()
        
        return state, reward, self.done, False,{"delta":self.delta}

    def _compute_reward(self, new_h, next_price, delta_h):
        if self.mode == "CF":
            return self._compute_cf_reward(new_h, next_price, delta_h)
        elif self.mode == "PL":
            return self._compute_pl_reward(new_h, next_price, delta_h)
        elif self.mode== "RA":
            return self._compute_ra_reward(new_h, next_price, delta_h)

    def _compute_cf_reward(self, new_h, next_price, delta_h):
        reward = -self.current_price * delta_h - self.trading_cost_para * np.abs(self.current_price * delta_h)

        new_option_price = self.option_price_model.compute_option_price(self.n, next_price, self.asset_price_model.sigma)
        new_option_delta = self.option_price_model.compute_delta(self.n, next_price)
        self.current_option_price = new_option_price
        self.delta = new_option_delta

        if self.done:
            asset_gain = next_price * new_h - self.trading_cost_para * np.abs(next_price * new_h)
            option_loss = self.L * max(0, next_price - self.strike_price)
            reward += asset_gain - option_loss

        return reward

    def _compute_pl_reward(self, new_h, next_price, delta_h):
        new_option_price = self.option_price_model.compute_option_price(self.n, next_price, self.asset_price_model.sigma)
        new_option_delta = self.option_price_model.compute_delta(self.n, next_price)
        reward_portfolio_value = self.L * ((self.current_option_price - new_option_price))\
                                        + new_h * (next_price - self.current_price)
        trading_cost = self.trading_cost_para*self.current_price*(abs(delta_h) + 0.01 * delta_h**2)
        #trading_cost = self.trading_cost_para*self.current_price*(abs(delta_h))
        
        self.cost += trading_cost
        self.current_option_price = new_option_price
        self.delta = new_option_delta
        return reward_portfolio_value - trading_cost
    
    def _compute_ra_reward(self, new_h, next_price, delta_h):
        delta_wealth = self._compute_pl_reward(new_h, next_price, delta_h)
        reward = delta_wealth - self.risk_averse_param/2*(delta_wealth**2)
        return reward


    def get_state(self):
        time_to_maturity = self.num_steps - self.n
        if self.mode == "PL"or self.mode =="RA":
            return np.array([self.h, self.current_price, time_to_maturity, self.current_option_price], dtype=np.float32)
        else:
            return np.array([self.h, self.current_price, time_to_maturity], dtype=np.float32)

    def reset(self):
        self.asset_price_model.reset()
        self.n = 0
        self.cost = 0
        self.done = False
        self.current_price = self.asset_price_model.get_current_price()
        self.current_option_price = self.option_price_model.compute_option_price(self.n, self.current_price, self.asset_price_model.sigma)
        self.delta = self.option_price_model.compute_delta(self.n, self.current_price)

        if self.initial_holding_delta:
            self.initial_holding = self.L*self.delta
        else:
            self.initial_holding = 0
        self.h = self.initial_holding

        return self.get_state(), {"delta":self.delta}

    def render(self, mode='human', close=False):
        return self.get_state()

    def close(self):
        pass  # For environments that might require cleanup
