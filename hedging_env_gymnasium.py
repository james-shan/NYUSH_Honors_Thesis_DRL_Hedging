import numpy as np
import gymnasium as gym
from gymnasium import spaces

class HedgingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, asset_price_model, dt, T, num_steps=100, trading_cost_para=0.01,
                 L=100, strike_price=None, int_holdings=False, initial_holding=0.5, mode="CF", max_holding=1, **kwargs):
        super(HedgingEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)  # Actions bound in [-1, 1]

        # Observation space adjustment based on mode
        if mode == "PL":
            # Including current_option_price and delta for 'PL' mode
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, -1], dtype=np.float32),  # Adjusted to float32 for SB3 compatibility
                high=np.array([np.inf, np.inf, np.inf, np.inf, 1], dtype=np.float32),
                dtype=np.float32)
        else:
            # Basic setup for 'CF' mode
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
                dtype=np.float32)

        self.asset_price_model = asset_price_model
        self.current_price = asset_price_model.get_current_price()
        self.n = 0
        self.T = T
        self.dt = dt
        self.num_steps = num_steps
        self.mode = mode
        self.L = L
        self.initial_holding = initial_holding
        self.max_holding = max_holding  # Maximum holding to interpret the action correctly
        self.h = initial_holding
        self.int_holdings = int_holdings
        self.trading_cost_para = trading_cost_para
        self.done = False
        self.strike_price = strike_price if strike_price else self.current_price
        
        if mode == "PL":
            self.option_price_model = kwargs.get('option_price_model', None)
            assert self.option_price_model is not None, "If 'PL' is chosen, an option_price_model must be provided."
            self.current_option_price = self.option_price_model.compute_option_price(self.n, self.current_price)
            self.delta = self.option_price_model.compute_delta(0, self.current_price)

    def step(self, action):
        # Scale the action to the actual range of holdings
        delta_h = (action[0] * self.max_holding)
        if self.int_holdings:
            delta_h = round(delta_h)
        new_h = self.h + delta_h
        self.asset_price_model.compute_next_price()
        next_price = self.asset_price_model.get_current_price()
        self.n += 1
        self.done = self.n >= self.num_steps
        reward = self._compute_reward(new_h, next_price, delta_h)
        self.h = new_h
        self.current_price = next_price
        state = self.get_state()
        return state, reward, self.done, False,{}

    def _compute_reward(self, new_h, next_price, delta_h):
        if self.mode == "CF":
            return self._compute_cf_reward(new_h, next_price, delta_h)
        elif self.mode == "PL":
            return self._compute_pl_reward(new_h, next_price, delta_h)[0]  # Just the first element for simplicity

    def _compute_cf_reward(self, new_h, next_price, delta_h):
        reward = -self.current_price * delta_h - self.trading_cost_para * np.abs(self.current_price * delta_h)
        if self.done:
            asset_value = next_price * new_h - self.trading_cost_para * np.abs(next_price * new_h)
            payoff = self.L * max(0, next_price - self.strike_price)
            reward += asset_value - payoff
        return reward

    def _compute_pl_reward(self, new_h, next_price, delta_h):
        new_option_price = self.option_price_model.compute_option_price(self.n, next_price)
        reward_option_price = self.L * ((new_option_price - self.current_option_price) - new_h * (next_price - self.current_price))
        reward_trading_cost = -self.trading_cost_para * self.dt * (abs(delta_h) + 0.01 * delta_h**2)
        self.current_option_price = new_option_price
        return reward_option_price, reward_trading_cost

    def get_state(self):
        time_to_maturity = self.T - self.n * self.dt
        if self.mode == "PL":
            return np.array([self.h, self.current_price, time_to_maturity, self.current_option_price, self.delta], dtype=np.float32)
        else:
            return np.array([self.h, self.current_price, time_to_maturity], dtype=np.float32)

    def reset(self):
        self.asset_price_model.reset()
        self.n = 0
        self.done = False
        self.current_price = self.asset_price_model.get_current_price()
        self.h = self.initial_holding
        if self.int_holdings:
            self.h = round(self.h)
        if self.mode == "PL":
            self.current_option_price = self.option_price_model.compute_option_price(self.n, self.current_price)
            self.delta = self.option_price_model.compute_delta(0, self.current_price)
        return self.get_state(), {}

    def render(self, mode='human', close=False):
        return self.get_state()

    def close(self):
        pass  # For environments that might require cleanup
