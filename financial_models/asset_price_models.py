import numpy as np
from abc import ABC, abstractmethod


class GenericAssetPriceModel(ABC):
    @abstractmethod
    def get_current_price(self):
        pass

    @abstractmethod
    def compute_next_price(self, *action):
        pass

    @abstractmethod
    def reset(self):
        pass


class GBM(GenericAssetPriceModel):
    def __init__(self, mu=0, dt=0.1, s_0=100, sigma=0.2):
        self.mu = mu
        self.dt = dt
        self.s_0 = s_0
        self.sigma = sigma
        self.current_price = s_0

    def compute_next_price(self):
        i = np.random.normal(0, np.sqrt(self.dt))
        new_price = self.current_price * np.exp((self.mu - self.sigma ** 2 / 2) * self.dt
                   + self.sigma * i)
        self.current_price = new_price

    def reset(self):
        self.current_price = self.s_0

    def get_current_price(self):
        return self.current_price
    


class StochasticVolatilityModel(GenericAssetPriceModel):
    def __init__(self, mu=0, v=0.2, rho=-0.5, dt=0.1, s_0=100, sigma_0=0.2):
        self.mu = mu
        self.v = v
        self.rho = rho
        self.dt = dt
        self.s_0 = s_0
        self.sigma_0 = sigma_0
        self.sigma = sigma_0
        self.current_price = s_0

    def compute_next_price(self):
        dw1 = np.random.normal(0, np.sqrt(self.dt))
        dw2 = self.rho * dw1 + np.sqrt(1 - self.rho ** 2) * np.random.normal(0, np.sqrt(self.dt))

        new_price = self.current_price * np.exp((self.mu - self.sigma ** 2 / 2) * self.dt
                     + self.sigma * dw1)
        self.current_price = new_price

        new_sigma = self.sigma * np.exp(-(self.v ** 2 / 2) * self.dt + self.v * dw2)
        self.sigma = new_sigma


    def reset(self):
        self.current_price = self.s_0
        self.sigma = self.sigma_0

    def get_current_price(self):
        return self.current_price

    

