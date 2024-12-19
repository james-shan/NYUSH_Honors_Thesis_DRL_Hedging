import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm
import warnings


class GenericOptionPriceModel(ABC):
    """
    Generic option price class. The use this with the gym-hedging environment, the option class needs to have
    a function called 'compute_option_price' that computes the option price for a given the asset price and

    """
    @abstractmethod
    def compute_option_price(self, *inputs):
        pass


class BSM(GenericOptionPriceModel):
    def __init__(self, strike_price, risk_free_interest_rate, volatility, T, dt):
        self.strike_price = strike_price
        self.risk_free_interest_rate = risk_free_interest_rate
        self.volatility = volatility
        self.T = T
        self.dt = dt

    def compute_option_price(self, n, asset_price, vol, mode="step"):
        if mode == "step":
            time_to_maturity = self.T - n * self.dt
        elif mode == "ttm":
            time_to_maturity = n
        else:
            raise ValueError("'mode' must be either 'step' or 'ttm'")

        if time_to_maturity < 1e-7:
            if time_to_maturity != 0.0:
                warnings.warn("'time_to_maturity' is smaller than 1e-7. This can cause numerical instability.")
            return max(0, asset_price - self.strike_price)
        d_1 = (np.log(asset_price / self.strike_price) + (self.risk_free_interest_rate + self.volatility**2 / 2)
               * time_to_maturity) / (self.volatility * np.sqrt(time_to_maturity))
        d_2 = d_1 - self.volatility * np.sqrt(time_to_maturity)
        PVK = self.strike_price * np.exp(-self.risk_free_interest_rate * time_to_maturity)
        option_price = norm.cdf(d_1) * asset_price - norm.cdf(d_2) * PVK
        return option_price



    def compute_delta(self, n, asset_price):
        time_to_maturity = self.T - n * self.dt
        d_1 = (np.log(asset_price / self.strike_price) + (self.risk_free_interest_rate + self.volatility**2 / 2)
               * time_to_maturity) / (self.volatility * np.sqrt(time_to_maturity))
        delta = norm.cdf(d_1)
        return delta
    


    
    
class BSMSABR(BSM):
    def __init__(self, strike_price, risk_free_interest_rate, T, dt, v, rho):
        super().__init__(strike_price, risk_free_interest_rate, None, T, dt)
        self.v = v
        self.rho = rho
        # must compute delta after computing option price to update volatility

    def sabr_volatility(self, S,T,vol, K, r,  volvol, rho,beta=1):
        F = S * np.exp(r * T)
        x = (F * K) ** ((1 - beta) / 2)
        y = (1 - beta) * np.log(F / K)
        A = vol / (x * (1 + y**2 / 24 + y ** 4 / 1920))
        B = 1 + T * (
            ((1 - beta) ** 2) * (vol * vol) / (24 * x**2)
            + rho * beta * volvol * vol / (4 * x)
            + volvol * volvol * (2 - 3 * rho * rho) / 24
        )
        Phi = (volvol * x / vol) * np.log(F / K)
        Chi = np.log((np.sqrt(1 - 2 * rho * Phi + Phi * Phi) + Phi - rho) / (1 - rho))

        SABRIV = np.where(F == K, vol * B / (F ** (1 - beta)), A * B * Phi / Chi)
        return SABRIV

    def compute_option_price(self, n, asset_price, vol, mode="step"):
        time_to_maturity = self.T - n * self.dt
        self.volatility = self.sabr_volatility(S=asset_price, T = time_to_maturity, vol = vol, K=self.strike_price,
                                               r=self.risk_free_interest_rate, volvol=self.v, rho=self.rho)
        return super().compute_option_price(n, asset_price, mode)

