# src/models/linear_model.py

import numpy as np
from config import config
from src.models.abstract_model import AbstractModel

class LinearModel(AbstractModel):
    """
    A simple linear model that predicts the next price of a token based on:
    - Current price
    - Summation of fee parameters
    - Summation of reward parameters
    - Volume
    - Liquidity
    - Circulation
    - Noise (from config.NOISE_STD)
    
    This model does not learn online; it simply applies a fixed formula.
    """

    def __init__(self):
        """
        Initialize any constants or hyperparameters. For this basic example,
        we do not store separate coefficients in the class because they
        are hard-coded in the predict() function.
        """
        super().__init__()  # Not strictly necessary here, but good practice.

    def predict(self, P, params, volume, liquidity, circulation):
        """
        Predict the next price for a single token using a linear formula.
        
        :param P: Current token price.
        :param params: Dictionary with keys "fees" and "rewards", each itself a dict.
        :param volume: Market volume (float).
        :param liquidity: Market liquidity (float).
        :param circulation: Current token circulation (float).
        :return: Predicted next price (float).
        """
        # Sum up fee parameters
        fee_effect = sum(params["fees"].values())
        
        # Sum up reward parameters
        reward_effect = sum(params["rewards"].values())

        # Sample random noise
        noise = np.random.normal(0, config.NOISE_STD)

        # A simple example linear formula; adapt as needed
        P_next = (
            0.1               # base offset
            + 0.9 * P         # factor on current price
            + 0.05 * volume   # factor on volume
            + 0.05 * liquidity
            - 0.00001 * circulation
            + fee_effect
            + reward_effect
            + noise
        )
        return P_next

    def update(self, data):
        """
        For the linear model, we do not perform online training.
        This method is a no-op, but remains here to fulfill the AbstractModel interface.
        """
        pass
