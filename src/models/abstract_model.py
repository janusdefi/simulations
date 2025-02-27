# abstract_model.py

from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def predict(self, P, params, volume, liquidity, circulation):
        """
        Predict the next price given the current state.
        :param P: Current price.
        :param params: Dictionary of parameters (fees and rewards).
        :param volume: Market volume.
        :param liquidity: Market liquidity.
        :param circulation: Tokens in circulation.
        :return: Predicted next price.
        """
        pass

    @abstractmethod
    def update(self, data):
        """
        Optional: Update the model parameters (e.g., via training).
        :param data: Training data.
        """
        pass
