# src/models/nudge_model.py

import numpy as np
from config import config
from src.models.abstract_model import AbstractModel

class NudgeModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.lambda_nudge = 0.01
        self.k_fee = 0.001
        self.k_reward = 0.001
        self.k_market = 0.001
        self.k_circ = 0.000000001
        self.k_collateral = 0.000000001

        # Minting/Burning parameters
        self.mint_factor_reward = 0.5
        self.mint_factor_fee = 0.2
        self.burn_surplus_pct = 0.1
        self.burn_deficit_pct = 0.01

    def predict(self, state, volume, liquidity):
        """
        Return (P_next, Q_next), the next price and next circulation.
        """
        P = state["price"]
        Q = state["circulation"]
        fees = state["fees"]
        rewards = state["rewards"]
        target = state["target"]
        collateral = state["collateral"]
        max_supply = state["max_supply"]

        # 1) Price Update (Nudge)
        error = target - P
        nudge = self.lambda_nudge * error

        fee_effect = self.k_fee * sum(fees.values())
        reward_effect = self.k_reward * sum(rewards.values())
        market_effect = self.k_market * (volume + liquidity)

        # Collateral effect
        if Q > collateral:
            coll_effect = -self.k_collateral * (Q - collateral)
        else:
            coll_effect = self.k_collateral * (collateral - Q)

        circ_effect = -self.k_circ * Q
        noise = np.random.normal(0, config.NOISE_STD)

        P_next = P + nudge + fee_effect + reward_effect + market_effect + coll_effect + circ_effect + noise

        # -------------- CLAMP THE PRICE ---------------
        # Option A: clamp to zero
        P_next = max(0.0, P_next)
        # Option B: clamp to a small positive floor
        # P_next = max(1e-3, P_next)

        # 2) Circulation Update
        total_fee = sum(fees.values())
        total_reward = sum(rewards.values())
        minted_tokens = self.mint_factor_reward * total_reward + self.mint_factor_fee * total_fee

        burn_amount = 0.0
        if Q > max_supply:
            surplus = Q - max_supply
            burn_amount += self.burn_surplus_pct * surplus

        if (P_next * Q) > collateral:
            deficit = (P_next * Q) - collateral
            burn_amount += self.burn_deficit_pct * deficit

        Q_next = Q + minted_tokens - burn_amount

        return P_next, Q_next

    def update(self, data):
        pass
