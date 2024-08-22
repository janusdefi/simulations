import numpy as np
import matplotlib.pyplot as plt
import random

class TokenSimulation:
    def __init__(self, initial_supply_token1, initial_supply_token2):
        self.initial_supply_token1 = initial_supply_token1
        self.initial_supply_token2 = initial_supply_token2
        self.supply_token1 = initial_supply_token1
        self.supply_token2 = initial_supply_token2
        self.price_token1 = [1.0]
        self.price_token2 = [1.0]
        self.total_supply_token1 = [initial_supply_token1]
        self.total_supply_token2 = [initial_supply_token2]
        self.demand_factor = 1.0
        self.reward_rate = 0.01
        self.fee_rate = 0.01
        self.drift = 0.001  # Small positive drift term for price appreciation
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def simulate_price_change(self, alpha1=0.01, alpha2=0.01, beta=0.01):
        # Price influenced by supply, demand factor, and positive drift
        supply_factor1 = self.supply_token1 / self.initial_supply_token1
        supply_factor2 = self.supply_token2 / self.initial_supply_token2
        
        price_token1 = self.price_token1[-1] * (1 + np.random.randn() * alpha1 - supply_factor1 * beta + self.demand_factor * alpha2 + self.drift)
        price_token2 = self.price_token2[-1] * (1 + np.random.randn() * alpha1 - supply_factor2 * beta + self.demand_factor * alpha2 + self.drift)

        self.price_token1.append(price_token1)
        self.price_token2.append(price_token2)

    def distribute_rewards(self):
        reward_token1 = self.reward_rate * self.supply_token1
        reward_token2 = self.reward_rate * self.supply_token2
        self.supply_token1 += reward_token1
        self.supply_token2 += reward_token2

    def apply_fees(self):
        fees_token1 = self.fee_rate * self.supply_token1
        fees_token2 = self.fee_rate * self.supply_token2
        self.supply_token1 -= fees_token1
        self.supply_token2 -= fees_token2

    def update_demand_factor(self):
        # Simple model to update the demand factor
        self.demand_factor = 1 + np.random.randn() * 0.01

    def balance_prices_rule_based(self):
        # Rule-based mechanism to balance token prices
        price_token1 = self.price_token1[-1]
        price_token2 = self.price_token2[-1]
        
        # Detect rapid appreciation
        if len(self.price_token1) > 1:
            appreciation_rate_token1 = self.price_token1[-1] / self.price_token1[-2]
            appreciation_rate_token2 = self.price_token2[-1] / self.price_token2[-2]
        else:
            appreciation_rate_token1 = 1.0
            appreciation_rate_token2 = 1.0

        if price_token1 > 1.2 * price_token2:
            # Increase supply of token1 and increase rewards
            self.supply_token1 *= 1.05
            self.reward_rate = min(self.reward_rate * 1.05, 0.05)  # Cap reward rate to prevent runaway increase
        elif price_token2 > 1.2 * price_token1:
            # Increase supply of token2 and increase rewards
            self.supply_token2 *= 1.05
            self.reward_rate = min(self.reward_rate * 1.05, 0.05)  # Cap reward rate to prevent runaway increase

        # Reduce supply if there is rapid appreciation
        if appreciation_rate_token1 > 1.1:
            self.fee_rate = min(self.fee_rate * 1.1, 0.1)  # Increase fees to cap the maximum fee rate
        if appreciation_rate_token2 > 1.1:
            self.fee_rate = min(self.fee_rate * 1.1, 0.1)  # Increase fees to cap the maximum fee rate

    def balance_prices_rl_based(self):
        # Use Q-learning to balance token prices
        state = (round(self.supply_token1, 2), round(self.supply_token2, 2), round(self.price_token1[-1], 2), round(self.price_token2[-1], 2))
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]  # Initialize Q-values for actions: [increase_supply1, increase_supply2, increase_fee_rate, increase_reward_rate]

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1, 2, 3])
        else:
            action = np.argmax(self.q_table[state])

        # Perform the chosen action
        if action == 0:
            self.supply_token1 *= 1.02  # Smaller supply adjustment
        elif action == 1:
            self.supply_token2 *= 1.02  # Smaller supply adjustment
        elif action == 2:
            self.fee_rate = min(self.fee_rate * 1.05, 0.1)  # Smaller fee adjustment
        elif action == 3:
            self.reward_rate = min(self.reward_rate * 1.02, 0.05)  # Smaller reward adjustment

        next_state = (round(self.supply_token1, 2), round(self.supply_token2, 2), round(self.price_token1[-1], 2), round(self.price_token2[-1], 2))
        price_diff = abs(self.price_token1[-1] - self.price_token2[-1])
        if len(self.price_token1) > 1:
            depreciation_rate_token1 = self.price_token1[-1] / self.price_token1[-2]
            depreciation_rate_token2 = self.price_token2[-1] / self.price_token2[-2]
        else:
            depreciation_rate_token1 = 1.0
            depreciation_rate_token2 = 1.0

        # Reward function penalizes large price deviations and rapid depreciation
        reward = -price_diff - max(1 - depreciation_rate_token1, 0) - max(1 - depreciation_rate_token2, 0)

        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0, 0]

        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        )

    def run_simulation(self, num_steps, use_rl=False):
        for _ in range(num_steps):
            self.distribute_rewards()
            self.apply_fees()
            self.update_demand_factor()
            self.simulate_price_change()

            if use_rl:
                self.balance_prices_rl_based()
            else:
                self.balance_prices_rule_based()

            self.total_supply_token1.append(self.supply_token1)
            self.total_supply_token2.append(self.supply_token2)

    def plot_results(self):
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(self.price_token1, label='Token 1 Price')
        plt.plot(self.price_token2, label='Token 2 Price')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.title('Token Prices Over Time')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.total_supply_token1, label='Token 1 Total Supply')
        plt.plot(self.total_supply_token2, label='Token 2 Total Supply')
        plt.xlabel('Time Steps')
        plt.ylabel('Total Supply')
        plt.title('Total Supply of Tokens Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Parameters
initial_supply_token1 = 500000
initial_supply_token2 = 500000

# Create simulation
sim = TokenSimulation(initial_supply_token1, initial_supply_token2)

# Run simulation with rule-based approach
num_steps = 100
sim.run_simulation(num_steps)

# Plot results for rule-based approach
sim.plot_results()

# Run simulation with RL-based approach
sim.run_simulation(num_steps, use_rl=False)

# Plot results for RL-based approach
sim.plot_results()
