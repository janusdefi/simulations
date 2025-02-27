# src/simulation.py

import numpy as np
import matplotlib.pyplot as plt
from config import config
from src.controllers.controller import PIDController, compute_gradients
from src.database import database

# Import the model based on config
if config.MODEL_TYPE == "nudge":
    from src.models.nudge_model import NudgeModel as BaseModel
elif config.MODEL_TYPE == "linear":
    from src.models.linear_model import LinearModel as BaseModel
elif config.MODEL_TYPE == "nn":
    from src.models.nn_model import NNModel as BaseModel
else:
    raise ValueError(f"Unknown MODEL_TYPE in config: {config.MODEL_TYPE}")

def simulate():
    # 1. Initialize token states from config
    tokens = {}
    for token_name, token_info in config.TOKENS.items():
        tokens[token_name] = {
            "price": token_info["initial_price"],
            "circulation": token_info["initial_circulation"],
            "collateral": token_info["collateral"],
            "fees": token_info["fees"].copy(),
            "rewards": token_info["rewards"].copy(),
            "target": token_info["target_price"],
            "max_supply": token_info["max_supply"]
        }

    # 2. Create model & PID for each token
    models = {}
    pids = {}
    for token_name in tokens:
        models[token_name] = BaseModel()
        pids[token_name] = PIDController(config.K_P, config.K_I, config.K_D)

    # 3. Prepare data for plotting
    prices_history = {token_name: [] for token_name in tokens}
    circulation_history = {token_name: [] for token_name in tokens}

    # 4. Initialize DB if storing results
    if config.STORE_RESULTS:
        database.init_db(config.DB_PATH, config.SCHEMA_PATH)

    # 5. Store initial state in DB if needed
    if config.STORE_RESULTS:
        for token_name, token_state in tokens.items():
            database.store_simulation_step(
                db_path=config.DB_PATH,
                time_step=0,
                token_name=token_name,
                price=token_state["price"],
                circulation=token_state["circulation"],
                fees=token_state["fees"],
                rewards=token_state["rewards"]
            )

    # Example volume/liquidity
    volume = 10.0
    liquidity = 5.0

    # 6. Main simulation loop
    for t in range(1, config.TIME_STEPS + 1):
        # A) For each token, we do:
        #    1) Update fees & rewards with PID + gradient
        #    2) Clamp them
        #    3) Let the nudge model compute next price & supply

        for token_name, token_state in tokens.items():
            current_price = token_state["price"]
            target_price = token_state["target"]
            error = current_price - target_price
            control_signal = pids[token_name].update(error)

            # Compute gradients
            grad_fees, grad_rewards = compute_gradients(
                P_next=current_price,
                target=target_price,
                fees_coefs=config.FEES_COEFS[token_name],
                rewards_coefs=config.REWARDS_COEFS[token_name]
            )

            # Update fees
            for prod in token_state["fees"]:
                token_state["fees"][prod] -= config.LEARNING_RATE * grad_fees.get(prod, 0)
                token_state["fees"][prod] += control_signal

            # Update rewards
            for prod in token_state["rewards"]:
                token_state["rewards"][prod] -= config.LEARNING_RATE * grad_rewards.get(prod, 0)
                token_state["rewards"][prod] += control_signal

            # Clamp fees/rewards (0 to 5)
            for prod in token_state["fees"]:
                token_state["fees"][prod] = max(0.0, min(5.0, token_state["fees"][prod]))
            for prod in token_state["rewards"]:
                token_state["rewards"][prod] = max(0.0, min(5.0, token_state["rewards"][prod]))

        # B) Compute next price & supply using the nudge model
        for token_name, token_state in tokens.items():
            model = models[token_name]
            P_next, circ_next = model.predict(token_state, volume, liquidity)
            token_state["price"] = P_next
            token_state["circulation"] = circ_next

        # C) Log data
        for token_name, token_state in tokens.items():
            prices_history[token_name].append(token_state["price"])
            circulation_history[token_name].append(token_state["circulation"])

            if config.STORE_RESULTS:
                database.store_simulation_step(
                    db_path=config.DB_PATH,
                    time_step=t,
                    token_name=token_name,
                    price=token_state["price"],
                    circulation=token_state["circulation"],
                    fees=token_state["fees"],
                    rewards=token_state["rewards"]
                )

    # 7. Plot results
    # Plot prices
    plt.figure(figsize=(10, 6))
    for token_name in tokens:
        plt.plot(prices_history[token_name], label=f'{token_name} Price')
    plt.title('Token Prices Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot circulation
    plt.figure(figsize=(10, 6))
    for token_name in tokens:
        plt.plot(circulation_history[token_name], label=f'{token_name} Circulation')
    plt.title('Tokens in Circulation Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Circulation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    simulate()
