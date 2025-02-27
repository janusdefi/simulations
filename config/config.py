# config/config.py

# Select the model to use: "nudge", "linear", or "nn"
MODEL_TYPE = "nudge"

# Simulation settings
TIME_STEPS = 100
LEARNING_RATE = 0.01

# Multi-token configuration.
# Each token is defined with its starting price, supply, collateral, fees, rewards, target price, and max supply.
TOKENS = {
    "alpha": {
        "initial_price": 0.1,               # Starting price at $0.1
        "initial_circulation": 1_000_000,     # Initial token supply
        "collateral": 100_000,              # e.g., $100,000 of collateral backing
        "fees": {
            "minting": 0.2,
            "staking": 0.1,
            "transfers": 0.05
        },
        "rewards": {
            "minting": 0.4,
            "staking": 0.6,
            "transfers": 0.3
        },
        "target_price": 1.0,                # Target price we aim to achieve (e.g. $1.0)
        "max_supply": 2_000_000             # Maximum desired token supply
    },
    "omega": {
        "initial_price": 0.1,
        "initial_circulation": 500_000,
        "collateral": 50_000,
        "fees": {
            "minting": 0.1,
            "staking": 0.05,
            "transfers": 0.02
        },
        "rewards": {
            "minting": 0.2,
            "staking": 0.3,
            "transfers": 0.15
        },
        "target_price": 1.0,
        "max_supply": 1_500_000
    }
}

# Coefficients for computing gradients (used with PID updates)
FEES_COEFS = {
    "alpha": {
        "minting": -0.2,
        "staking": -0.1,
        "transfers": -0.15
    },
    "omega": {
        "minting": -0.15,
        "staking": -0.1,
        "transfers": -0.2
    }
}

REWARDS_COEFS = {
    "alpha": {
        "minting": 0.3,
        "staking": 0.15,
        "transfers": 0.2
    },
    "omega": {
        "minting": 0.25,
        "staking": 0.1,
        "transfers": 0.15
    }
}

# PID Controller parameters
K_P = 0.1
K_I = 0.01
K_D = 0.05

# Noise level for simulation (standard deviation)
NOISE_STD = 0.5

# Database options (for future use; can disable by setting STORE_RESULTS to False)
STORE_RESULTS = False
DB_PATH = "simulation.db"
SCHEMA_PATH = "schema.sql"
