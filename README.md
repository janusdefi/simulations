 # Janus Protocol Simulation
 
 The Janus Protocol Simulation project provides a framework to model a two-token flatcoin system (Alpha and Omega) using a rule-based Nudge Model.
 The simulation incorporates key economic parameters—such as fees, rewards, collateral, and supply dynamics—and uses PID control to adjust incentive parameters in order to nudge token prices toward a target peg while dynamically managing token circulation (minting and burning).
 
  Overview
 
 The Janus Protocol is designed to achieve price stability and manage risk by leveraging two tokens:
 - **Alpha**: The primary token, whose price is maintained near a target value (e.g., \$1.0).
 - **Omega**: The secondary token that absorbs market volatility and risk.
 
 Our simulation framework features:
 - **Multi-token simulation**: Both Alpha and Omega are simulated with their own parameters.
 - **Nudge Model**: A rule-based model that nudges token prices toward their target and updates supply using minting and burning rules.
 - **PID Control**: Fee and reward parameters are adjusted based on the error between the current price and the target price, with clamping to prevent runaway values.
 - **Collateral Management**: Each token has a collateral value that influences both its price and supply adjustments.
 - **Supply Dynamics**: The model mints new tokens when fees/rewards are high and burns tokens if supply exceeds a maximum threshold or if the system is undercollateralized.
 - **Price Clamping**: The simulation ensures that token prices never fall below zero (or a specified positive floor).
 - **Visualization**: The simulation produces plots showing the evolution of token prices and circulation over time.
 - **Optional Database Storage**: Simulation results can be optionally stored in an SQLite database for further analysis.
 
  Folder Structure
 
 The project is organized as follows:
 
 ```
 JanusProtocol/
 ├── config/
 │   ├── __init__.py
 │   └── config.py          # Global configuration for tokens, PID parameters, etc.
 ├── docs/
 │   └── README.md          # This file (documentation)
 ├── schema/
 │   └── schema.sql         # (Optional) SQL schema for database storage
 ├── src/
 │   ├── __init__.py
 │   ├── main.py            # Main entry point to run the simulation
 │   ├── simulation.py      # Contains the simulation loop
 │   ├── controllers/
 │   │   ├── __init__.py
 │   │   └── controller.py  # PID controller and gradient computation functions
 │   ├── database/
 │   │   ├── __init__.py
 │   │   └── database.py    # Functions to initialize and store data in SQLite
 │   └── models/
 │       ├── __init__.py
 │       ├── abstract_model.py  # Abstract base model interface
 │       ├── nudge_model.py     # Nudge Model: updates both price and supply
 │       ├── linear_model.py    # (Optional) Linear model implementation
 │       └── nn_model.py        # (Optional) Neural network model implementation
 ├── requirements.txt       # Python dependencies
 └── README.md              # This documentation file
 ```
 
  Installation
 
 1. **Clone the Repository**
 
    ```bash
    git clone https://github.com/yourusername/JanusProtocol.git
    cd JanusProtocol
    ```
 
 2. **Install Dependencies**
 
    Use Python 3.7 or later. Install the required packages with pip:
 
    ```bash
    pip install -r requirements.txt
    ```
 
    *Note*: If you are not using the neural network model, you can remove the `torch` dependency from `requirements.txt`.
 
  Configuration
 
 All simulation parameters are defined in `config/config.py`. This includes:
 - **MODEL_TYPE**: Set to `"nudge"` (alternatively `"linear"` or `"nn"`).
 - **TIME_STEPS**: Number of simulation iterations.
 - **LEARNING_RATE**: Used for updating fee and reward parameters via PID.
 - **TOKENS**: A dictionary that defines initial parameters for each token. For example:
 
    ```python
    TOKENS = {
        "alpha": {
            "initial_price": 0.1,
            "initial_circulation": 1000000,
            "collateral": 100000,
            "fees": {"minting": 0.2, "staking": 0.1, "transfers": 0.05},
            "rewards": {"minting": 0.4, "staking": 0.6, "transfers": 0.3},
            "target_price": 1.0,
            "max_supply": 2000000
        },
        "omega": {
            "initial_price": 0.1,
            "initial_circulation": 500000,
            "collateral": 50000,
            "fees": {"minting": 0.1, "staking": 0.05, "transfers": 0.02},
            "rewards": {"minting": 0.2, "staking": 0.3, "transfers": 0.15},
            "target_price": 1.0,
            "max_supply": 1500000
        }
    }
    ```
 
 - **FEES_COEFS and REWARDS_COEFS**: Coefficients for computing gradients during PID updates.
 - **PID Parameters**: `K_P`, `K_I`, and `K_D`.
 - **NOISE_STD**: Standard deviation for the random noise term.
 - **Database Options**: Set `STORE_RESULTS` to `False` if you do not wish to store simulation outputs.
 
  Running the Simulation
 
 To run the simulation, execute the main script:
 
 ```bash
 python src/main.py
 ```
 
 This will:
 - Initialize the tokens (Alpha and Omega) with parameters defined in the configuration.
 - Update fee and reward parameters using PID control.
 - Use the Nudge Model to compute new token prices and circulation values at each timestep.
 - Log and plot the evolution of token prices and circulation over time.
 
  Simulation Output
 
 The simulation produces two primary plots:
 - **Price Plot**: Shows how token prices (Alpha and Omega) evolve over time.
 - **Circulation Plot**: Displays the dynamics of token supply (circulation) over time.
 
 These visualizations help to analyze the performance of the Janus Protocol under various economic conditions and parameter settings.
 
  Customization and Tuning
 
 - **Nudge Model Parameters**: Adjust coefficients in `src/models/nudge_model.py` to fine-tune the price and supply dynamics.
 - **PID Controller Settings**: Modify `K_P`, `K_I`, and `K_D` in `config/config.py` to affect how fees and rewards are updated.
 - **Mint/Burn Rules**: The supply update logic in the Nudge Model can be customized. For example, you can change the minting factors or the burn percentages.
 
  Contact
 
 For questions or feedback, please contact:
 **Stylianos Kampakis, PhD**  
 Email: [stelios@janusdefi.com](mailto:stelios@janusdefi.com)
