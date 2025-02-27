# Janus Protocol Simulation

This repository implements a flexible simulation framework for the Janus Protocol. The protocol dynamically adjusts token prices by optimizing various parameters (e.g., fees and rewards for multiple products such as minting, staking, and transfers) and tracks state variables such as tokens in circulation. Additionally, simulation results are stored in a local SQLite database using a schema defined in an external SQL file.

## File Structure

- **main.py**: Entry point to run the simulation.
- **abstract_model.py**: Defines an abstract model interface.
- **linear_model.py**: Implements a simple linear price prediction model.
- **nn_model.py**: Implements a neural network model for price prediction (using PyTorch).
- **controller.py**: Contains the PID controller (with detailed documentation) and gradient descent update functions.
- **database.py**: Provides functions to initialize and store simulation results in a SQLite database.
- **schema.sql**: SQL file defining the database schema for simulation results.
- **simulation.py**: Runs the simulation, selecting the model based on configuration, handling multiple products, and storing results if enabled.
- **config.py**: Configuration file with simulation parameters, model settings, coefficients, and database options.
- **requirements.txt**: Python dependencies.
- **README.md**: This file.

## Requirements

- Python 3.x  
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
