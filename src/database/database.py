# src/database/database.py

import sqlite3
import json
import os
from config import config

def init_db(db_path, schema_path=config.SCHEMA_PATH):
    """
    Initialize the SQLite database using the provided SQL schema file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = f.read()
        cursor.executescript(schema)
    else:
        # Fallback if no schema file found
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_step INTEGER NOT NULL,
                token_name TEXT NOT NULL,
                price REAL NOT NULL,
                circulation REAL NOT NULL,
                fees TEXT NOT NULL,
                rewards TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    conn.commit()
    conn.close()

def store_simulation_step(db_path, time_step, token_name, price, circulation, fees, rewards):
    """
    Store a simulation step's results in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    fees_json = json.dumps(fees)
    rewards_json = json.dumps(rewards)
    cursor.execute('''
        INSERT INTO simulation_results (time_step, token_name, price, circulation, fees, rewards)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (time_step, token_name, price, circulation, fees_json, rewards_json))
    conn.commit()
    conn.close()
