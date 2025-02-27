-- schema.sql
CREATE TABLE IF NOT EXISTS simulation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_step INTEGER NOT NULL,
    token_name TEXT NOT NULL,
    price REAL NOT NULL,
    circulation REAL NOT NULL,
    fees TEXT NOT NULL,
    rewards TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
