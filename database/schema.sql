-- E-Commerce Fraud Detection Database Schema
-- SQLite dialect

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_amount REAL NOT NULL,
    transaction_time REAL NOT NULL,
    prediction TEXT NOT NULL,
    fraud_probability REAL NOT NULL,
    risk_level TEXT NOT NULL,
    features_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for rapid query and dashboard filtering
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_prediction ON transactions (prediction);
CREATE INDEX IF NOT EXISTS idx_transactions_risk_level ON transactions (risk_level);
