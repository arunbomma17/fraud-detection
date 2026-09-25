"""Unit tests for SQLite audit database management and transaction queries."""

import pytest

from src.database import DatabaseManager


@pytest.fixture
def temp_db() -> DatabaseManager:
    """Fixture providing an isolated in-memory SQLite database instance."""
    db_mgr = DatabaseManager(db_url="sqlite:///:memory:")
    db_mgr.init_db()
    return db_mgr


def test_database_initialization(temp_db: DatabaseManager) -> None:
    """Tests that tables initialize cleanly without errors."""
    stats = temp_db.get_transaction_stats()
    assert stats["total_transactions"] == 0
    assert stats["fraud_transactions"] == 0
    assert stats["genuine_transactions"] == 0


def test_insert_and_retrieve_transaction(temp_db: DatabaseManager) -> None:
    """Tests inserting records and querying them back."""
    rec_id = temp_db.insert_transaction(
        transaction_amount=250.75,
        transaction_time=4500.0,
        prediction="Fraud",
        fraud_probability=0.92,
        risk_level="HIGH",
        features={"Amount": 250.75, "V14": -5.5},
    )
    assert rec_id == 1

    records = temp_db.get_recent_transactions(limit=10)
    assert len(records) == 1
    assert records[0]["transaction_amount"] == 250.75
    assert records[0]["prediction"] == "Fraud"
    assert records[0]["risk_level"] == "HIGH"


def test_transaction_statistics_calculation(temp_db: DatabaseManager) -> None:
    """Tests aggregate KPI computation."""
    # Insert 3 genuine, 1 fraud
    temp_db.insert_transaction(50.0, 100.0, "Genuine", 0.05, "LOW")
    temp_db.insert_transaction(80.0, 200.0, "Genuine", 0.10, "LOW")
    temp_db.insert_transaction(120.0, 300.0, "Genuine", 0.25, "LOW")
    temp_db.insert_transaction(999.0, 400.0, "Fraud", 0.95, "HIGH")

    stats = temp_db.get_transaction_stats()
    assert stats["total_transactions"] == 4
    assert stats["genuine_transactions"] == 3
    assert stats["fraud_transactions"] == 1
    assert stats["fraud_rate"] == 25.0
    assert stats["high_risk_count"] == 1
    assert stats["low_risk_count"] == 3
    assert stats["total_amount_audited"] == 1249.0
