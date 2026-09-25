"""Database Module for E-Commerce Fraud Detection System.

Implements SQLite transaction logging and auditing via SQLAlchemy ORM:
- Transaction history recording
- Real-time KPI queries (counts, fraud percentage, average risk probability)
- Filtered historical queries for Streamlit audit tables
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    func,
)
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from src.config import DATABASE_PATH, DATABASE_URL, SCHEMA_PATH

logger = logging.getLogger(__name__)

Base = declarative_base()


class Transaction(Base):
    """SQLAlchemy ORM Model representing an audited e-commerce transaction."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_amount = Column(Float, nullable=False)
    transaction_time = Column(Float, nullable=False)
    prediction = Column(String(20), nullable=False)  # 'Genuine' or 'Fraud'
    fraud_probability = Column(Float, nullable=False)  # 0.0 to 1.0
    risk_level = Column(String(20), nullable=False)  # 'LOW', 'MEDIUM', 'HIGH'
    features_json = Column(Text, nullable=True)  # JSON serialized input dictionary
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Converts model instance to standard dictionary format."""
        return {
            "id": self.id,
            "transaction_amount": self.transaction_amount,
            "transaction_time": self.transaction_time,
            "prediction": self.prediction,
            "fraud_probability": round(self.fraud_probability, 4),
            "risk_level": self.risk_level,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }


class DatabaseManager:
    """Manages database connection lifecycle and transaction queries."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        self.db_url = db_url or DATABASE_URL
        self.engine = create_engine(self.db_url, echo=False, connect_args={"check_same_thread": False})
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)

    def init_db(self) -> None:
        """Initializes database schema, creating tables and indexes."""
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(self.engine)
        logger.info("Initialized SQLite database at: %s", self.db_url)

    def insert_transaction(
        self,
        transaction_amount: float,
        transaction_time: float,
        prediction: str,
        fraud_probability: float,
        risk_level: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Inserts an analyzed transaction record into the audit table."""
        session = self.Session()
        try:
            record = Transaction(
                transaction_amount=float(transaction_amount),
                transaction_time=float(transaction_time),
                prediction=str(prediction),
                fraud_probability=float(fraud_probability),
                risk_level=str(risk_level),
                features_json=json.dumps(features) if features else None,
                created_at=datetime.now(timezone.utc),
            )
            session.add(record)
            session.commit()
            return record.id
        except Exception as exc:
            session.rollback()
            logger.error("Failed to insert transaction record: %s", exc)
            raise
        finally:
            session.close()

    def get_recent_transactions(
        self,
        limit: int = 100,
        risk_filter: Optional[str] = None,
        prediction_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves recent audited transactions with optional filtering.

        Args:
            limit: Maximum records to return.
            risk_filter: Optional filter ('LOW', 'MEDIUM', 'HIGH').
            prediction_filter: Optional filter ('Genuine' or 'Fraud').

        Returns:
            List of transaction dictionaries.
        """
        session = self.Session()
        try:
            query = session.query(Transaction)
            if risk_filter and risk_filter.upper() != "ALL":
                query = query.filter(Transaction.risk_level == risk_filter.upper())
            if prediction_filter and prediction_filter.upper() != "ALL":
                query = query.filter(Transaction.prediction == prediction_filter.capitalize())

            records = query.order_by(desc(Transaction.created_at)).limit(limit).all()
            return [r.to_dict() for r in records]
        finally:
            session.close()

    def count_total_transactions(self) -> int:
        """Returns total number of recorded transactions."""
        session = self.Session()
        try:
            return session.query(func.count(Transaction.id)).scalar() or 0
        finally:
            session.close()

    def count_fraud_transactions(self) -> int:
        """Returns total number of transactions flagged as Fraud."""
        session = self.Session()
        try:
            return session.query(func.count(Transaction.id)).filter(Transaction.prediction == "Fraud").scalar() or 0
        finally:
            session.close()

    def count_genuine_transactions(self) -> int:
        """Returns total number of transactions classified as Genuine."""
        session = self.Session()
        try:
            return session.query(func.count(Transaction.id)).filter(Transaction.prediction == "Genuine").scalar() or 0
        finally:
            session.close()

    def calculate_fraud_percentage(self) -> float:
        """Calculates percentage of recorded transactions flagged as fraud."""
        total = self.count_total_transactions()
        if total == 0:
            return 0.0
        fraud = self.count_fraud_transactions()
        return round((fraud / total) * 100.0, 2)

    def get_transaction_stats(self) -> Dict[str, Any]:
        """Calculates comprehensive aggregate KPI statistics for dashboard display."""
        session = self.Session()
        try:
            total = session.query(func.count(Transaction.id)).scalar() or 0
            if total == 0:
                return {
                    "total_transactions": 0,
                    "fraud_transactions": 0,
                    "genuine_transactions": 0,
                    "fraud_rate": 0.0,
                    "avg_fraud_probability": 0.0,
                    "high_risk_count": 0,
                    "medium_risk_count": 0,
                    "low_risk_count": 0,
                    "total_amount_audited": 0.0,
                }

            fraud = session.query(func.count(Transaction.id)).filter(Transaction.prediction == "Fraud").scalar() or 0
            genuine = total - fraud
            avg_prob = session.query(func.avg(Transaction.fraud_probability)).scalar() or 0.0
            high_risk = session.query(func.count(Transaction.id)).filter(Transaction.risk_level == "HIGH").scalar() or 0
            med_risk = session.query(func.count(Transaction.id)).filter(Transaction.risk_level == "MEDIUM").scalar() or 0
            low_risk = session.query(func.count(Transaction.id)).filter(Transaction.risk_level == "LOW").scalar() or 0
            total_amount = session.query(func.sum(Transaction.transaction_amount)).scalar() or 0.0

            return {
                "total_transactions": total,
                "fraud_transactions": fraud,
                "genuine_transactions": genuine,
                "fraud_rate": round((fraud / total) * 100.0, 2),
                "avg_fraud_probability": round(float(avg_prob) * 100.0, 2),
                "high_risk_count": high_risk,
                "medium_risk_count": med_risk,
                "low_risk_count": low_risk,
                "total_amount_audited": round(float(total_amount), 2),
            }
        finally:
            session.close()


# Global default instance
db = DatabaseManager()


def init_db() -> None:
    """Convenience functional interface to initialize database."""
    db.init_db()


def insert_prediction(
    amount: float,
    time: float,
    prediction: str,
    probability: float,
    risk_level: str,
    features: Optional[Dict[str, Any]] = None,
) -> int:
    """Convenience function to log a prediction."""
    return db.insert_transaction(
        transaction_amount=amount,
        transaction_time=time,
        prediction=prediction,
        fraud_probability=probability,
        risk_level=risk_level,
        features=features,
    )
