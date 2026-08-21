import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Load .env explicitly from the project root directory
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

def get_db_engine() -> Engine:
    """Instantiates and returns a SQLAlchemy PostgreSQL engine."""
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")

    if not all([user, password, db_name]):
        raise ValueError("Missing database credentials. Please check your .env file.")

    connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(connection_url, pool_pre_ping=True)
    return engine