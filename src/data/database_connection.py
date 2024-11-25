from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from ..config import DatabaseConfig
# from .exceptions import ConnectionError

class DatabaseConnection:
    """Manages database connections"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine: Optional[Engine] = None
    
    def connect(self) -> Engine:
        """Create database connection"""
        try:
            conn_str = ';'.join([f"{k}={v}" for k, v in self.config.to_params().items()])
            self._engine = create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str}")
            return self._engine
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    def dispose(self) -> None:
        """Close database connection"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
