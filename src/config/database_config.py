from dataclasses import dataclass
from typing import Dict, Optional
@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    server: str
    database: str
    driver: str = '{ODBC Driver 17 for SQL Server}'
    trusted_connection: str = 'Yes'
    
    def to_params(self) -> Dict[str, str]:
        """Convert config to connection parameters"""
        return {
            'Trusted_Connection': self.trusted_connection,
            'Driver': self.driver,
            'Server': self.server,
            'Database': self.database
        }