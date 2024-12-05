from typing import Optional, Union, List
import pandas as pd
from pathlib import Path
from .connection import DatabaseConnection
from .database_config import DatabaseConfig
from .query_builder import QueryBuilder, JoinConfig

class DatabaseReader:
    """Base class for database readers"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = DatabaseConnection(config)

    def read_table(self, table_name: str) -> pd.DataFrame:
        """
        Read table from database.
        
        Args:
            table_name: Name of table to read
            
        Returns:
            DataFrame of table contents
        """
        try: 
            engine = self.connection.connect()
            query = f"SELECT * FROM {table_name}"
            return pd.read_sql_query(query, engine)
        except Exception as e:
            print(f"Failed to read table: {str(e)}")
        finally:
            self.connection.dispose()
    
    def read_joined_tables(self, main_table: str, joins: List[JoinConfig], cache_path: Optional[Path] = None, use_cache: bool = False) -> pd.DataFrame:
        """
        Read joined tables from database.
        
        Args:
            main_table: Primary table name
            joins: List of join configurations
            where_clause: Optional WHERE clause
            cache_file: Optional path to cache file
            use_cache: Whether to use cache if available
            
        Returns:
            DataFrame of joined table contents
        """
        # Check cache first if enabled
        if use_cache and cache_path:
            print(f"Reading from cache file: {cache_path}")
            return pd.read_csv(cache_path)
        try: 
            query = QueryBuilder.build_join_query(main_table, joins)
            print("Executing query:", query)

            engine = self.connection.connect()
            df = pd.read_sql_query(query, engine)
            print(f"Query executed successfully. DataFrame shape: {df.shape}")

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
                print(f"Results cached to file: {cache_path}")
            return df
        except Exception as e:
            raise Exception(f"Failed to read joined tables: {str(e)}")
        finally:
            self.connection.dispose()