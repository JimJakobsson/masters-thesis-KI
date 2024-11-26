#time now
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

from DatabaseReader import DatabaseReader

class ServerConnectionIPT1(DatabaseReader):
    @staticmethod
    def read_table():
        params = {
            'Trusted_Connection': 'Yes',
            'Driver': '{ODBC Driver 17 for SQL Server}',
            'Server': 'kosmos',
            'Database': 'SATSA_ARKIV'
        }

        # Create a connection string
        conn_str = ';'.join([f"{k}={v}" for k, v in params.items()])

        try:
            # Create a SQLAlchemy engine
            engine = create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str}")
            
            print("Connection successful")
            
            # Define table names
            ipt1_table = "archive.ipt1_20130909"
            mortality_table = "archive.resp_0121_mortality2022"

            # Query to join ipt1_table and mortality_table
            query = f"""
            SELECT i.*, m.birthdate1, m.death_yrmon, m.age_death
            FROM {ipt1_table} i
            JOIN {mortality_table} m
            ON i.TWINNR = m.TWINNR
            """
            print("Executing query...")
            combined_tables = pd.read_sql_query(query, engine)
      
            #write to file
            #combined_tables.to_csv('combined_tables.csv', index=False)
            #combined_tables = pd.read_csv('/nfs/home/jimjak/Master-Thesis/misc/combined_tables.csv')
            
            print(f"Query executed successfully. DataFrame shape: {combined_tables.shape}")
            return combined_tables
        
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Close the connection
            if 'engine' in locals():
                engine.dispose()

