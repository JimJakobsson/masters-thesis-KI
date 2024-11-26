from typing import List, Optional

from .join_config import JoinConfig


class QueryBuilder:
    """Builds SQL queries"""

    def build_join_query(main_table: str, 
                        joins: List[JoinConfig], 
                        where_clause: Optional[str] = None) -> str:
        """
        Build SQL query with joins.
        
        Args:
            main_table: Primary table name
            joins: List of join configurations
            where_clause: Optional WHERE clause
        
        Returns:
            SQL query string
        """
        # Start with base query
        query = f"SELECT "
        
        # Add column selections
        columns = []
        columns.append("m.*")  # Select all from main table
        
        for i, join in enumerate(joins, 1):
            if join.columns:
                columns.extend([f"j{i}.{col}" for col in join.columns])
        
        query += ", ".join(columns)
        
        # Add FROM clause
        query += f"\nFROM {main_table} m"
        
        # Add JOINs
        for i, join in enumerate(joins, 1):
            conditions = join.join_conditions or []
            conditions_str = " AND ".join(conditions)
            query += f"\n{join.join_type} {join.table} j{i} ON {conditions_str}"
        
        # Add WHERE clause if provided
        if where_clause:
            query += f"\nWHERE {where_clause}"
        
        return query