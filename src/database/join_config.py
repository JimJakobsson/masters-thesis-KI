from dataclasses import dataclass
from typing import List


@dataclass
class JoinConfig:
    """Configuration for table joins"""
    table: str
    join_type: str = 'JOIN'
    join_conditions: List[str] = None
    columns: List[str] = None