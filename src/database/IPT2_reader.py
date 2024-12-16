from pathlib import Path
from typing import Optional, Union

import pandas as pd
from database.join_config import JoinConfig
from .reader import DatabaseReader

class IPT1Reader(DatabaseReader):
    """Specific reader for IPT1 data"""

    def read_ipt1_data(self, 
                        ipt2_table: str = "archive.ipt2",
                        mortality_table: str = "archive.resp_0121_mortality2022",
                        use_cache: bool = False,
                        cache_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Read and join IPT1 and mortality data.
        
        Args:
            ipt1_table: Name of IPT1 table
            mortality_table: Name of mortality table
            cache_path: Optional path to cache file
            
        Returns:
            DataFrame with joined data
        """
        joins = [
            JoinConfig(
                table=mortality_table,
                join_conditions=["m.TWINNR = j1.TWINNR"],
                #These are the columns we want to select from the mortality table
                columns=['birthdate1', 'death_yrmon'] 
            )
        ]
        
        return self.read_joined_tables(main_table=ipt2_table, joins=joins, cache_path=cache_path, use_cache=use_cache)