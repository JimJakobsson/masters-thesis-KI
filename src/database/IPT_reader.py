from pathlib import Path
from typing import Optional, Union

import pandas as pd
from database.join_config import JoinConfig
from .reader import DatabaseReader

class IPTReader(DatabaseReader):
    """Specific reader for IPT data"""

    IPTs = {
        "IPT1": "archive.ipt1_20130909",
        "IPT2": "archive.ipt2_20130910",
        "IPT3": "archive.ipt3_20130909",
        # "IPT5": "archive.ipt4_20130909",
        # "IPT6": "archive.ipt6_20130909",
        # "IPT7": "archive.ipt7_20130909",
        # "IPT8": "archive.ipt8_20221208",
        # "IPT9": "archive.ipt9_20221208",
    }

    def read_ipt_data(self, 
                       ipt_table: str, 
                        # ipt_table: str = "archive.ipt1_20130909", #Default as ipt1 
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
        # cache_path = cache_path or Path("misc")

        #match the ipt_table to the correct table in the dictionary
        if ipt_table in self.IPTs:
            ipt_table = self.IPTs[ipt_table]
        else:
            raise ValueError(f"Unknown IPT table: {ipt_table}")

        cache_path = None
        print(f"current cache path: {cache_path}")
        joins = [
            JoinConfig(
                table=mortality_table,
                join_conditions=["m.TWINNR = j1.TWINNR"],
                #These are the columns we want to select from the mortality table
                columns=['birthdate1', 'death_yrmon'] 
            )
        ]
        
        return self.read_joined_tables(main_table=ipt_table, joins=joins, cache_path=cache_path, use_cache=use_cache)