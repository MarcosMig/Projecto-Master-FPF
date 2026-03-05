"""
Module responsible for saving and loading data to parquet files.
"""

import os
import pandas as pd

from .constants import CAMPOS_DIR

def save_field_to_parquet(df_new, filename=f"{CAMPOS_DIR}/fields_database.parquet"):
    """
    Appends new field data to the parquet database.
    Deduplicates by 'session_id' to avoid repeated entries.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Exists read old data and add new row
    if os.path.exists(filename):
        df_old = pd.read_parquet(filename)
        df_final = pd.concat([df_old, df_new]).drop_duplicates(
            subset=['BL_lat', 'BR_lat', 'TL_lat', 'TR_lat'], keep='last'  # keep='last' so new data overwrites old
        )

    else:
        df_final = df_new

    df_final.to_parquet(filename, index=False)



def read_field_from_parquet(filename=f"{CAMPOS_DIR}/fields_database.parquet"):
    """
    Appends new field data to the parquet database.
    Deduplicates by 'session_id' to avoid repeated entries.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Exists read old data and add new row
    if os.path.exists(filename):
        df = pd.read_parquet(filename)

    else:
        return pd.DataFrame()

    return df
