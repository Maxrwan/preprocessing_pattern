""" 
    SAVE DATA
"""

def save_dataframe(df, output_path):
    df.to_parquet(output_path)
    return