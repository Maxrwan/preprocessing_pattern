""" 
    CLEAN DATA
"""

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df