import pandas as pd
class Concat_DF:
    def __init__(self) -> None:
        pass 
    def Concat_DF_Sort(df1,df2):
    # Concatenate the DataFrames
        df_combined = pd.concat([df1, df2])

        # Sort by date
        df_combined = df_combined.sort_values(by='date')
        df_combined = df_combined.drop_duplicates(subset='date', keep='first')
        return df_combined