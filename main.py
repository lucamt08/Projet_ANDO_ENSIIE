import pandas as pd

alz_df = pd.read_csv("alzheimer_data.csv")
print(alz_df.head())

non_diagnosed_count = len(alz_df.loc[alz_df['Diagnosis'] == 0].index)
diagnosed_count = len(alz_df.loc[alz_df['Diagnosis'] == 1].index)

print(f"Number of observations : {len(alz_df.index)}")
print(f"Non Diagnosed cases : {non_diagnosed_count}\nDiagnosed cases : {diagnosed_count}")

