import pandas as pd


df = pd.read_csv('P9_front_1 Labels.csv')

df = df.iloc[616:]

df.reset_index(drop=True, inplace=True)

df.to_csv('morningfirstone_datetime_cutted.csv', index=False)