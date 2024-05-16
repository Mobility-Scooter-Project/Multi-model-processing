import pandas as pd

# Load the data from a CSV file, assuming it has no header
df = pd.read_csv('p13_front_3.mp4.csv', header=None)

# Specify rows to remove
rows_to_remove = [127, 4585, 10162, 10163, 10164, 10165, 10551, 10553, 10554, 10555, 10556, 10557, 10558, 10559, 10560, 10561, 10623, 10624, 10625, 10626]
rows_to_remove = [num - 1 for num in rows_to_remove]
df = df.drop(index=rows_to_remove, errors='ignore')

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Set new index starting from 0
df.index = range(0, len(df))

# Assign the new index to the 'Label' column
df['Label'] = df.index
df.iloc[:, 0] = df['Label']

df = df.iloc[:, :2]

# Rename columns
df.columns = [None, 'Label']

df.to_csv('P13_front_3_Labels.csv', index=False)
