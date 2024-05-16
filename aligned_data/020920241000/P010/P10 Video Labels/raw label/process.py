import pandas as pd

# Load the data from a CSV file, assuming it has no header
df = pd.read_csv('P10_front_3 Labels.csv', header=None)

# Specify rows to remove
rows_to_remove = [8274, 8275, 9780, 9781, 10318, 10323, 10325, 10326, 10327, 10353, 10383, 10384, 10385, 10386, 10387, 10452]
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


df.to_csv('P10_front_3_Labels.csv', index=False)
