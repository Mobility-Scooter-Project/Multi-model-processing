import pandas as pd

# Load the data from a CSV file, assuming it has no header
df = pd.read_csv('P9_front_1 Labels.csv', header=None)

rows_to_remove = [1292, 393, 216]  # Adjusted for zero-based index
df = df.drop(rows_to_remove)

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Add a header
df.columns = ['Label', 'Upper Label']  # Adjust based on your actual needed headers

# Assuming 'A' is the column name you want to replace and it's the first column
new_column_names = ['Timestamp' if name == 'A' else name for name in df.columns]
df.columns = new_column_names

# Save the modified DataFrame back to a CSV file
df.to_csv('P9_front_1_Labels.csv', index=False)
