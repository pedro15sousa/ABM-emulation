import sqlite3
import pickle
import pandas as pd

db_path = 'simulation.db'

# Connect to the SQLite database
conn = sqlite3.connect(db_path)

# Define the SQL query to load results
# This query assumes you want to join the 'jobs' and 'results' tables to fetch
# the parameter index, iteration, and associated result data for all completed jobs
query = '''
SELECT j.param_index, j.iteration, r.result_data
FROM jobs j
JOIN results r ON j.param_index = r.param_index AND j.iteration = r.iteration
WHERE j.status = 'completed'
'''

# Execute the query and load the results into a pandas DataFrame
df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# A helper function to deserialize the binary data
def deserialize(data):
    return pickle.loads(data)

# Apply the deserialize function to each row in the result_data column
df['result_data'] = df['result_data'].apply(lambda x: deserialize(x))

concatenated_df = pd.concat(df['result_data'].tolist(), ignore_index=True)

# Display the first few rows of the DataFrame to verify
print(len(concatenated_df))
print(concatenated_df.head(20))