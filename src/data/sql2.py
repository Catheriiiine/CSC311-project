import sqlite3
import pandas as pd


new_train_data = pd.read_csv('new_train_data.csv')
new_valid_data = pd.read_csv('new_valid_data.csv')
data = pd.read_csv('new_test_data.csv')
# question_correctness = pd.read_csv('question_correctness.csv')
# question_correctness1 = pd.read_csv('question_correctness1.csv')
# question_correctness2 = pd.read_csv('question_correctness2.csv')
#conn = sqlite3.connect(':memory:')
# new_train_data = new_train_data.drop(new_train_data.columns[0], axis=1)
# new_valid_data = new_valid_data.drop(new_valid_data.columns[0], axis=1)
# new_test_data = new_test_data.drop(new_test_data.columns[0], axis=1)

# sum = 0
# #for new_train_data in (new_train_data, new_valid_data, new_test_data):
#     # Identifying columns (assuming 'correctness' is the second to last and 'user_id' is the last column)
# user_id_col = new_train_data.columns[-2]
# correctness_col = new_train_data.columns[-1]
# subject_cols = new_train_data.columns[1:-2]  # All other columns are subject_id columns
#
# # Melting the DataFrame so each row contains a single subject_id for a user
# melted_data = new_train_data.melt(id_vars=[user_id_col, correctness_col], value_vars=subject_cols, var_name='subject_id', value_name='answered')
#
# # We only need rows where a subject was actually answered
# answered_data = melted_data[melted_data['answered'] == 1]
#
# # Calculating accuracy per user per subject
# accuracy_data = answered_data.groupby([user_id_col, 'subject_id'])[correctness_col].mean().reset_index()
#
# # Pivoting to get the desired matrix format: users as rows and subjects as columns
# accuracy_matrix = accuracy_data.pivot(index=user_id_col, columns='subject_id', values=correctness_col)
#
# # Filling missing values with 0.5
# accuracy_matrix.fillna(0.5, inplace=True)
#
# accuracy_matrix.reset_index(inplace=True)
#
# # Optionally, you might want to ensure that the subjects are in the correct order, you can sort the columns based on their name
# accuracy_matrix = accuracy_matrix.sort_index(axis=1)
#
# accuracy_matrix.to_sql('accuracy_matrix', conn, index=False, if_exists='replace')
# #train = load_train_sparse()
# query = ("""
#         SELECT *
#         FROM accuracy_matrix a LEFT JOIN question_correctness t ON a.user_id = t.user_id;
#         """)
#
# data = pd.read_sql_query(query, conn)
#
# data.to_csv('0.csv'.format(sum), index=False)
# sum += 1
# conn.close()
# sum = 0
# for data in (new_train_data, new_valid_data, new_test_data):
# Drop the question_id column as it's not needed for the calculation
data = data.drop(columns=['question_id'])

# Separate the subject columns (all columns except user_id and is_correct)
subject_columns = data.columns[:-2]

# Correcting the approach to calculate the correctness rate

# We'll use a more straightforward method to iterate over each subject column, calculate the rates, and then compile them

# Initialize an empty dataframe for storing the rate of correctness per user per subject
correctness_rate_df = pd.DataFrame(index=data['user_id'].unique(), columns=subject_columns)
correctness_rate_df.index.name = 'user_id'

# Calculate correctness rate for each subject
for subject in subject_columns:
    # Filter rows where the subject is involved
    subject_data = data[data[subject] == 1]

    # Calculate rate of correctness for this subject
    correctness_rate = subject_data.groupby('user_id')['is_correct'].mean()

    # Assign the calculated rate to the corresponding user_id and subject in the dataframe
    correctness_rate_df[subject] = correctness_rate

# Fill NaN values with 0 (for users with no data on specific subjects)
for index, row in correctness_rate_df.iterrows():
    subject_1_accuracy = row['0']
    # Iterate through all subjects for the row
    for subject in correctness_rate_df.columns:
        if pd.isna(row[subject]):
            correctness_rate_df.at[index, subject] = subject_1_accuracy

# Reset index to make user_id a column again
correctness_rate_final = correctness_rate_df.reset_index()
# correctness_rate_final.to_sql('correctness_rate_final', conn, index=False, if_exists='replace')
# #train = load_train_sparse()
# query = ("""
#         SELECT *
#         FROM correctness_rate_final a LEFT JOIN question_correctness1 t ON a.user_id = t.user_id;
#         """)

#data = pd.read_sql_query(query, conn)

correctness_rate_final.to_csv('2.csv', index=False)
# sum += 1
#conn.close()
#correctness_rate_final.to_csv('0.csv', index=False)