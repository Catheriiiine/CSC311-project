import sqlite3
import pandas as pd

new_train_data = pd.read_csv('new_train_data.csv')
new_valid_data = pd.read_csv('new_valid_data.csv')
new_test_data = pd.read_csv('new_test_data.csv')
train_data = pd.read_csv('train_data.csv')

# new_train_data = new_train_data.drop(new_train_data.columns[0], axis=1)
# new_valid_data = new_valid_data.drop(new_valid_data.columns[0], axis=1)
# new_test_data = new_test_data.drop(new_test_data.columns[0], axis=1)

sum = 0
for new_train_data in (new_train_data, new_valid_data, new_test_data):
    # Identifying columns (assuming 'correctness' is the second to last and 'user_id' is the last column)
    user_id_col = new_train_data.columns[-2]
    correctness_col = new_train_data.columns[-1]
    subject_cols = new_train_data.columns[1:-2]  # All other columns are subject_id columns

    # Melting the DataFrame so each row contains a single subject_id for a user
    melted_data = new_train_data.melt(id_vars=[user_id_col, correctness_col], value_vars=subject_cols, var_name='subject_id', value_name='answered')

    # We only need rows where a subject was actually answered
    # answered_data = melted_data[melted_data['answered'] == 1]

    # Calculating accuracy per user per subject
    accuracy_data = melted_data.groupby([user_id_col, 'subject_id'])[correctness_col].mean().reset_index()

    # Pivoting to get the desired matrix format: users as rows and subjects as columns
    accuracy_matrix = accuracy_data.pivot(index=user_id_col, columns='subject_id', values=correctness_col)

    # Filling missing values with 0.5
    accuracy_matrix.fillna(0.5, inplace=True)

    # Optionally, you might want to ensure that the subjects are in the correct order, you can sort the columns based on their name
    accuracy_matrix = accuracy_matrix.sort_index(axis=1)
    conn = sqlite3.connect(':memory:')
    accuracy_matrix.to_sql('accuracy_matrix', conn, index=False, if_exists='replace')
    train = load_train_sparse()
    query = ("""
            SELECT *
            FROM accuracy_matrix natural join train_data;
            """)

    accuracy_matrix.to_csv('{}.csv'.format(sum), index=False)
    sum += 1
