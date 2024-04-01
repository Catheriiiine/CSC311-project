import sqlite3
import pandas as pd
import ast

train_data = pd.read_csv('train_data.csv')
question_meta = pd.read_csv('question_meta.csv')
subject_meta = pd.read_csv('subject_meta.csv')
student_meta = pd.read_csv('student_meta.csv')
valid_data = pd.read_csv('valid_data.csv')
test_data = pd.read_csv('test_data.csv')

conn = sqlite3.connect(':memory:')
train_data.to_sql('train_data', conn, index=False, if_exists='replace')
question_meta.to_sql('question_meta', conn, index=False, if_exists='replace')
subject_meta.to_sql('subject_meta', conn, index=False, if_exists='replace')
student_meta.to_sql('student_meta', conn, index=False, if_exists='replace')
valid_data.to_sql('valid_data', conn, index=False, if_exists='replace')
test_data.to_sql('test_data', conn, index=False, if_exists='replace')

# # Convert the string representation of lists in 'subject_id' to actual lists
# question_meta['subject_id'] = question_meta['subject_id'].apply(ast.literal_eval)
#
# # Explode the 'subject_id' list into separate rows
# exploded_question_meta = question_meta.explode('subject_id')
#
# # Convert 'subject_id' in exploded_question_meta to int for proper comparison
# exploded_question_meta['subject_id'] = exploded_question_meta['subject_id'].astype(int)
#
# subject_meta['subject_id'] = subject_meta['subject_id'].astype(int)
#
# # Create a flag in exploded_question_meta to indicate presence
# exploded_question_meta['present'] = 1
#
# all_combinations = (
#     question_meta[['question_id']]
#     .assign(key=1)
#     .merge(subject_meta.assign(key=1), how='outer', on='key')
#     .drop('key', axis=1)
#     .rename(columns={'subject_id': 'sid'})
# )
#
# # Mark with 1 where a subject_id is associated with a question_id, else 0
# all_combinations['value'] = all_combinations.apply(lambda x: 1 if x['sid'] in question_meta.loc[question_meta['question_id'] == x['question_id'], 'subject_id'].values[0] else 0, axis=1)
#
#
# # Merge all_combinations with exploded_question_meta to mark presence
# optimized_combinations = all_combinations.merge(
#     exploded_question_meta[['question_id', 'subject_id', 'present']],
#     left_on=['question_id', 'sid'],
#     right_on=['question_id', 'subject_id'],
#     how='left'
# ).drop(['subject_id'], axis=1).fillna(0)
#
# # Pivot this optimized combination DataFrame
# optimized_pivoted_table = optimized_combinations.pivot_table(index='question_id', columns='sid', values='present', fill_value=0).reset_index()
# if 'sid' in optimized_pivoted_table.columns:
#     optimized_pivoted_table = optimized_pivoted_table.drop(columns=['sid'])
# # Show the first few rows of the optimized pivoted table
# print(optimized_pivoted_table.head())
# optimized_pivoted_table.to_csv('question_subject.csv', index=False)
question_subject = pd.read_csv('question_subject.csv')
question_subject.to_sql('question_subject', conn, index=False, if_exists='replace')


query = ("""
        SELECT *
        FROM question_subject q join train_data t on q.question_id = t.question_id;
        """)

df_query_result = pd.read_sql_query(query, conn)

# Process the query result
print(df_query_result.head())

# Close the connection
conn.close()