import sqlite3
import pandas as pd
import ast

train_data = pd.read_csv('train_data.csv')
question_meta = pd.read_csv('question_meta.csv')
subject_meta = pd.read_csv('subject_meta.csv')
valid_data = pd.read_csv('valid_data.csv')
test_data = pd.read_csv('test_data.csv')

conn = sqlite3.connect(':memory:')

train_data.to_sql('train_data', conn, index=False, if_exists='replace')
question_meta.to_sql('question_meta', conn, index=False, if_exists='replace')
subject_meta.to_sql('subject_meta', conn, index=False, if_exists='replace')
valid_data.to_sql('valid_data', conn, index=False, if_exists='replace')
test_data.to_sql('test_data', conn, index=False, if_exists='replace')

question_subject = pd.read_csv('question_subject.csv')
question_subject.to_sql('question_subject', conn, index=False, if_exists='replace')


query1 = ("""
        SELECT *
        FROM question_subject natural join train_data t;
        """)

query2 = ("""
        SELECT *
        FROM question_subject natural join valid_data t;
        """)

query3 = ("""
        SELECT *
        FROM question_subject natural join test_data t;
        """)

train_data = pd.read_sql_query(query1, conn)
valid_data = pd.read_sql_query(query2, conn)
test_data = pd.read_sql_query(query3, conn)

# Process the query result
print(train_data.head())
print(valid_data.head())
print(test_data.head())

train_data.to_csv('new_train_data.csv', index=False)
valid_data.to_csv('new_valid_data.csv', index=False)
test_data.to_csv('new_test_data.csv', index=False)


# Close the connection
conn.close()