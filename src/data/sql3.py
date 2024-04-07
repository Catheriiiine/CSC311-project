import csv
import sqlite3
import pandas as pd

train_data = pd.read_csv('train_data.csv')
valid_data = pd.read_csv('valid_data.csv')
test_data = pd.read_csv('test_data.csv')

all_question_ids = pd.Series(pd.concat([train_data['question_id'], valid_data['question_id'], test_data['question_id']]).unique()).sort_values()

pivoted_data = (train_data.pivot(index='user_id', columns='question_id', values='is_correct'))
pivoted_data = train_data.pivot(index='user_id', columns='question_id', values='is_correct')

pivoted_data.reset_index(inplace=True)

pivoted_data.to_csv('question_correctness.csv', index=False)

pivoted_data1 = (valid_data.pivot(index='user_id', columns='question_id', values='is_correct'))
pivoted_data1 = pivoted_data1.reindex(columns=all_question_ids)

pivoted_data1.reset_index(inplace=True)

pivoted_data1.to_csv('question_correctness1.csv', index=False)

pivoted_data2 = (test_data.pivot(index='user_id', columns='question_id', values='is_correct'))
pivoted_data2 = pivoted_data2.reindex(columns=all_question_ids)

pivoted_data2.reset_index(inplace=True)

pivoted_data2.to_csv('question_correctness2.csv', index=False)
