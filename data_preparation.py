import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train.csv'))
df['toxic'] = df.iloc[:, 2:].any(axis='columns').astype(int)
del df['id']
df = df.iloc[:, 0:2]
df.to_csv(os.path.join(os.getcwd(), 'data', 'new_labels_train.csv'))
print(df.columns)
print(df.head()['comment_text'])
print(df.iloc[0])
print(df)
