import pandas as pd
import os
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
# stopwords = set(stopwords.words('english'))
# print(stopwords)
# with open('stopwords.txt', 'w') as f:
#     for item in stopwords:
#         f.write("%s\n" % item)

df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train.csv'))
df['toxic'] = df.iloc[:, 2:].any(axis='columns').astype(int)
del df['id']
df = df.iloc[:, 0:2]

# get balanced distribution
df_positive = df.loc[df['toxic'] == 1]
positive_count = df_positive.shape[0]
df_negative = df.loc[df['toxic'] == 0][:positive_count]
df = pd.concat([df_positive, df_negative])
print(df.head())


def clean_string(s,
                 stopwords_path,
                 to_lower=True,
                 replace_new_line=True,
                 remove_special_signs=True,
                 remove_stopwords=True):
    if to_lower:
        s = s.lower()
    if replace_new_line:
        s = s.replace('\n', '')
    if remove_special_signs:
        s = " ".join(re.findall(r"[a-zA-Z0-9]+", s))
    if remove_stopwords:
        with open(stopwords_path, 'r') as f:
            stopwords = [line.rstrip('\n') for line in f]
        s = ' '.join([word for word in s.split() if word not in stopwords])
    return s


stopwords_path = os.path.join(os.getcwd(), 'stopwords.txt')
# data_preprocessing
df['comment_text'] = df['comment_text'].apply(lambda x: clean_string(x, stopwords_path))

df.to_csv(os.path.join(os.getcwd(), 'data', 'new_labels_train.csv'))
print(df.head()['comment_text'])

df.hist(column='toxic')
plt.show()

