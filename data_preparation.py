import pandas as pd
import os
import matplotlib.pyplot as plt
import re


def clean_string(s,
                 stopwords_path,
                 to_lower=True,
                 replace_new_line=True,
                 remove_special_signs=True,
                 remove_digits=True,
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
    if remove_digits:
        s = re.sub(r"[0-9]+", '', s)
    return s


if __name__ == '__main__':
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

    # data_preprocessing
    stopwords_path = os.path.join(os.getcwd(), 'stopwords.txt')
    df['comment_text'] = df['comment_text'].apply(lambda x: clean_string(x, stopwords_path))

    # save processed dataframe
    df.to_csv(os.path.join(os.getcwd(), 'data', 'processed_data.csv'))
    print(df.head()['comment_text'])

    # show classes histogram
    df.hist(column='toxic')
    plt.show()

