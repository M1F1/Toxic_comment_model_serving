import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import clean_string

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train.csv'))
    df['toxic'] = df.iloc[:, 2:].any(axis='columns').astype(int)
    del df['id']
    df = df.iloc[:, 0:2]
    print(df.head())

    # data_preprocessing
    stopwords_path = os.path.join(os.getcwd(), 'data', 'stopwords.txt')
    df['comment_text'] = df['comment_text'].apply(lambda x: clean_string(x, stopwords_path))

    # save processed dataframe
    df.to_csv(os.path.join(os.getcwd(), 'data', 'processed_data.csv'))
    print(df.head()['comment_text'])

    # show classes histogram
    df.hist(column='toxic')
    plt.show()

