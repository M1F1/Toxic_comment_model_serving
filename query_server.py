from utils import load_data, prepare_texts, query_for_answers

if __name__ == '__main__':
    # TODO: add argparse -   path to file, path to stopwords, create virtual env and requirements config
    texts, labels = load_data()
    texts = prepare_texts(texts)
    # SERVER_URL = 'http://localhost:8501/v1/models/cnn:predict'
    # google cloud kubernetes cluster ip address with docker image
    print('Connecting with gcp...')
    SERVER_URL = 'http://35.234.121.157:8501/v1/models/cnn:predict'
    acc = query_for_answers(texts, labels, SERVER_URL)
    print('Accuracy for data test set:')
    print(acc)
