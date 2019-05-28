from utils import load_data, prepare_texts, query_for_answers

if __name__ == '__main__':
    # TODO: add argparse -   path to file, path to stopwords
    # create virtual env and requirements
    texts, labels = load_data()
    texts = prepare_texts(texts)
    SERVER_URL = 'http://localhost:8501/v1/models/cnn:predict'
    acc = query_for_answers(texts, labels, SERVER_URL)
    print(acc)
