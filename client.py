from utils import load_data, prepare_texts, query_for_answers
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Client for querying docker image hosted on specified server')
    parser.add_argument('test_path', type=str, help='path for test.txt file')
    parser.add_argument('-s', '--separator', default=';', help='separator in test.txt file')
    parser.add_argument('-u', '--server_url', default='http://35.234.121.157:8501/v1/models/cnn:predict',
                        help='url to hosted docker image with trained model, served with tensorflow serving on GCP')

    args = parser.parse_args()
    texts, labels = load_data(filename=args.test_path, separator=args.separator)
    texts = prepare_texts(texts)
    print('Connecting with gcp...')
    # google cloud kubernetes cluster ip address with docker image
    SERVER_URL = args.server_url
    acc = query_for_answers(texts, labels, SERVER_URL)
    print('Accuracy for data test set:')
    print(acc)
