import argparse
import logging
import os
import pdb
import sys
import traceback
import pickle
import json
from embedding import Embedding
from preprocessor import Preprocessor

def main(args):
    file_path = args.dest_dir
    data_path = args.data_path

    preprocessor = Preprocessor(None)

    # load embedding only for words in the data
    logging.info('loading embedding from {}embedding.pkl'.format(data_path))
    with open(os.path.join(data_path, 'embedding.pkl'), 'rb') as f:
        embedding = pickle.load(f)

    # update embedding used by preprocessor
    preprocessor.embedding = embedding

    # test
    logging.info('Processing data from {}'.format(file_path))
    test = preprocessor.get_dataset(
        file_path, args.n_workers,
        {'n_positive': -1, 'n_negative': -1, 'shuffle': False}
    )
    test_pkl_path = os.path.join(data_path, 'test.pkl')
    logging.info('Saving test to {}'.format(test_pkl_path))
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('dest_dir', type=str,
                        help='[input] Path of test file')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--data_path', type=str, 
                        help='[input] Path to the directory that .')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
