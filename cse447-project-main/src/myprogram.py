#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter
import json


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self):
        self.unigrams = Counter()
        self.vocab = set()
        self.top3_chars = []

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        data_path = os.path.join("data", "wiki_clean.txt")
        data = []

        if not os.path.exists(data_path):
            print("Training data not found at", data_path)
            return []

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    data.append(line)

        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                # inp = line[:-1]  # the last character is a newline
                # data.append(inp)
                data.append(line.rstrip("\n"))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        for line in data:
            for c in line:
                self.unigrams[c] += 1
                self.vocab.add(c)

        self.top3_chars = [c for c, _ in self.unigrams.most_common(3)]  
    
        print("Training complete.")
        print("Vocab size:", len(self.vocab))
        print("Top 3 characters:", self.top3_chars)


    def run_pred(self, data):
        # preds = []
        # all_chars = string.ascii_letters
        # for inp in data:
        #     # this model just predicts a random character each time
        #     top_guesses = [random.choice(all_chars) for _ in range(3)]
        #     preds.append(''.join(top_guesses))
        # return preds
        preds = []
        guess_string = ''.join(self.top3_chars)

        if len(guess_string) < 3:
            guess_string += "   "
            guess_string = guess_string[:3]

        for _ in data:
            preds.append(guess_string)

        return preds

    def save(self, work_dir):
        checkpoint = {
            "unigrams": dict(self.unigrams),
            "top3_chars": self.top3_chars
        }

        with open(os.path.join(work_dir, 'model.checkpoint'), 'w', encoding="utf-8") as f:
            json.dump(checkpoint, f)

    @classmethod
    def load(cls, work_dir):
        model = cls()

        with open(os.path.join(work_dir, 'model.checkpoint'), encoding="utf-8") as f:
            checkpoint = json.load(f)

        model.unigrams = Counter(checkpoint["unigrams"])
        model.top3_chars = checkpoint["top3_chars"]

        return model

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
