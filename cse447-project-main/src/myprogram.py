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
        self.bigrams = Counter()
        self.trigrams = Counter()  # (prev_prev_char, prev_char) -> next_char
        self.vocab = set()
        self.top3_chars = []
        self.unk_token = '<UNK>'
        self.unk_threshold = 5  # chars appearing less than this are rare
        self.rare_chars = set()  # populated during training

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        data_path = os.path.join("src", "wiki_clean.txt")
        # data_path = os.path.join("example", "input.txt") 
        # Creating a bigram mapping on example input to test program without data.
        # Success rate: 0.6923
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
        # First pass: count character frequencies and build n-grams
        for line in data:
            prev_prev_char = None
            prev_char = None
            for c in line:
                self.unigrams[c] += 1
                self.vocab.add(c)
                if prev_char is not None:
                    self.bigrams[(prev_char, c)] += 1
                if prev_prev_char is not None and prev_char is not None:
                    self.trigrams[((prev_prev_char, prev_char), c)] += 1
                prev_prev_char = prev_char
                prev_char = c

        # Identify rare characters (appearing less than threshold times)
        self.rare_chars = {c for c, count in self.unigrams.items() if count < self.unk_threshold}
        
        # Replace rare chars with UNK token in unigrams, bigrams, and trigrams
        if self.rare_chars:
            # Update unigrams by merging rare char counts into UNK token
            new_unigrams = Counter()
            for c, count in self.unigrams.items():
                if c in self.rare_chars:
                    new_unigrams[self.unk_token] += count
                else:
                    new_unigrams[c] += count
            self.unigrams = new_unigrams
            
            # Update bigrams by replacing rare chars with UNK token
            new_bigrams = Counter()
            for (pc, c), count in self.bigrams.items():
                new_pc = self.unk_token if pc in self.rare_chars else pc
                new_c = self.unk_token if c in self.rare_chars else c
                new_bigrams[(new_pc, new_c)] += count
            self.bigrams = new_bigrams
            
            # Update trigrams by replacing rare chars with UNK token
            new_trigrams = Counter()
            for ((ppc, pc), c), count in self.trigrams.items():
                new_ppc = self.unk_token if ppc in self.rare_chars else ppc
                new_pc = self.unk_token if pc in self.rare_chars else pc
                new_c = self.unk_token if c in self.rare_chars else c
                new_trigrams[((new_ppc, new_pc), new_c)] += count
            self.trigrams = new_trigrams

        # Get top 3 most common characters (excluding space and UNK)
        self.top3_chars = [c for c, _ in self.unigrams.most_common() if c != ' ' and c != self.unk_token][:3]


    def run_pred(self, data):
        preds = []
        for line in data:
            if not line:
                preds.append(''.join(self.top3_chars))
                continue

            # Try trigrams first (need at least 2 characters)
            next_chars = []
            if len(line) >= 2:
                prev_prev_char = line[-2]
                prev_char = line[-1]
                # Convert rare characters to UNK token
                prev_prev_char = self.unk_token if prev_prev_char in self.rare_chars else prev_prev_char
                prev_char = self.unk_token if prev_char in self.rare_chars else prev_char
                
                next_chars = [c for ((ppc, pc), c), _ in self.trigrams.most_common() 
                             if ppc == prev_prev_char and pc == prev_char and c != ' ' and c != self.unk_token]
            
            # Fall back to bigrams if no trigram matches
            if not next_chars:
                prev_char = line[-1]
                prev_char = self.unk_token if prev_char in self.rare_chars else prev_char
                next_chars = [c for (pc, c), _ in self.bigrams.most_common() 
                             if pc == prev_char and c != ' ' and c != self.unk_token]
            
            # Fall back to top 3 most common chars
            if next_chars:
                guess_string = ''.join(next_chars[:3])
            else:
                guess_string = ''.join(self.top3_chars)
                
            # Pad with top3 chars if needed
            if len(guess_string) < 3:
                for c in self.top3_chars:
                    if c not in guess_string and c != ' ' and c != self.unk_token:
                        guess_string += c
                    if len(guess_string) == 3:
                        break

            preds.append(guess_string)

        return preds


    def save(self, work_dir):
        checkpoint = {
            "unigrams": dict(self.unigrams),
            "bigrams": {f"{pc}{c}": count for (pc, c), count in self.bigrams.items()},
            "trigrams": {f"{ppc}{pc}{c}": count for ((ppc, pc), c), count in self.trigrams.items()},
            "top3_chars": self.top3_chars,
            "rare_chars": list(self.rare_chars),
            "unk_token": self.unk_token,
            "unk_threshold": self.unk_threshold
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
        model.unk_token = checkpoint.get("unk_token", '<UNK>')
        model.unk_threshold = checkpoint.get("unk_threshold", 5)
        model.rare_chars = set(checkpoint.get("rare_chars", []))
        
        model.bigrams = Counter()
        for k, v in checkpoint["bigrams"].items():
            model.bigrams[(k[0], k[1])] = v
        
        model.trigrams = Counter()
        for k, v in checkpoint.get("trigrams", {}).items():
            # Trigram key is 3 chars: ppc + pc + c
            model.trigrams[((k[0], k[1]), k[2])] = v

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
