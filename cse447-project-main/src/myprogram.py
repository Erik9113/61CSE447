#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter, defaultdict
import json


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self):
        self.max_n = 5
        self.ngrams = {i: defaultdict(Counter) for i in range(1, self.max_n + 1)}
        self.context_totals = {i: Counter() for i in range(1, self.max_n + 1)}
        self.vocab = set()
        self.top3_chars = []
        self.unk_token = '<UNK>'
        self.unk_threshold = 5  # chars appearing less than this are rare
        self.rare_chars = set()  # populated during training
        self.lambdas = {5: 0.4, 4: 0.3, 3: 0.15, 2: 0.1, 1: 0.05}

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        data_path = os.path.join("src", "wiki_clean2.txt")
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
        unigrams_temp = Counter()
        for line in data:
            for c in line:
                unigrams_temp[c] += 1

        # Identify rare characters (appearing less than threshold times)
        self.rare_chars = {c for c, count in unigrams_temp.items() if count < self.unk_threshold}
        
        # Replace rare chars with UNK token in unigrams, bigrams, and trigrams
        for line in data:
            processed_line = [self.unk_token if c in self.rare_chars else c for c in line]
            for i in range(len(processed_line)):
                c = processed_line[i]
                self.vocab.add(c)
                
                for n in range(1, self.max_n + 1):
                    start_idx = i - (n - 1)
                    if start_idx < 0:
                        continue
                    context = tuple(processed_line[start_idx:i])
                    self.ngrams[n][context][c] += 1
                    self.context_totals[n][context] += 1

        # Get top 3 most common characters (excluding space and UNK)
        unigram_context = tuple()
        self.top3_chars = [c for c, _ in self.ngrams[1][unigram_context].most_common() 
                           if c != ' ' and c != self.unk_token][:3]

    def run_pred(self, data):
        preds = []
        for line in data:
            if not line:
                preds.append(''.join(self.top3_chars))
                continue

            # Convert rare characters to UNK token
            processed_line = [self.unk_token if c in self.rare_chars else c for c in line]
            
            # Try trigrams first (need at least 2 characters)
            # Fall back to bigrams if no trigram matches
            candidates = set()
            for n in range(1, self.max_n + 1):
                start_idx = len(processed_line) - (n - 1)
                if start_idx < 0:
                    continue
                context = tuple(processed_line[start_idx:])
                candidates.update(self.ngrams[n][context].keys())
            
            candidates.discard(' ')
            candidates.discard(self.unk_token)

            # Fall back to top 3 most common chars
            if not candidates:
                preds.append(''.join(self.top3_chars))
                continue

            scores = []
            for c in candidates:
                score = 0
                for n in range(1, self.max_n + 1):
                    start_idx = len(processed_line) - (n - 1)
                    if start_idx < 0:
                        continue
                    context = tuple(processed_line[start_idx:])
                    count_c = self.ngrams[n][context][c]
                    total = self.context_totals[n][context]
                    prob = count_c / total if total > 0 else 0
                    score += self.lambdas[n] * prob
                scores.append((score, c))

            scores.sort(key=lambda x: x[0], reverse=True)
            guess_string = ''.join([char for score, char in scores[:3]])
            
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
        export_ngrams = {str(n): {"".join(ctx): dict(counter) for ctx, counter in self.ngrams[n].items()} for n in self.ngrams}
        export_totals = {str(n): {"".join(ctx): total for ctx, total in self.context_totals[n].items()} for n in self.context_totals}
        
        checkpoint = {
            "ngrams": export_ngrams,
            "totals": export_totals,
            "top3_chars": self.top3_chars,
            "rare_chars": list(self.rare_chars),
            "unk_token": self.unk_token,
            "unk_threshold": self.unk_threshold,
            "max_n": self.max_n
        }

        with open(os.path.join(work_dir, 'model.checkpoint'), 'w', encoding="utf-8") as f:
            json.dump(checkpoint, f)

    @classmethod
    def load(cls, work_dir):
        model = cls()

        with open(os.path.join(work_dir, 'model.checkpoint'), encoding="utf-8") as f:
            checkpoint = json.load(f)

        model.max_n = checkpoint.get("max_n", 5)
        model.top3_chars = checkpoint["top3_chars"]
        model.unk_token = checkpoint.get("unk_token", '<UNK>')
        model.unk_threshold = checkpoint.get("unk_threshold", 5)
        model.rare_chars = set(checkpoint.get("rare_chars", []))
        
        # Trigram key is 3 chars: ppc + pc + c
        for n_str, contexts in checkpoint["ngrams"].items():
            n = int(n_str)
            for ctx_str, counter_dict in contexts.items():
                model.ngrams[n][tuple(ctx_str)] = Counter(counter_dict)
                
        for n_str, contexts in checkpoint["totals"].items():
            n = int(n_str)
            for ctx_str, total in contexts.items():
                model.context_totals[n][tuple(ctx_str)] = total

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