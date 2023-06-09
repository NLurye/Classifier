import pandas as pd
import numpy as np
from enum import Enum


class A(Enum):
    pclass = 0
    age = 1
    sex = 2
    survived = 3


class Value:
    def __init__(self, value):
        self.value = value
        self.count = 0


def parse_data(txt):
    with open(txt, 'r') as file:
        lines = file.readlines()
    data = []
    header = lines[0].split('\t')
    for line in lines[1:]:
        values = line.split('\t')
        data.append(dict(zip(header, values)))
    return len(data), data


def get_entropy(pos_examples, neg_examples):
    print(899)


def count_examples(feature, options, examples):
    counters = [0] * len(options)
    for example in examples:
        for index, option in enumerate(options):
            if example[feature] == option:
                counters[index] += 1
                continue
    return counters

#  TODO: extract possible values no hardcode
if __name__ == '__main__':
    file_path = 'train.txt'
    ttl_examples, train_examples = parse_data(file_path)
    age_count = count_examples('age', ("adult", "child"), train_examples)
    p = 0
